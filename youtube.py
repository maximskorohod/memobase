"""YouTube ingestion ladder for memobase.

Final role split (decided/verified live on 2026-07-06):

  * **Channel video listing** -> ScrapeCreators ``/v1/youtube/channel-videos``
    (paginated via a continuation token, 30 videos/page) is PRIMARY;
    Apify ``streamers/youtube-channel-scraper`` is the fallback if
    ScrapeCreators errors (not installed/configured, network failure, or a
    non-2xx response) — not for "channel has zero videos", which is a valid
    empty result, not a failure.
  * **Transcripts** -> ``memobase.youtube.transcript_providers`` (default
    ``["scrapecreators", "apify"]``) is an auto-failover ORDER over
    PROVIDER-LEVEL failures (a provider erroring/unavailable) — a provider
    cleanly reporting "no captions for this video" is a terminal, definitive
    signal (not a reason to try the next provider for the same video), and
    routes to the audio+STT fallback instead. See :func:`get_transcript`.
  * **Audio fallback** (no captions from either transcript provider) ->
    Apify ``lurkapi/youtube-to-mp3-audio-downloader`` (pay-on-success) ->
    the downloaded audio goes through ``stt.py``'s Groq/Gemini ladder.

Every chunk built from a transcript/STT segment carries its timecode as
``block["page"]`` in the exact shape ``"?t=<sec>s"`` (a ready-to-append
YouTube URL query fragment) — this flows through chunk.py's existing
``page_or_timecode`` field unmodified (chunk.py already picks the first
non-None ``page`` value across a chunk's units), so no chunk.py/ingest.py
schema change was needed for video timecodes.

Cost note: channel listing + transcript credits (ScrapeCreators: 1
credit/request; Apify: priced per the actor's own rate card) are
comparatively cheap; the audio+STT fallback is the expensive path. Big
channels are gated the same way ``ingest.py`` gates a large local corpus —
see :func:`estimate_channel_cost_usd` and ``ingest_channel``'s own
confirm-over-video-count threshold.

Caveat, UPDATED 2026-07-06 after live integration probes against the real
ScrapeCreators API (3 real calls: channel-videos param discovery, a second
channel-videos call to confirm pagination, one video/transcript call):

  * ``/v1/youtube/video/transcript`` -- LIVE-VERIFIED: ``{"url": <watch
    url>, "language": "ru"}`` is correct as originally written. However its
    ``transcript[].startMs``/``endMs`` come back as STRINGS (e.g. ``"280"``),
    not numbers -- :func:`transcript_scrapecreators` now casts them; the
    original unconditional ``/ 1000.0`` raised ``TypeError`` on every real
    call before this fix.
  * ``/v1/youtube/channel-videos`` -- the original ``{"url": channel}`` was
    WRONG: the live API rejects it with HTTP 400
    ``{"error": "missing_parameter", "message": "You must provide a handle
    or a channelId"}``. Fixed to use :func:`_channel_query_param` (sends
    ``handle`` or ``channelId``, never ``url``). Continuation is via a
    ``continuationToken`` (camelCase) REQUEST param, not the previously
    assumed ``continuation_token`` -- confirmed live: the wrong param name
    silently returns page 1 again instead of erroring, which would have
    made channel ingestion loop forever re-appending the same 30 videos.
    Response video objects use ``lengthSeconds``/``durationMs`` and
    ``publishedTime``/``publishDate``, not ``duration``/``publishedAt`` --
    fixed via :func:`_sc_video_entry`. The response also splits results
    across separate ``videos``/``shorts``/``lives`` arrays; only ``videos``
    was read before, silently returning an empty list for Shorts-heavy
    channels -- now both ``videos`` and ``shorts`` are included.

Apify's actors (``channel_videos_apify``/``transcript_apify``/
``download_audio_apify``) remain UNVERIFIED against the live API (no Apify
credits were spent on the required-minimal-probes budget for this round) --
their field-name assumptions carry the same "best-effort reconstruction"
caveat the ScrapeCreators ones used to. Re-verify those the same way
(isolated in their own small functions specifically so that's cheap) before
depending on the Apify fallback path in production.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from . import ledger
from .secrets import get_env
from .security import DEFAULT_USER_AGENT, describe_request_error, safe_get

logger = logging.getLogger("memobase.youtube")

SCRAPECREATORS_BASE = "https://api.scrapecreators.com"
APIFY_BASE = "https://api.apify.com/v2"

APIFY_CHANNEL_ACTOR = "streamers~youtube-channel-scraper"
APIFY_TRANSCRIPT_ACTOR = "supreme_coder~youtube-transcript-scraper"
APIFY_AUDIO_ACTOR = "lurkapi~youtube-to-mp3-audio-downloader"

DEFAULT_TRANSCRIPT_PROVIDERS = ["scrapecreators", "apify"]
DEFAULT_MAX_CHANNEL_VIDEOS = 200  # memobase.youtube.max_videos — hard cap per channel ingest
SC_PAGE_SIZE = 30  # videos/page, per the live-verified probe

DEFAULT_TIMEOUT_S = 30.0
MAX_RETRIES = 4
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# Apify PPR actor run cost estimates, $ per 1000 units — used only for the
# pre-ingest cost estimate shown to the owner, never for billing.
_APIFY_CHANNEL_LISTING_USD_PER_1000 = 0.50
_APIFY_TRANSCRIPT_USD_PER_1000 = 0.50
_APIFY_AUDIO_USD_PER_1000_MINUTES = 2.00
_SC_CREDIT_USD = 0.0  # 1000 free credits/once; treated as $0 marginal cost in the estimate


class YoutubeError(RuntimeError):
    """Raised when EVERY configured provider for a given step (listing,
    transcript ladder, audio download) fails. Callers (ingest.py via
    :func:`extract_video`) catch this and degrade to an honest
    ``skipped``-reason empty doc — matching extract.py's own "never raises
    out of the extract step" contract."""


# ---------------------------------------------------------------------------
# Video id / URL helpers
# ---------------------------------------------------------------------------

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_VIDEO_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/|embed/|live/))([A-Za-z0-9_-]{11})"
)


def parse_video_id(url_or_id: str) -> str:
    """Extract an 11-char YouTube video id from a URL, or return *url_or_id*
    unchanged if it already looks like one. Raises :class:`YoutubeError` if
    neither shape matches (programmer/caller error — a bad source string)."""
    s = (url_or_id or "").strip()
    if _VIDEO_ID_RE.match(s):
        return s
    m = _URL_VIDEO_ID_RE.search(s)
    if m:
        return m.group(1)
    raise YoutubeError(f"could not parse a YouTube video id out of {url_or_id!r}")


def video_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def is_channel_source(source: str) -> bool:
    """Heuristic: a channel URL/handle (``@handle``, ``/channel/UC...``,
    ``/c/...``, ``/user/...``) vs a single video URL/id. Used by
    ``ingest.py``/``tools.py`` to route ``memobase_ingest(source_type="youtube")``
    to :func:`ingest_channel` vs the single-video path."""
    s = (source or "").strip()
    if not s:
        return False
    if _VIDEO_ID_RE.match(s) or _URL_VIDEO_ID_RE.search(s):
        return False
    return True


_CHANNEL_ID_RE = re.compile(r"^UC[0-9A-Za-z_-]{20,}$")


def _channel_query_param(channel: str) -> Dict[str, str]:
    """Normalize a channel URL/handle/bare-id into the single query param
    ScrapeCreators' ``/v1/youtube/channel-videos`` actually accepts.

    LIVE-VERIFIED 2026-07-06 (see youtube.py module docstring): the endpoint
    rejects ``{"url": ...}`` outright with HTTP 400
    ``{"error": "missing_parameter", "message": "You must provide a handle
    or a channelId"}`` — it wants ``handle`` (e.g. ``"@vdud"``) or
    ``channelId`` (``UC...``), never a full URL.
    """
    s = (channel or "").strip()
    if not s:
        raise YoutubeError("empty channel identifier")
    if _CHANNEL_ID_RE.match(s):
        return {"channelId": s}
    # Pull @handle or /channel/UC... out of a full URL if one was given.
    m = re.search(r"youtube\.com/(@[\w.-]+)", s)
    if m:
        return {"handle": m.group(1)}
    m = re.search(r"youtube\.com/channel/(UC[0-9A-Za-z_-]{20,})", s)
    if m:
        return {"channelId": m.group(1)}
    if s.startswith("@"):
        return {"handle": s}
    # Bare name with no @ and not URL-shaped -- best-effort as a handle
    # (ScrapeCreators' own docs show handles both with and without the
    # leading "@"; this is a plugin-side judgment call, not verified against
    # every possible input shape).
    return {"handle": s if s.startswith("@") else f"@{s}"}


# ---------------------------------------------------------------------------
# HTTP helper (trusted provider APIs — ScrapeCreators/Apify — not
# security.safe_get's SSRF-guarded path, which is for arbitrary untrusted
# URLs; these are fixed, first-party API hosts, same posture as embed.py's
# own provider calls).
# ---------------------------------------------------------------------------


def _request_with_backoff(method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
                           params: Optional[Dict[str, Any]] = None, json_body: Optional[dict] = None,
                           timeout: float = DEFAULT_TIMEOUT_S, max_retries: int = MAX_RETRIES):
    import requests  # heavy/optional import kept local, per project convention

    req_headers = {"User-Agent": DEFAULT_USER_AGENT, **(headers or {})}
    attempt = 0
    while True:
        try:
            resp = requests.request(
                method, url, headers=req_headers, params=params, json=json_body, timeout=timeout
            )
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise YoutubeError(
                    f"request failed after {attempt} retries: {describe_request_error(url, exc)}"
                ) from exc
            time.sleep(min(2 ** attempt, 30))
            attempt += 1
            continue

        if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
            time.sleep(min(2 ** attempt, 30))
            attempt += 1
            continue
        return resp


# ---------------------------------------------------------------------------
# ScrapeCreators: channel listing
# ---------------------------------------------------------------------------


def _sc_api_key() -> str:
    key = get_env("SCRAPECREATORS_API_KEY")
    if not key:
        raise YoutubeError("SCRAPECREATORS_API_KEY not set in environment")
    return key


def _sc_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{SCRAPECREATORS_BASE}{path}"
    headers = {"x-api-key": _sc_api_key()}
    ledger.note_external_call("scrapecreators", "request", 1)
    resp = _request_with_backoff("GET", url, headers=headers, params=params)
    if resp.status_code != 200:
        logger.debug("ScrapeCreators %s failed: HTTP %s body=%s", path, resp.status_code, resp.text[:300])
        raise YoutubeError(f"ScrapeCreators {path} failed: HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError as exc:
        raise YoutubeError(f"ScrapeCreators {path} response was not JSON: {exc}") from exc


def _sc_video_entry(v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one ScrapeCreators ``videos``/``shorts`` list entry into this
    module's ``{video_id, url, title, published_at, duration_s}`` shape.

    LIVE-VERIFIED 2026-07-06 field names (real response sample, ``@vdud``):
    ``id``, ``lengthSeconds`` (int seconds; ``durationMs`` also present),
    ``publishedTime`` (ISO 8601; ``publishDate``/``publishedTimeText`` also
    present). The previous ``duration``/``duration_s``/``publishedAt``/
    ``published_at`` keys this used to read do not exist in the real
    response, so those two fields silently came back ``None`` always."""
    vid = v.get("id") or v.get("video_id")
    if not vid:
        return None
    duration_s = v.get("lengthSeconds")
    if duration_s is None and v.get("durationMs") is not None:
        duration_s = v["durationMs"] / 1000.0
    published_at = v.get("publishedTime") or v.get("publishDate")
    return {
        "video_id": vid,
        "url": v.get("url") or video_watch_url(vid),
        "title": v.get("title"),
        "published_at": published_at,
        "duration_s": duration_s,
    }


def channel_videos_scrapecreators(channel: str, *, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """Paginate ``/v1/youtube/channel-videos`` (``SC_PAGE_SIZE`` videos/page)
    following its continuation token until exhausted or *max_pages* is hit.
    Raises :class:`YoutubeError` on any request/parse failure — callers
    (``list_channel_videos``) treat that as "try the next provider", not
    "channel has 0 videos".

    Includes both the ``videos`` and ``shorts`` arrays of each page — a
    channel that posts primarily Shorts would otherwise come back looking
    like an empty channel, which is a silent-wrong-answer failure mode, not
    an honest one."""
    videos: List[Dict[str, Any]] = []
    token: Optional[str] = None
    pages = 0
    channel_param = _channel_query_param(channel)
    while True:
        params: Dict[str, Any] = dict(channel_param)
        if token:
            params["continuationToken"] = token
        payload = _sc_get("/v1/youtube/channel-videos", params)
        page_entries = list(payload.get("videos") or []) + list(payload.get("shorts") or [])
        for v in page_entries:
            entry = _sc_video_entry(v)
            if entry is not None:
                videos.append(entry)
        token = payload.get("continuationToken") or payload.get("continuation_token")
        pages += 1
        if not token or (max_pages is not None and pages >= max_pages):
            break
    return videos


def _apify_token() -> str:
    token = get_env("APIFY_TOKEN")
    if not token:
        raise YoutubeError("APIFY_TOKEN not set in environment")
    return token


def _apify_run_actor(actor_id: str, run_input: Dict[str, Any], *, poll_interval_s: float = 8.0,
                      timeout_s: float = 1800.0) -> List[Dict[str, Any]]:
    """Async run -> poll -> dataset items, following the standard Apify
    integration pattern (batches don't fit the 300s sync
    limit; poll every 5-10s; client-side timeout 30-60min; PPR actors don't
    charge for failed items so retries are free). Raises
    :class:`YoutubeError` on submit failure, run failure, or timeout."""
    # Token in the Authorization header, never in the query string (requests
    # embeds the full URL in its exception text).
    auth = {"Authorization": f"Bearer {_apify_token()}"}
    submit_url = f"{APIFY_BASE}/acts/{actor_id}/runs"
    resp = _request_with_backoff("POST", submit_url, headers=auth, json_body=run_input)
    if resp.status_code not in (200, 201):
        logger.debug("Apify %s submit failed: HTTP %s body=%s", actor_id, resp.status_code, resp.text[:300])
        raise YoutubeError(f"Apify actor {actor_id} run submission failed: HTTP {resp.status_code}")
    try:
        run = resp.json()["data"]
    except (ValueError, KeyError) as exc:
        raise YoutubeError(f"Apify actor {actor_id} run submission response malformed: {exc}") from exc

    run_id = run.get("id")
    if not run_id:
        raise YoutubeError(f"Apify actor {actor_id} run submission response missing id")

    deadline = time.monotonic() + timeout_s
    status_url = f"{APIFY_BASE}/actor-runs/{run_id}"
    dataset_id = run.get("defaultDatasetId")
    while True:
        resp = _request_with_backoff("GET", status_url, headers=auth)
        if resp.status_code != 200:
            raise YoutubeError(f"Apify run {run_id} status poll failed: HTTP {resp.status_code}")
        data = resp.json().get("data", {})
        status = data.get("status")
        dataset_id = data.get("defaultDatasetId") or dataset_id
        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise YoutubeError(f"Apify actor {actor_id} run {run_id} ended with status={status}")
        if time.monotonic() >= deadline:
            raise YoutubeError(f"Apify actor {actor_id} run {run_id} timed out after {timeout_s}s (client-side)")
        time.sleep(poll_interval_s)

    if not dataset_id:
        raise YoutubeError(f"Apify actor {actor_id} run {run_id} succeeded but has no defaultDatasetId")
    items_url = f"{APIFY_BASE}/datasets/{dataset_id}/items"
    resp = _request_with_backoff("GET", items_url, headers=auth, params={"format": "json"})
    if resp.status_code != 200:
        raise YoutubeError(f"Apify dataset {dataset_id} fetch failed: HTTP {resp.status_code}")
    try:
        items = resp.json()
    except ValueError as exc:
        raise YoutubeError(f"Apify dataset {dataset_id} response was not JSON: {exc}") from exc
    if not isinstance(items, list):
        raise YoutubeError(f"Apify dataset {dataset_id} response was not a list")
    return items


def channel_videos_apify(channel: str) -> List[Dict[str, Any]]:
    """Fallback channel listing via ``streamers/youtube-channel-scraper``."""
    items = _apify_run_actor(APIFY_CHANNEL_ACTOR, {"startUrls": [{"url": channel}]})
    ledger.note_external_call("apify", "channel_list", len(items))
    videos: List[Dict[str, Any]] = []
    for it in items:
        vid = it.get("id") or it.get("videoId")
        if not vid:
            continue
        videos.append(
            {
                "video_id": vid,
                "url": it.get("url") or video_watch_url(vid),
                "title": it.get("title"),
                "published_at": it.get("date") or it.get("publishedAt"),
                "duration_s": it.get("duration"),
            }
        )
    return videos


def list_channel_videos(
    channel: str, *, memobase_cfg: Optional[Dict[str, Any]] = None, max_pages: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Return ``(videos, provider_used)``. Tries ScrapeCreators first
    (primary, per the finalized role split); on ANY failure (missing key,
    network error, bad response) falls back to Apify's channel-scraper actor.
    Raises :class:`YoutubeError` only if BOTH fail.

    *max_pages* bounds how many ScrapeCreators pages (``SC_PAGE_SIZE`` videos
    each) are walked; ``None`` (the default, used by ``ingest_channel``) means
    "every page". The Apify fallback returns the whole channel in one run and
    ignores it — callers that need a hard cap truncate the result themselves
    (see ``sort_and_filter_videos``)."""
    try:
        videos = channel_videos_scrapecreators(channel, max_pages=max_pages)
        return videos, "scrapecreators"
    except Exception as exc:  # noqa: BLE001 - any ScrapeCreators failure triggers fallback
        logger.warning("ScrapeCreators channel listing failed for %r (%s); falling back to Apify", channel, exc)
    videos = channel_videos_apify(channel)
    return videos, "apify"


# ---------------------------------------------------------------------------
# Shared segment/date/duration helpers (used by the transcript parsers and by
# the tools.py YouTube wrappers). Pure, no I/O — unit-tested directly.
# ---------------------------------------------------------------------------


def _seg_start_seconds(seg: Dict[str, Any]) -> float:
    """Start time in SECONDS from a transcript segment across provider shapes:
    ``start`` (seconds — supreme_coder/pintostudio/karamelo) or ``startMs``
    (milliseconds — ScrapeCreators/some Apify actors). 0.0 if unparseable."""
    if seg.get("start") is not None:
        try:
            return float(seg["start"])
        except (TypeError, ValueError):
            return 0.0
    if seg.get("startMs") is not None:
        try:
            return float(seg["startMs"]) / 1000.0
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _seg_end_seconds(seg: Dict[str, Any], start_sec: float) -> Optional[float]:
    """End time in SECONDS across shapes: ``end`` (seconds — karamelo),
    ``endMs`` (milliseconds), or ``duration``/``dur`` (seconds — supreme_coder/
    pintostudio) added to *start_sec*. None if none are present."""
    if seg.get("end") is not None:
        try:
            return float(seg["end"])
        except (TypeError, ValueError):
            return None
    if seg.get("endMs") is not None:
        try:
            return float(seg["endMs"]) / 1000.0
        except (TypeError, ValueError):
            return None
    dur = seg.get("duration") if seg.get("duration") is not None else seg.get("dur")
    if dur is not None:
        try:
            return start_sec + float(dur)
        except (TypeError, ValueError):
            return None
    return None


def parse_since_date(since: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse a ``YYYY-MM-DD`` string into a ``datetime.date``. Returns
    ``(date, None)`` on success or ``(None, error_message)`` on malformed
    input (the tool layer surfaces the message to the caller)."""
    import datetime as _dt

    try:
        return _dt.datetime.strptime(since, "%Y-%m-%d").date(), None
    except (ValueError, TypeError):
        return None, f"Неверный формат даты since: «{since}». Ожидается YYYY-MM-DD."


def _parse_published_key(published_at: Any) -> Optional[Any]:
    """Best-effort parse of a video's ``published_at`` into a NAIVE
    ``datetime`` for sorting/filtering. ScrapeCreators ``publishedTime`` is ISO
    8601 (possibly tz-aware, possibly trailing 'Z'); Apify may give a date or
    relative text. Returns None for unparseable/relative strings ('2 days ago')
    — such videos sort last and are excluded from a ``since`` filter. tzinfo is
    stripped so every key compares as naive (mixed aware/naive would raise)."""
    import datetime as _dt

    if not published_at or not isinstance(published_at, str):
        return None
    s = published_at.strip()
    try:
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    try:
        return _dt.datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


def sort_and_filter_videos(
    videos: List[Dict[str, Any]], *, limit: int, since_date: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], bool]:
    """Sort *videos* newest-first by ``published_at``, optionally drop those
    older than *since_date*, and cap to *limit*. Returns ``(result,
    possibly_more)`` — ``possibly_more`` is True only when a ``since_date``
    filter was applied AND the oldest INPUT video is still on/after the cutoff
    (so the caller's page cap may have hidden more in-range videos). Videos with
    an unparseable date sort last and are excluded when *since_date* is set.
    Равные даты упорядочиваются по video_id (детерминированный tie-break)."""
    import datetime as _dt

    def key(v: Dict[str, Any]):
        return _parse_published_key(v.get("published_at")) or _dt.datetime.min

    # Stable tie-break: pre-sort by video_id so equal published_at (e.g. a
    # batch published in the same second) keeps a deterministic order.
    ordered = sorted(videos, key=lambda v: str(v.get("video_id") or ""))
    ordered = sorted(ordered, key=key, reverse=True)

    possibly_more = False
    if since_date is not None:
        cutoff = _dt.datetime(since_date.year, since_date.month, since_date.day)
        kept = [
            v for v in ordered
            if (_parse_published_key(v.get("published_at")) or _dt.datetime.min) >= cutoff
            and _parse_published_key(v.get("published_at")) is not None
        ]
        oldest = _parse_published_key(ordered[-1].get("published_at")) if ordered else None
        possibly_more = bool(ordered) and oldest is not None and oldest >= cutoff
        ordered = kept

    return ordered[:limit], possibly_more


def format_hms(seconds: Any) -> str:
    """Seconds -> ``mm:ss`` or ``h:mm:ss`` (used for both durations and
    transcript timecodes). Non-numeric/None/negative -> ``00:00``."""
    try:
        total = int(float(seconds or 0))
    except (TypeError, ValueError):
        total = 0
    if total < 0:
        total = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_duration(duration_s: Any) -> str:
    """Like :func:`format_hms` but returns ``?`` for unknown/None duration
    (a listing row with no duration must not read as ``00:00``)."""
    if duration_s is None:
        return "?"
    try:
        float(duration_s)
    except (TypeError, ValueError):
        return "?"
    return format_hms(duration_s)


# ---------------------------------------------------------------------------
# Transcript ladder
# ---------------------------------------------------------------------------


def transcript_scrapecreators(
    video_id: str, *, language: str = "ru", video_meta: Optional[Dict[str, Any]] = None, **_kw: Any
) -> Optional[List[Dict[str, Any]]]:
    """Return a list of ``{"text", "start_sec", "end_sec"}`` segments, or
    ``None`` if ScrapeCreators cleanly reports "no captions for this video"
    (a definitive, terminal result — not an error). Raises
    :class:`YoutubeError` for an actual request/parse failure.

    *video_meta*, if given: best-effort ``video_meta["title"] = payload["title"]``
    when this endpoint's response happens to carry one and the caller didn't
    already know a title (single-video case, see :func:`get_transcript`).
    LIVE-UNVERIFIED whether ``/v1/youtube/video/transcript`` actually returns a
    ``title`` field (module docstring) — absence is silently a no-op, never
    an error, and never worth a dedicated call of its own."""
    payload = _sc_get(
        "/v1/youtube/video/transcript", {"url": video_watch_url(video_id), "language": language}
    )
    if video_meta is not None and not video_meta.get("title") and payload.get("title"):
        video_meta["title"] = payload["title"]
    raw_segments = payload.get("transcript")
    if raw_segments is None:
        return None  # no captions for this video — terminal, not a provider error
    segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        # LIVE-VERIFIED 2026-07-06: startMs/endMs come back as STRINGS
        # (e.g. "280", not 280) in the real response, not numbers -- casting
        # is required before arithmetic or this raises TypeError on every
        # single real call.
        start_ms = seg.get("startMs")
        end_ms = seg.get("endMs")
        try:
            start_ms = float(start_ms) if start_ms is not None else 0.0
        except (TypeError, ValueError):
            start_ms = 0.0
        try:
            end_ms = float(end_ms) if end_ms is not None else None
        except (TypeError, ValueError):
            end_ms = None
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "start_sec": start_ms / 1000.0,
                "end_sec": (end_ms / 1000.0) if end_ms is not None else None,
            }
        )
    return segments


def transcript_apify(
    video_id: str, *, language: str = "ru", video_meta: Optional[Dict[str, Any]] = None, **_kw: Any
) -> Optional[List[Dict[str, Any]]]:
    """Fallback transcript source via ``supreme_coder/youtube-transcript-scraper``.
    Same return contract as :func:`transcript_scrapecreators`.

    LIVE-UNVERIFIED (no Apify credits spent — see module docstring). Field
    names are best-effort per the actor's public store page (2026-07): INPUT
    field is ``urls`` (NOT ``videoUrls`` — the previous value silently sent the
    actor nothing); each dataset item is one video whose segments carry
    ``text`` + ``start`` (seconds) + ``duration`` (seconds). The parser is
    deliberately tolerant of ``transcript``/``captions``/``segments`` wrapping
    keys and of ``start``/``startMs`` + ``end``/``endMs``/``duration`` so a
    provider swap (e.g. the documented karamelo reserve, which uses ``end``)
    needs no code change. ``language`` is forwarded as the actor's ``languages``
    preference (ignored by actors that don't support it).

    *video_meta*, if given: same best-effort ``["title"]`` side-channel as
    :func:`transcript_scrapecreators`, read from this item's own ``title``
    field (same key ``channel_videos_apify`` already reads for the
    channel-listing case) when present and the caller didn't already know one."""
    items = _apify_run_actor(
        APIFY_TRANSCRIPT_ACTOR,
        {"urls": [video_watch_url(video_id)], "languages": [language]},
    )
    ledger.note_external_call("apify", "transcript", 1)
    if not items:
        return None
    item = items[0]
    if video_meta is not None and not video_meta.get("title") and item.get("title"):
        video_meta["title"] = item["title"]
    raw_segments = item.get("transcript") or item.get("captions") or item.get("segments")
    if not raw_segments:
        return None
    segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = _seg_start_seconds(seg)
        end = _seg_end_seconds(seg, start)
        segments.append({"text": text, "start_sec": start, "end_sec": end})
    return segments


_TRANSCRIPT_FETCHERS = {
    "scrapecreators": transcript_scrapecreators,
    "apify": transcript_apify,
}


def get_transcript(
    video_id: str, *, memobase_cfg: Optional[Dict[str, Any]] = None, language: str = "ru",
    video_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Walk ``memobase.youtube.transcript_providers`` in configured order (default
    :data:`DEFAULT_TRANSCRIPT_PROVIDERS`). For each provider:

      * an EXCEPTION (network/API failure, missing credentials) -> log and
        try the NEXT provider (auto-failover, this is the smoke-tested
        contract);
      * a clean ``None`` return (provider explicitly says "no captions") ->
        STOP immediately and return ``(None, None)`` — this is a terminal
        signal for the whole ladder, not a reason to ask a different
        provider for the same non-existent captions; the caller
        (:func:`extract_video`) falls through to the audio+STT path.

    Returns ``(segments, provider_name)`` on success, or ``(None, None)``
    if no provider had captions, or raises :class:`YoutubeError` if EVERY
    configured provider raised (a real outage, not "no captions").

    *language* is passed through to each provider (providers that don't
    accept it ignore it).

    *video_meta*, if given, is an OPTIONAL output parameter: a fetcher that
    happens to find a title alongside the transcript in its own response body
    MAY set ``video_meta["title"]`` as a side effect — never overwriting a
    title the caller already knew (see :func:`extract_video`'s ``meta_hint``).
    This is a free look at a response already being fetched for the
    transcript itself, never an extra call, and never required (a fetcher/
    provider that doesn't carry a title just leaves it untouched).
    """
    providers = (memobase_cfg or {}).get("youtube", {}).get("transcript_providers") or DEFAULT_TRANSCRIPT_PROVIDERS
    last_exc: Optional[Exception] = None
    any_succeeded_as_provider = False
    for name in providers:
        fetcher = _TRANSCRIPT_FETCHERS.get(name)
        if fetcher is None:
            logger.warning("unknown youtube transcript provider %r in config; skipping", name)
            continue
        try:
            segments = fetcher(video_id, language=language, video_meta=video_meta)
            any_succeeded_as_provider = True
        except Exception as exc:  # noqa: BLE001 - provider-level failure -> try next
            logger.warning("transcript provider %r failed for video %s (%s); trying next", name, video_id, exc)
            last_exc = exc
            continue
        if segments is None:
            return None, None  # terminal: no captions, do not keep trying other providers
        return segments, name

    if not any_succeeded_as_provider and last_exc is not None:
        raise YoutubeError(f"all transcript providers failed for video {video_id}: {last_exc}")
    return None, None


# ---------------------------------------------------------------------------
# Audio fallback (no captions from either transcript provider)
# ---------------------------------------------------------------------------


def download_audio_apify(video_id: str) -> Tuple[bytes, str]:
    """Download this video's audio via ``lurkapi/youtube-to-mp3-audio-downloader``
    (pay-on-success). Returns ``(audio_bytes, suggested_filename)``. Raises
    :class:`YoutubeError` on any failure — caller (:func:`extract_video`)
    treats that as a final, honest "could not transcribe" skip reason."""
    items = _apify_run_actor(APIFY_AUDIO_ACTOR, {"videoUrls": [video_watch_url(video_id)]})
    if items:
        try:
            minutes = float(items[0].get("duration") or 0) / 60.0
        except (TypeError, ValueError):
            minutes = 0.0
        # Duration unknown up front: charge the same 12-minute assumption the
        # channel estimate uses.
        ledger.note_external_call("apify", "audio_download", minutes or 12.0)
    if not items:
        raise YoutubeError(f"Apify audio actor returned no items for video {video_id}")
    item = items[0]
    # CONFIRMED live 2026-07-15 (real actor JSON) -- the actual key is
    # `audioFileUrl`; the other three were never-verified guesses that never
    # matched a real response, so the whole STT audio fallback unconditionally
    # raised "has no audio URL" below. Kept as fallback keys in case the actor
    # changes shape again; module-level LIVE-UNVERIFIED caveat still applies
    # to the rest of this actor's assumptions.
    audio_url = (
        item.get("audioFileUrl") or item.get("mp3Url") or item.get("audioUrl") or item.get("downloadUrl")
    )
    if not audio_url:
        logger.debug("Apify audio actor item without audio URL: %r", item)
        raise YoutubeError(f"Apify audio actor result for video {video_id} has no audio URL")
    # The audio URL is a first-party Apify KV-store link, not owner-supplied
    # untrusted input, but security.safe_get's SSRF guard + size cap + retry/
    # backoff is exactly the right shared primitive for "download a file from
    # a URL" regardless of source, so it is reused here rather than
    # reimplemented.
    audio_bytes = safe_get(audio_url, max_bytes=500 * 1024 * 1024)  # audio files can be large; generous cap
    return audio_bytes, f"{video_id}.mp3"


# ---------------------------------------------------------------------------
# Cost estimate (pre-ingest confirm gate for a whole channel)
# ---------------------------------------------------------------------------


def estimate_channel_cost_usd(
    video_count: int, *, no_captions_fraction: float = 0.15, avg_video_minutes: float = 12.0
) -> Dict[str, Any]:
    """Rough $ estimate for ingesting a channel of *video_count* videos
    ("STT длинного видео стоит времени/
    денег -> входит в смету перед загрузкой"). ``no_captions_fraction`` is a
    conservative guess (most channels have captions on most videos) for how
    many videos will need the audio+STT fallback — shown as a labeled
    assumption, not hidden in the total.
    """
    listing_usd = (video_count / 1000.0) * _APIFY_CHANNEL_LISTING_USD_PER_1000
    transcript_usd = (video_count / 1000.0) * _APIFY_TRANSCRIPT_USD_PER_1000
    no_caption_videos = round(video_count * no_captions_fraction)
    audio_minutes = no_caption_videos * avg_video_minutes
    audio_usd = (audio_minutes / 1000.0) * _APIFY_AUDIO_USD_PER_1000_MINUTES
    stt_usd = (audio_minutes / 60.0) * 0.04  # Groq whisper-large-v3-turbo, $0.04/hour audio
    total = listing_usd + transcript_usd + audio_usd + stt_usd
    return {
        "video_count": video_count,
        "assumed_no_captions_fraction": no_captions_fraction,
        "estimated_no_caption_videos": no_caption_videos,
        "listing_usd": round(listing_usd, 4),
        "transcript_usd": round(transcript_usd, 4),
        "audio_download_usd": round(audio_usd, 4),
        "stt_usd": round(stt_usd, 4),
        "total_usd": round(total, 4),
        "note": (
            "ScrapeCreators channel listing/transcripts are effectively free within its "
            "1000 free credits; the $ figures above are the Apify-fallback ceiling."
        ),
    }


# ---------------------------------------------------------------------------
# Single-video doc building (extract.py-shaped) — ingest.py's entry point
# ---------------------------------------------------------------------------


def _segments_to_blocks(segments: List[Dict[str, Any]], *, section: Optional[str]) -> List[Dict[str, Any]]:
    blocks = []
    for seg in segments:
        sec = int(seg.get("start_sec") or 0)
        blocks.append(
            {
                "text": seg["text"],
                "page": f"?t={sec}s",
                "section": section,
                "is_code": False,
            }
        )
    return blocks


def build_video_doc(
    video_meta: Dict[str, Any], segments: List[Dict[str, Any]], *, provider: str
) -> Dict[str, Any]:
    """Build the extract.py-shaped ``{text, blocks, meta, skipped}`` doc for
    one video's transcript/STT segments. Every block's ``page`` carries the
    ``?t=<sec>s`` timecode fragment (see module docstring)."""
    title = video_meta.get("title") or video_meta.get("video_id")
    blocks = _segments_to_blocks(segments, section=title)
    text = "\n\n".join(b["text"] for b in blocks)
    return {
        "text": text,
        "blocks": blocks,
        "meta": {
            "title": title,
            "pages": None,
            "channel": video_meta.get("channel"),
            "published_at": video_meta.get("published_at"),
            "video_id": video_meta.get("video_id"),
            "transcript_provider": provider,
        },
        "skipped": [],
    }


def extract_video(
    source: str, *, memobase_cfg: Optional[Dict[str, Any]] = None, meta_hint: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Full single-video route: parse id -> transcript ladder -> (if no
    captions anywhere) audio download + STT fallback -> extract.py-shaped
    doc. NEVER raises — every failure becomes a ``skipped`` reason with
    empty text, exactly like extract.py's ``extract()`` contract, so
    ``ingest.py`` handles it identically regardless of source_type.

    *meta_hint* (``documents.title`` used to always come back as the bare
    video id): an OPTIONAL caller-known ``{"title": ...}`` hint —
    ``ingest_channel`` already has each video's title from the channel
    listing (:func:`_sc_video_entry`/``channel_videos_apify``) and passes it
    through here so the title doesn't need to be re-derived. When no hint is
    given (a single-video ingest with no channel listing behind it), a title
    found INCIDENTALLY in the transcript provider's own response (see
    :func:`get_transcript`'s ``video_meta`` out-param) is used instead — this
    never triggers an extra network call; only the request already made for
    the transcript itself is looked at. If neither source has a title,
    :func:`build_video_doc` falls back to the bare video id, same as before.
    """
    try:
        video_id = parse_video_id(source)
    except YoutubeError as exc:
        return {"text": "", "blocks": [], "meta": {"title": None, "pages": None}, "skipped": [{"reason": str(exc)}]}

    video_meta: Dict[str, Any] = {"video_id": video_id}
    if meta_hint and meta_hint.get("title"):
        video_meta["title"] = meta_hint["title"]

    try:
        segments, provider = get_transcript(video_id, memobase_cfg=memobase_cfg, video_meta=video_meta)
    except YoutubeError as exc:
        return {
            "text": "", "blocks": [], "meta": {"title": None, "pages": None},
            "skipped": [{"reason": f"all transcript providers failed: {exc}"}],
        }

    if segments:
        return build_video_doc(video_meta, segments, provider=provider or "unknown")

    # No captions anywhere -> audio + STT fallback.
    try:
        from . import stt as stt_mod

        audio_bytes, filename = download_audio_apify(video_id)
        stt_result = stt_mod.transcribe_long_audio(audio_bytes, filename_hint=filename, memobase_cfg=memobase_cfg)
    except Exception as exc:  # noqa: BLE001 - final fallback failure is an honest skip, not a crash
        return {
            "text": "", "blocks": [], "meta": {"title": None, "pages": None},
            "skipped": [{"reason": f"video has no captions and audio/STT fallback failed: {exc}"}],
        }

    stt_segments = stt_result.get("segments") or []
    if not stt_segments:
        return {
            "text": "", "blocks": [], "meta": {"title": None, "pages": None},
            "skipped": [{"reason": "video has no captions; STT fallback produced no segments"}],
        }
    doc = build_video_doc(video_meta, stt_segments, provider=f"stt:{stt_result.get('provider', 'unknown')}")
    if stt_result.get("trust") == "low":
        doc["skipped"].append({"reason": "STT timecodes are model-reported (Gemini fallback), lower trust"})
    return doc


# ---------------------------------------------------------------------------
# Whole-channel ingestion (multi-document orchestrator)
# ---------------------------------------------------------------------------


def ingest_channel(
    conn,
    collection_row: Dict[str, Any],
    channel: str,
    *,
    memobase_cfg: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
    uploader_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List every video on *channel*, cost-estimate + confirm-gate (per
    ``memobase.youtube.confirm_over_videos``, default 20 — deliberately separate
    from ``memobase.confirm_over_chunks``, since a channel's cost driver is video
    *count*, not chunk count, before any video has even been fetched), then
    ``ingest.ingest_source`` each video as its own document
    (source_type="youtube", source=video URL). Resumability: re-running
    this after a crash is cheap — already-ingested videos come back
    ``"unchanged"`` via ingest.py's own content-hash dedup (no separate
    watermark bookkeeping needed, see obsidian.py's ``ingest_vault`` for the
    same reasoning applied to Obsidian vaults).
    """
    from . import config as kb_config
    from . import db
    from . import ingest as ingest_mod

    if conn is None or not isinstance(collection_row, dict) or "id" not in collection_row:
        raise YoutubeError("ingest_channel requires a real conn and a valid collection_row")

    memobase_cfg = memobase_cfg if memobase_cfg is not None else kb_config.get_memobase_config_readonly()
    job_id = db.create_ingestion_job(conn, collection_id=collection_row["id"], kind="youtube_channel", stage="list")

    try:
        with ledger.collect_external_spend() as spend_bucket:
            try:
                videos, list_provider = list_channel_videos(channel, memobase_cfg=memobase_cfg)
            finally:
                listing_usd = ledger.flush_external_spend(
                    conn, spend_bucket, collection_id=collection_row["id"], user_id=uploader_user_id,
                )
                if uploader_user_id is not None and listing_usd:
                    db.record_guest_usage(conn, uploader_user_id, usd_spent=listing_usd)
    except YoutubeError as exc:
        db.update_ingestion_job(conn, job_id, status="failed", stage="failed")
        return {"status": "failed", "error": f"could not list channel videos: {exc}", "job_id": job_id}

    # Hard cap regardless of confirmation: one call can never enqueue an
    # unbounded number of paid per-video transcript/STT runs.
    max_videos = int((memobase_cfg.get("youtube") or {}).get("max_videos") or DEFAULT_MAX_CHANNEL_VIDEOS)
    videos_total_listed = len(videos)
    videos = videos[:max_videos]

    threshold = (memobase_cfg.get("youtube") or {}).get("confirm_over_videos", 20)
    if len(videos) > threshold and not confirm:
        estimate = estimate_channel_cost_usd(len(videos))
        db.update_ingestion_job(conn, job_id, status="done", stage="needs_confirmation", items_total=len(videos))
        return {
            "status": "needs_confirmation",
            "job_id": job_id,
            "video_count": len(videos),
            "list_provider": list_provider,
            "estimate": estimate,
            "videos_preview": [
                {"video_id": v.get("video_id"), "published_at": v.get("published_at"), "title": v.get("title")}
                for v in videos[:10]
            ],
            "message": (
                f"Канал «{channel}»: {len(videos)} видео (порог подтверждения: {threshold}). "
                f"Оценка: ${estimate['total_usd']:.4f}. Повторите запрос с подтверждением, чтобы продолжить."
            ),
        }

    db.update_ingestion_job(conn, job_id, stage="ingest_videos", items_total=len(videos))
    per_video: List[Dict[str, Any]] = []
    counts = {"done": 0, "unchanged": 0, "failed": 0}
    for i, v in enumerate(videos):
        video_title = v.get("title")
        result = ingest_mod.ingest_source(
            # The channel-level confirm above already covered the owner; a
            # guest keeps the per-video chunk threshold (no model-supplied
            # confirm can lift it).
            conn, collection_row, v["url"], "youtube", memobase_cfg=memobase_cfg,
            confirm=uploader_user_id is None, uploader_user_id=uploader_user_id,
            # ingest_channel's own channel listing (_sc_video_entry/
            # channel_videos_apify) already has this video's title -- pass it
            # through as a hint so documents.title doesn't degrade to the bare
            # video id.
            meta_hint={"title": video_title} if video_title else None,
        )
        status = result.get("status")
        counts[status] = counts.get(status, 0) + 1
        per_video.append(
            {
                "video_id": v.get("video_id"),
                "status": status,
                "title": v.get("title"),
                "published_at": v.get("published_at"),
                # Surface WHY a video failed -- per_video used to only carry
                # `status`, silently dropping ingest_source's own `error`
                # (e.g. a provider HTTP 402 "no credits" response was
                # previously invisible in the channel summary).
                "reason": result.get("error"),
                # Which transcript source actually won for this video
                # (scrapecreators/apify/stt:groq/stt:gemini) -- already
                # computed by build_video_doc/extract_media, just threaded
                # out past ingest_source here.
                "transcript_provider": result.get("transcript_provider"),
                # Reuses the SAME per-call embed-cost figure ingest_source
                # already computed for its own ledger write -- no extra query.
                "cost_estimate_usd": result.get("cost_estimate_usd"),
            }
        )
        db.update_ingestion_job(conn, job_id, items_done=i + 1)

    db.update_ingestion_job(conn, job_id, status="done", stage="done")
    total_cost = sum(r["cost_estimate_usd"] for r in per_video if r.get("cost_estimate_usd"))
    return {
        "status": "done",
        "job_id": job_id,
        "list_provider": list_provider,
        "video_count": len(videos),
        "videos_listed": videos_total_listed,
        "videos_capped": videos_total_listed > len(videos),
        "videos_done": counts.get("done", 0),
        "videos_unchanged": counts.get("unchanged", 0),
        "videos_failed": counts.get("failed", 0),
        "per_video": per_video,
        "total_cost_estimate_usd": round(total_cost, 6),
    }
