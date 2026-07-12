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

from .security import DEFAULT_USER_AGENT, safe_get

logger = logging.getLogger("memobase.youtube")

SCRAPECREATORS_BASE = "https://api.scrapecreators.com"
APIFY_BASE = "https://api.apify.com/v2"

APIFY_CHANNEL_ACTOR = "streamers~youtube-channel-scraper"
APIFY_TRANSCRIPT_ACTOR = "supreme_coder~youtube-transcript-scraper"
APIFY_AUDIO_ACTOR = "lurkapi~youtube-to-mp3-audio-downloader"

DEFAULT_TRANSCRIPT_PROVIDERS = ["scrapecreators", "apify"]
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
                raise YoutubeError(f"request to {url} failed after {attempt} retries: {exc}") from exc
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
    key = os.environ.get("SCRAPECREATORS_API_KEY")
    if not key:
        raise YoutubeError("SCRAPECREATORS_API_KEY not set in environment")
    return key


def _sc_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{SCRAPECREATORS_BASE}{path}"
    headers = {"x-api-key": _sc_api_key()}
    resp = _request_with_backoff("GET", url, headers=headers, params=params)
    if resp.status_code != 200:
        raise YoutubeError(f"ScrapeCreators {path} failed: HTTP {resp.status_code}: {resp.text[:300]}")
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
    token = os.environ.get("APIFY_TOKEN")
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
    token = _apify_token()
    submit_url = f"{APIFY_BASE}/acts/{actor_id}/runs"
    resp = _request_with_backoff("POST", submit_url, params={"token": token}, json_body=run_input)
    if resp.status_code not in (200, 201):
        raise YoutubeError(f"Apify actor {actor_id} run submission failed: HTTP {resp.status_code}: {resp.text[:300]}")
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
        resp = _request_with_backoff("GET", status_url, params={"token": token})
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
    resp = _request_with_backoff("GET", items_url, params={"token": token, "format": "json"})
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


def list_channel_videos(channel: str, *, memobase_cfg: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], str]:
    """Return ``(videos, provider_used)``. Tries ScrapeCreators first
    (primary, per the finalized role split); on ANY failure (missing key,
    network error, bad response) falls back to Apify's channel-scraper
    actor. Raises :class:`YoutubeError` only if BOTH fail."""
    try:
        videos = channel_videos_scrapecreators(channel)
        return videos, "scrapecreators"
    except Exception as exc:  # noqa: BLE001 - any ScrapeCreators failure triggers fallback
        logger.warning("ScrapeCreators channel listing failed for %r (%s); falling back to Apify", channel, exc)
    videos = channel_videos_apify(channel)
    return videos, "apify"


# ---------------------------------------------------------------------------
# Transcript ladder
# ---------------------------------------------------------------------------


def transcript_scrapecreators(video_id: str, *, language: str = "ru") -> Optional[List[Dict[str, Any]]]:
    """Return a list of ``{"text", "start_sec", "end_sec"}`` segments, or
    ``None`` if ScrapeCreators cleanly reports "no captions for this video"
    (a definitive, terminal result — not an error). Raises
    :class:`YoutubeError` for an actual request/parse failure."""
    payload = _sc_get(
        "/v1/youtube/video/transcript", {"url": video_watch_url(video_id), "language": language}
    )
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


def transcript_apify(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """Fallback transcript source via ``supreme_coder/youtube-transcript-scraper``.
    Same return contract as :func:`transcript_scrapecreators`."""
    items = _apify_run_actor(APIFY_TRANSCRIPT_ACTOR, {"videoUrls": [video_watch_url(video_id)]})
    if not items:
        return None
    item = items[0]
    raw_segments = item.get("transcript") or item.get("captions")
    if not raw_segments:
        return None
    segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start") if seg.get("start") is not None else seg.get("startMs", 0) / 1000.0
        end = seg.get("end") if seg.get("end") is not None else (
            seg.get("endMs") / 1000.0 if seg.get("endMs") is not None else None
        )
        segments.append({"text": text, "start_sec": float(start or 0), "end_sec": end})
    return segments


_TRANSCRIPT_FETCHERS = {
    "scrapecreators": transcript_scrapecreators,
    "apify": transcript_apify,
}


def get_transcript(
    video_id: str, *, memobase_cfg: Optional[Dict[str, Any]] = None
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
            segments = fetcher(video_id)
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
    if not items:
        raise YoutubeError(f"Apify audio actor returned no items for video {video_id}")
    item = items[0]
    audio_url = item.get("mp3Url") or item.get("audioUrl") or item.get("downloadUrl")
    if not audio_url:
        raise YoutubeError(f"Apify audio actor result for video {video_id} has no audio URL: {item!r}")
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


def extract_video(source: str, *, memobase_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Full single-video route: parse id -> transcript ladder -> (if no
    captions anywhere) audio download + STT fallback -> extract.py-shaped
    doc. NEVER raises — every failure becomes a ``skipped`` reason with
    empty text, exactly like extract.py's ``extract()`` contract, so
    ``ingest.py`` handles it identically regardless of source_type.
    """
    try:
        video_id = parse_video_id(source)
    except YoutubeError as exc:
        return {"text": "", "blocks": [], "meta": {"title": None, "pages": None}, "skipped": [{"reason": str(exc)}]}

    video_meta: Dict[str, Any] = {"video_id": video_id}

    try:
        segments, provider = get_transcript(video_id, memobase_cfg=memobase_cfg)
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
        videos, list_provider = list_channel_videos(channel, memobase_cfg=memobase_cfg)
    except YoutubeError as exc:
        db.update_ingestion_job(conn, job_id, status="failed", stage="failed")
        return {"status": "failed", "error": f"could not list channel videos: {exc}", "job_id": job_id}

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
            "message": (
                f"Канал «{channel}»: {len(videos)} видео (порог подтверждения: {threshold}). "
                f"Оценка: ${estimate['total_usd']:.4f}. Повторите запрос с подтверждением, чтобы продолжить."
            ),
        }

    db.update_ingestion_job(conn, job_id, stage="ingest_videos", items_total=len(videos))
    per_video: List[Dict[str, Any]] = []
    counts = {"done": 0, "unchanged": 0, "failed": 0}
    for i, v in enumerate(videos):
        result = ingest_mod.ingest_source(
            conn, collection_row, v["url"], "youtube", memobase_cfg=memobase_cfg, confirm=True
        )
        status = result.get("status")
        counts[status] = counts.get(status, 0) + 1
        per_video.append({"video_id": v.get("video_id"), "status": status})
        db.update_ingestion_job(conn, job_id, items_done=i + 1)

    db.update_ingestion_job(conn, job_id, status="done", stage="done")
    return {
        "status": "done",
        "job_id": job_id,
        "list_provider": list_provider,
        "video_count": len(videos),
        "videos_done": counts.get("done", 0),
        "videos_unchanged": counts.get("unchanged", 0),
        "videos_failed": counts.get("failed", 0),
        "per_video": per_video,
    }
