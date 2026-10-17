"""Speech-to-text for memobase.

Two presets (``memobase.stt.preset``, default ``"groq"``):

  * **Groq** ``whisper-large-v3-turbo`` (default) — OpenAI-compatible
    ``https://api.groq.com/openai/v1/audio/transcriptions``,
    ``verbose_json`` + ``timestamp_granularities=[segment]`` for real
    decoder-produced segment timecodes (not model-guessed). 25MB/request
    limit -> files over that are chunked via ffmpeg: 16kHz mono FLAC,
    ~20-minute pieces with ~8s overlap, transcribed independently, then
    merged with TOKEN-level longest-suffix/prefix alignment across the
    overlap window (explicitly NOT fuzzy string matching) with timestamps
    offset by each chunk's start and low-confidence
    merges flagged rather than silently guessed.
  * **Gemini** ``gemini-2.5-flash-lite`` (fallback, on Groq 429/quota/missing
    key) — whole file up to ~9.5h, no chunking; its timecodes are
    model-*reported* text (drift on long audio, per Google's own forum
    discussion) so results from this path are tagged ``trust="low"``.

Downloaded audio is deleted after a SUCCESSFUL transcription (§1.9 #19 —
"диск на 40 ГБ переполнится молча"); a failed transcription leaves the file
in place for inspection/retry (deleting on failure would make debugging a
non-transcribing file impossible).

ffmpeg discovery: ``shutil.which("ffmpeg")`` first (works on a VPS with
``apt install ffmpeg``); falls back to this machine's known WinGet install
path pattern (``Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe`` under
``%LOCALAPPDATA%/Microsoft/WinGet/Packages``) so local dev/tests work
without requiring ffmpeg on PATH.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import ledger
from .secrets import get_env
from .security import DEFAULT_USER_AGENT, describe_request_error

logger = logging.getLogger("memobase.stt")

DEFAULT_PRESET = "groq"
GROQ_MODEL = "whisper-large-v3-turbo"
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MAX_BYTES = 25 * 1024 * 1024
# Gemini inline_data requests are capped at ~20 MB by the API; base64 adds a
# third on top — never read a bigger file into memory for it.
GEMINI_MAX_INLINE_BYTES = 15 * 1024 * 1024
# Whole-input cap for one STT job (local file or downloaded audio).
MAX_STT_INPUT_BYTES = 500 * 1024 * 1024
# ffmpeg/ffprobe may only read local files: a crafted "audio" file (HLS
# playlist, concat list) must not make them fetch URLs.
_FFMPEG_SAFE_INPUT_FLAGS = ["-nostdin", "-protocol_whitelist", "file"]
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

CHUNK_MINUTES = 20
CHUNK_OVERLAP_S = 8
CHUNK_SAMPLE_RATE = 16000

DEFAULT_TIMEOUT_S = 120.0
MAX_RETRIES = 4
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class SttError(RuntimeError):
    """Raised when EVERY configured STT path (Groq then Gemini fallback)
    fails. Callers (youtube.py's audio fallback, ingest.py's audio/video
    source_type dispatch) treat this as an honest skip reason, never a
    crash."""


# ---------------------------------------------------------------------------
# ffmpeg discovery
# ---------------------------------------------------------------------------

_WINGET_FFMPEG_GLOB = str(
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg_*" / "ffmpeg-*" / "bin" / "ffmpeg.exe"
)


def find_ffmpeg() -> Optional[str]:
    """Return a path to an ffmpeg executable, or None if not found. Never
    raises. Checked in order: PATH (``shutil.which`` — the apt-installed VPS
    case), then this machine's known WinGet install glob (local dev)."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    matches = sorted(glob.glob(_WINGET_FFMPEG_GLOB), reverse=True)
    return matches[0] if matches else None


def _run_ffmpeg(args: List[str], *, timeout_s: float = 600.0) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SttError("ffmpeg not found (checked PATH and the local WinGet install path)")
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", *_FFMPEG_SAFE_INPUT_FLAGS, *args],
            capture_output=True, timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SttError(f"ffmpeg invocation failed: {exc}") from exc
    if proc.returncode != 0:
        raise SttError(f"ffmpeg exited {proc.returncode}: {proc.stderr.decode('utf-8', errors='replace')[:500]}")


def _probe_duration_s(path: str) -> float:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SttError("ffmpeg not found")
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))
    if not Path(ffprobe).exists():
        ffprobe = "ffprobe"  # hope it's on PATH too
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-protocol_whitelist", "file", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, timeout=60, check=False,
        )
        return float(proc.stdout.decode("utf-8", errors="replace").strip())
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise SttError(f"ffprobe duration probe failed: {exc}") from exc


def _convert_to_flac_16k_mono(src_path: str, dst_path: str) -> None:
    _run_ffmpeg(["-i", src_path, "-ac", "1", "-ar", str(CHUNK_SAMPLE_RATE), dst_path])


def _slice_chunk(src_path: str, dst_path: str, *, start_s: float, duration_s: float) -> None:
    _run_ffmpeg(["-ss", str(start_s), "-t", str(duration_s), "-i", src_path,
                 "-ac", "1", "-ar", str(CHUNK_SAMPLE_RATE), dst_path])


def chunk_audio_file(path: str, *, chunk_minutes: int = CHUNK_MINUTES,
                      overlap_s: int = CHUNK_OVERLAP_S, tmp_dir: Optional[str] = None) -> List[Tuple[str, float]]:
    """Convert *path* to 16kHz mono FLAC and slice it into
    ``chunk_minutes``-long pieces with ``overlap_s`` seconds of overlap.
    Returns ``[(chunk_path, start_offset_sec), ...]`` — caller is
    responsible for deleting the chunk files after use (they live under
    *tmp_dir* or a fresh ``tempfile.mkdtemp()`` if not given)."""
    duration_s = _probe_duration_s(path)
    workdir = tmp_dir or tempfile.mkdtemp(prefix="hermes_memobase_stt_")
    flac_path = str(Path(workdir) / "full_16k_mono.flac")
    _convert_to_flac_16k_mono(path, flac_path)

    chunk_s = chunk_minutes * 60
    step_s = max(1, chunk_s - overlap_s)
    chunks: List[Tuple[str, float]] = []
    start = 0.0
    idx = 0
    while start < duration_s:
        this_duration = min(chunk_s, duration_s - start)
        chunk_path = str(Path(workdir) / f"chunk_{idx:04d}.flac")
        _slice_chunk(flac_path, chunk_path, start_s=start, duration_s=this_duration)
        chunks.append((chunk_path, start))
        idx += 1
        start += step_s
    return chunks


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _request_with_backoff(method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
                           data: Optional[dict] = None, files: Optional[dict] = None,
                           json_body: Optional[dict] = None, timeout: float = DEFAULT_TIMEOUT_S,
                           max_retries: int = MAX_RETRIES):
    import requests  # heavy/optional import kept local

    req_headers = {"User-Agent": DEFAULT_USER_AGENT, **(headers or {})}
    attempt = 0
    while True:
        try:
            resp = requests.request(
                method, url, headers=req_headers, data=data, files=files, json=json_body, timeout=timeout
            )
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise SttError(
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
# Groq whisper-large-v3-turbo
# ---------------------------------------------------------------------------


def transcribe_groq(audio_path: str, *, filename: str = "audio.flac") -> List[Dict[str, Any]]:
    """Transcribe ONE file (must already be <= 25MB) via Groq's
    OpenAI-compatible endpoint with ``verbose_json`` + segment timestamps.
    Returns ``[{"text","start_sec","end_sec"}, ...]``. Raises
    :class:`SttError` on any failure (missing key, HTTP error, malformed
    response)."""
    api_key = get_env("GROQ_API_KEY")
    if not api_key:
        raise SttError("GROQ_API_KEY not set in environment")

    size = os.path.getsize(audio_path)  # checked BEFORE reading into memory
    if size > GROQ_MAX_BYTES:
        raise SttError(f"{os.path.basename(audio_path)} is {size} bytes, over Groq's {GROQ_MAX_BYTES}-byte limit")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (filename, audio_bytes)}
    data = {
        "model": GROQ_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }
    resp = _request_with_backoff("POST", GROQ_URL, headers=headers, data=data, files=files)
    if resp.status_code != 200:
        # The provider's raw response body (often JSON) must never reach a
        # user-facing message (tools.py's _audio_stt_output would otherwise
        # dump it straight into the chat) -- full body only to the debug
        # log, for whoever is actually debugging this.
        logger.debug("Groq STT call failed: HTTP %s body=%s", resp.status_code, resp.text)
        raise SttError(f"STT groq HTTP {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise SttError(f"Groq STT response was not JSON: {exc}") from exc

    segments = payload.get("segments")
    if segments is None:
        # Some verbose_json responses may omit segments for very short clips;
        # degrade to one whole-file segment rather than failing outright.
        text = (payload.get("text") or "").strip()
        return [{"text": text, "start_sec": 0.0, "end_sec": None}] if text else []

    out = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({"text": text, "start_sec": float(seg.get("start", 0.0)), "end_sec": float(seg.get("end")) if seg.get("end") is not None else None})
    return out


# ---------------------------------------------------------------------------
# Gemini fallback (whole file, prompted timecodes, lower trust)
# ---------------------------------------------------------------------------


def transcribe_gemini(audio_path: str) -> List[Dict[str, Any]]:
    """Whole-file fallback via ``gemini-2.5-flash-lite``. Prompts the model
    to emit ``[MM:SS] text`` lines; timecodes are model-reported (not
    decoder-derived) and DRIFT on long audio per Google's own forum
    discussion (see module docstring) — callers must tag results from this
    path ``trust="low"``. Raises :class:`SttError` on any failure."""
    import base64
    import mimetypes

    api_key = get_env("GEMINI_API_KEY")
    if not api_key:
        raise SttError("GEMINI_API_KEY not set in environment")

    size = os.path.getsize(audio_path)  # checked BEFORE reading into memory
    if size > GEMINI_MAX_INLINE_BYTES:
        raise SttError(
            f"audio is {size // (1024 * 1024)} MB, over the {GEMINI_MAX_INLINE_BYTES // (1024 * 1024)} MB "
            f"inline limit of the Gemini fallback"
        )
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    mime = mimetypes.guess_type(audio_path)[0] or "audio/flac"
    b64 = base64.b64encode(audio_bytes).decode("ascii")

    prompt = (
        "Transcribe this audio completely and accurately. Output ONLY lines of the exact "
        "form `[MM:SS] <text>` (one utterance/sentence per line, timestamp = when it starts). "
        "Do not add commentary, headers, or summaries."
    )
    body = {
        "contents": [
            {"parts": [{"text": prompt}, {"inline_data": {"mime_type": mime, "data": b64}}]}
        ]
    }
    # Key in a header, never in the URL query: requests puts the full URL
    # into its exception text, which would otherwise leak into errors/logs.
    url = GEMINI_URL_TMPL.format(model=GEMINI_MODEL)
    resp = _request_with_backoff(
        "POST", url, headers={"x-goog-api-key": api_key}, json_body=body, timeout=300.0
    )
    if resp.status_code != 200:
        # See transcribe_groq's identical note -- raw provider body only to
        # debug log, never into the user-facing exception message.
        logger.debug("Gemini STT call failed: HTTP %s body=%s", resp.status_code, resp.text)
        raise SttError(f"STT gemini HTTP {resp.status_code}")
    try:
        payload = resp.json()
        text_out = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError) as exc:
        raise SttError(f"Gemini STT response malformed: {exc}") from exc

    return _parse_bracket_timecode_lines(text_out)


_BRACKET_TIME_RE = None  # compiled lazily (module import stays regex-light)


def _parse_bracket_timecode_lines(text: str) -> List[Dict[str, Any]]:
    import re

    global _BRACKET_TIME_RE
    if _BRACKET_TIME_RE is None:
        _BRACKET_TIME_RE = re.compile(r"^\[(\d+):(\d{2})\]\s*(.+)$")
    segments = []
    for line in text.splitlines():
        m = _BRACKET_TIME_RE.match(line.strip())
        if not m:
            continue
        minutes, seconds, body = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        if body:
            segments.append({"text": body, "start_sec": minutes * 60 + seconds, "end_sec": None})
    return segments


# ---------------------------------------------------------------------------
# Token-level chunk-merge alignment (§1.9 #15 — NOT fuzzy string matching)
# ---------------------------------------------------------------------------


def _find_token_overlap(tail_tokens: List[str], head_tokens: List[str], *, max_check: int) -> int:
    """Return the largest ``k`` (0 <= k <= max_check) such that
    ``tail_tokens[-k:] == head_tokens[:k]`` EXACTLY (token equality, not
    fuzzy/substring matching) — the longest-suffix/prefix alignment the task
    requires. Returns 0 if no such k > 0 exists."""
    max_k = min(len(tail_tokens), len(head_tokens), max_check)
    for k in range(max_k, 0, -1):
        if tail_tokens[-k:] == head_tokens[:k]:
            return k
    return 0


def _trim_leading_tokens(segments: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Drop the first *k* whitespace tokens' worth of text from the start of
    *segments* (a list of ``{"text", "start_sec", "end_sec"}``), which may
    span more than one segment (fully-duplicated leading segments are
    dropped entirely)."""
    remaining = k
    out: List[Dict[str, Any]] = []
    for seg in segments:
        tokens = seg["text"].split()
        if remaining <= 0:
            out.append(seg)
            continue
        if remaining >= len(tokens):
            remaining -= len(tokens)
            continue
        new_seg = dict(seg)
        new_seg["text"] = " ".join(tokens[remaining:])
        remaining = 0
        out.append(new_seg)
    return out


def merge_chunk_transcripts(
    chunks: List[Dict[str, Any]], *, overlap_search_words: int = 40, boundary_lookback_segments: int = 3
) -> Dict[str, Any]:
    """Merge STT results from consecutive overlapping audio chunks into one
    continuous transcript.

    ``chunks`` = ``[{"segments": [{"text","start_sec","end_sec"}, ...],
    "offset_sec": <chunk start, seconds>}, ...]`` in chunk order.

    Algorithm (token-level, explicitly NOT fuzzy string matching): for each
    consecutive pair, take the last
    ``boundary_lookback_segments`` segments of the already-merged transcript
    and the first ``boundary_lookback_segments`` segments of the next chunk,
    tokenize both by whitespace, and find the longest EXACT token-sequence
    match between the tail's suffix and the head's prefix (within an
    ``overlap_search_words``-token search window — the physical overlap is
    only ~8s of audio, so this window comfortably covers it without risking
    a false match far outside the true overlap region). That many leading
    tokens are trimmed from the next chunk before it is appended. If no
    exact match is found, the boundary is flagged low-confidence (both
    chunks' text is kept as-is, un-trimmed, rather than guessing) — per the
    task's explicit requirement to log/flag low-confidence merges rather
    than silently drop or duplicate content.

    Every chunk's segment timestamps are offset by that chunk's
    ``offset_sec`` BEFORE merging, so the returned segments carry absolute
    timestamps into the original (pre-chunking) audio.

    Returns ``{"segments": [...], "low_confidence_boundaries": [...]}`` —
    never raises for empty/malformed input (returns empty segments).
    """
    if not chunks:
        return {"segments": [], "low_confidence_boundaries": []}

    absolute_chunks: List[List[Dict[str, Any]]] = []
    for c in chunks:
        offset = float(c.get("offset_sec") or 0.0)
        segs = []
        for s in c.get("segments") or []:
            start = (s.get("start_sec") or 0.0) + offset
            end = (s["end_sec"] + offset) if s.get("end_sec") is not None else None
            segs.append({"text": s.get("text", ""), "start_sec": start, "end_sec": end})
        absolute_chunks.append(segs)

    merged: List[Dict[str, Any]] = list(absolute_chunks[0])
    low_confidence: List[Dict[str, Any]] = []

    for idx in range(1, len(absolute_chunks)):
        next_segs = absolute_chunks[idx]
        if not next_segs:
            continue
        if not merged:
            merged = list(next_segs)
            continue

        tail_lookback = merged[-boundary_lookback_segments:]
        head_lookback = next_segs[:boundary_lookback_segments]
        tail_tokens = " ".join(s["text"] for s in tail_lookback).split()
        head_tokens = " ".join(s["text"] for s in head_lookback).split()

        k = _find_token_overlap(tail_tokens, head_tokens, max_check=overlap_search_words)
        if k > 0:
            trimmed = _trim_leading_tokens(next_segs, k)
            merged.extend(trimmed)
        else:
            low_confidence.append({
                "boundary_chunk_index": idx,
                "reason": "no exact token-suffix/prefix overlap found within search window",
            })
            logger.warning(
                "stt merge: low-confidence boundary before chunk %d (no exact token overlap found)", idx
            )
            merged.extend(next_segs)

    return {"segments": merged, "low_confidence_boundaries": low_confidence}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _delete_file_best_effort(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        logger.debug("could not delete %s (best-effort cleanup)", path, exc_info=True)


def _safe_probe_duration_s(path: str) -> Optional[float]:
    """Best-effort wrapper around :func:`_probe_duration_s` -- duration is an
    observability nicety, never worth failing an otherwise-successful
    transcription over. In particular the small-file Groq path below needs
    no ffmpeg at all, so a machine without it must still transcribe fine;
    this just means ``duration_s`` comes back ``None``."""
    try:
        return _probe_duration_s(path)
    except SttError:
        logger.debug("duration probe failed for %s (non-fatal)", path, exc_info=True)
        return None


def _note_stt(provider: str, duration_s: Optional[float], size_bytes: int) -> None:
    # Unknown duration (no ffprobe): estimate from size at ~128 kbit/s.
    seconds = duration_s if duration_s else size_bytes / 16000.0
    ledger.note_external_call(provider, "stt", seconds)


def transcribe_long_audio(
    audio: "bytes | str", *, filename_hint: str = "audio", memobase_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Transcribe a whole audio file (bytes in memory, or a path already on
    disk), applying the ffmpeg chunk+merge ladder for files over Groq's
    25MB cap, with a Gemini whole-file fallback if Groq is unavailable/over
    quota. Returns ``{"segments": [...], "provider": "groq"|"gemini",
    "trust": "high"|"low", "low_confidence_boundaries": [...],
    "chunk_count": <int>, "duration_s": <float | None>}`` (``chunk_count`` is
    ``1`` for a whole-file transcription — the direct Groq path or the Gemini
    fallback — and the actual ffmpeg-split piece count when Groq's 25MB chunk
    ladder ran; ``duration_s`` is a best-effort ffprobe read of the WHOLE
    input, ``None`` if ffmpeg/ffprobe is unavailable or the probe fails —
    see :func:`_safe_probe_duration_s`).

    Deletes any temp file THIS function created (the downloaded/converted/
    chunked audio) after a SUCCESSFUL transcription; if *audio* was passed
    as a path, that original file is left alone (caller owns its lifecycle)
    — only files created internally (chunks, format conversions) are
    cleaned up here.
    """
    preset = ((memobase_cfg or {}).get("stt") or {}).get("preset", DEFAULT_PRESET)

    tmp_input: Optional[str] = None
    if isinstance(audio, (bytes, bytearray)):
        fd, tmp_input = tempfile.mkstemp(suffix="_" + os.path.basename(filename_hint))
        with os.fdopen(fd, "wb") as f:
            f.write(audio)
        input_path = tmp_input
    else:
        input_path = audio

    workdir = tempfile.mkdtemp(prefix="hermes_memobase_stt_")
    created_paths: List[str] = [workdir]
    if tmp_input:
        created_paths.append(tmp_input)

    try:
        size = os.path.getsize(input_path)
        if size > MAX_STT_INPUT_BYTES:
            raise SttError(
                f"audio is {size // (1024 * 1024)} MB, over the {MAX_STT_INPUT_BYTES // (1024 * 1024)} MB STT limit"
            )
        # Probed ONCE, up front, regardless of which path below ends up
        # transcribing -- best-effort (None on failure), see
        # _safe_probe_duration_s. chunk_audio_file() below re-probes this
        # same file for its own slicing math; a second local ffprobe call is
        # a cheap, non-monetary duplication, not worth threading a value
        # through chunk_audio_file's signature to avoid.
        duration_s = _safe_probe_duration_s(input_path)
        if preset == "groq" or preset == DEFAULT_PRESET:
            try:
                if size <= GROQ_MAX_BYTES:
                    segments = transcribe_groq(input_path, filename=os.path.basename(filename_hint))
                    _note_stt("groq", duration_s, size)
                    return {
                        "segments": segments, "provider": "groq", "trust": "high",
                        "low_confidence_boundaries": [], "chunk_count": 1, "duration_s": duration_s,
                    }
                chunk_infos = chunk_audio_file(input_path, tmp_dir=workdir)
                chunk_results = []
                for chunk_path, offset in chunk_infos:
                    segs = transcribe_groq(chunk_path, filename=os.path.basename(chunk_path))
                    chunk_results.append({"segments": segs, "offset_sec": offset})
                merged = merge_chunk_transcripts(chunk_results)
                _note_stt("groq", duration_s, size)
                return {
                    "segments": merged["segments"], "provider": "groq", "trust": "high",
                    "low_confidence_boundaries": merged["low_confidence_boundaries"],
                    "chunk_count": len(chunk_infos), "duration_s": duration_s,
                }
            except SttError as exc:
                logger.warning("Groq STT failed (%s); falling back to Gemini", exc)

        segments = transcribe_gemini(input_path)
        _note_stt("gemini", duration_s, size)
        return {
            "segments": segments, "provider": "gemini", "trust": "low",
            "low_confidence_boundaries": [], "chunk_count": 1, "duration_s": duration_s,
        }
    finally:
        for p in created_paths:
            if p == workdir:
                shutil.rmtree(p, ignore_errors=True)
            else:
                _delete_file_best_effort(p)


def extract_media(source: str, *, source_type: str, memobase_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """extract.py-shaped entry point for ``source_type in ("audio", "video")``.
    For ``"video"``, extracts the audio track via ffmpeg first; for
    ``"audio"``, transcribes the file directly. NEVER raises — any failure
    becomes a ``skipped`` reason with empty text (extract.py's contract)."""
    empty = lambda reason: {"text": "", "blocks": [], "meta": {"title": None, "pages": None}, "skipped": [{"reason": reason}]}  # noqa: E731

    if not os.path.exists(source):
        return empty(f"file not found: {source}")

    audio_path = source
    tmp_audio: Optional[str] = None
    try:
        if source_type == "video":
            ffmpeg = find_ffmpeg()
            if not ffmpeg:
                return empty("ffmpeg not found; cannot extract audio track from video")
            fd, tmp_audio = tempfile.mkstemp(suffix=".flac")
            os.close(fd)
            try:
                _run_ffmpeg(["-i", source, "-vn", "-ac", "1", "-ar", str(CHUNK_SAMPLE_RATE), tmp_audio])
            except SttError as exc:
                return empty(f"ffmpeg audio extraction failed: {exc}")
            audio_path = tmp_audio

        try:
            result = transcribe_long_audio(audio_path, filename_hint=os.path.basename(source), memobase_cfg=memobase_cfg)
        except SttError as exc:
            return empty(f"STT failed: {exc}")

        segments = result.get("segments") or []
        if not segments:
            return empty("STT produced no segments")

        blocks = [
            {"text": s["text"], "page": f"?t={int(s.get('start_sec') or 0)}s", "section": None, "is_code": False}
            for s in segments
        ]
        text = "\n\n".join(b["text"] for b in blocks)
        skipped = []
        if result.get("trust") == "low":
            skipped.append({"reason": "STT timecodes are model-reported (Gemini fallback), lower trust"})
        for lc in result.get("low_confidence_boundaries") or []:
            skipped.append({"reason": f"low-confidence chunk merge at boundary {lc.get('boundary_chunk_index')}"})
        return {
            "text": text, "blocks": blocks,
            "meta": {
                "title": os.path.basename(source), "pages": None, "stt_provider": result.get("provider"),
                # Threaded straight through from transcribe_long_audio's own
                # result -- ingest.py's final "done" result picks these up
                # from this same meta dict (renamed there to avoid colliding
                # with its own chunks_* text-chunk fields).
                "chunk_count": result.get("chunk_count"),
                "duration_s": result.get("duration_s"),
            },
            "skipped": skipped,
        }
    finally:
        if tmp_audio:
            _delete_file_best_effort(tmp_audio)
