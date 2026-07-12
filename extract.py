"""Document extraction for memobase.

DESIGN_v1.md's module interface:

    def extract(path_or_url: str, source_type: str) -> dict
    # {text, blocks:[{text,page,section,is_code}], meta:{title,pages,...}, skipped:[{reason}]}

Supported ``source_type`` values: ``pdf``, ``docx``, ``html``, ``url``,
``md``, ``txt``, ``csv``.

Design notes:

* ``extract()`` NEVER raises — any failure (missing file, parser error, SSRF
  block, network error) is captured as a ``skipped`` entry and an empty/partial
  result is returned instead. Callers (ingest.py) treat "no text extracted"
  as an honest failure to report, not a crash.
* PDF: ``pdfplumber`` first (better layout/table handling), falls back to
  ``pypdf`` if pdfplumber is not installed or raises. Per-page blocks so
  citations can carry a page number.
* DOCX: ``mammoth.convert_to_markdown`` — this (unlike ``extract_raw_text``)
  preserves heading markers (``#``) and code-ish runs as markdown, which
  gives chunk.py's heading/code-fence detection something to work with.
* HTML/URL: for ``source_type == "url"`` the fetch goes through
  ``security.check_url`` (raises before any request) and
  ``security.safe_get`` (SSRF-safe, browser UA, size-capped, retrying) — NOT
  trafilatura's own fetcher, which would bypass all of that. For local HTML
  files, no SSRF check is needed (no network access).
* MD/TXT: read as-is; markdown heading/code-fence structure (if present) is
  picked up by the same generic block-builder used for DOCX/HTML.
* CSV: stdlib ``csv`` module; each data row becomes one self-describing
  ``"col: value"`` block (``is_table_row=True``) carrying the pipe-joined
  header line as ``table_header`` — consumed by chunk.py to re-attach the
  header when several rows land in the same chunk. Rows are individually
  self-describing so they remain useful even split across chunks.

Extra block keys ``is_table_row``/``table_header`` are an intentional,
backward-compatible extension of the ``{text,page,section,is_code}`` shape
in DESIGN_v1.md — chunk.py's docstring covers the contract in full; any
consumer that doesn't know about them can ignore them.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("memobase.extract")


class ExtractError(RuntimeError):
    """Raised only for programmer-error-shaped misuse (e.g. bad source_type
    dispatch table bug) — normal extraction failures never raise, they are
    reported via the returned ``skipped`` list instead."""


def _empty_result(reason: Optional[str] = None) -> Dict[str, Any]:
    result = {"text": "", "blocks": [], "meta": {"title": None, "pages": None}, "skipped": []}
    if reason:
        result["skipped"].append({"reason": reason})
    return result


# ---------------------------------------------------------------------------
# Shared: markdown-ish text -> blocks (heading tracking + code-fence split)
# ---------------------------------------------------------------------------

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FENCE_SPLIT_RE = re.compile(r"(```.*?```)", re.DOTALL)


def _split_by_headings(text: str) -> List[tuple]:
    """Return [(heading_or_None, body_text), ...] slicing *text* at each
    markdown heading line; heading text itself is not included in body."""
    matches = list(_HEADING_LINE_RE.finditer(text))
    if not matches:
        return [(None, text)]
    sections = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))
    for idx, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end]))
    return sections


def _split_code_fences(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    parts = _FENCE_SPLIT_RE.split(text)
    segments = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```") and part.endswith("```") and len(part) >= 6:
            segments.append({"text": part, "is_code": True})
        elif part.strip():
            segments.append({"text": part, "is_code": False})
    return segments


def _text_to_blocks(text: str, *, page: Optional[int] = None) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not text or not text.strip():
        return blocks
    for heading, body in _split_by_headings(text):
        for seg in _split_code_fences(body):
            blocks.append(
                {"text": seg["text"], "page": page, "section": heading, "is_code": seg["is_code"]}
            )
    return blocks


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _extract_pdf(path: str) -> Dict[str, Any]:
    skipped: List[Dict[str, str]] = []
    blocks: List[Dict[str, Any]] = []
    title = None
    page_count = 0

    used_pdfplumber = False
    try:
        import pdfplumber  # optional dependency

        used_pdfplumber = True
    except ImportError:
        pdfplumber = None  # type: ignore

    if used_pdfplumber:
        try:
            with pdfplumber.open(path) as pdf:  # type: ignore[union-attr]
                page_count = len(pdf.pages)
                meta = getattr(pdf, "metadata", None) or {}
                title = meta.get("Title") or None
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                    except Exception as exc:  # noqa: BLE001 - defensive, per-page
                        skipped.append({"reason": f"pdfplumber failed on page {i}: {exc}"})
                        text = ""
                    if text.strip():
                        blocks.extend(_text_to_blocks(text, page=i))
        except Exception as exc:  # noqa: BLE001 - whole-doc pdfplumber failure
            skipped.append({"reason": f"pdfplumber failed on {path}: {exc}; falling back to pypdf"})
            blocks = []
            used_pdfplumber = False

    if not used_pdfplumber or not blocks:
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            page_count = len(reader.pages)
            try:
                title = (reader.metadata or {}).get("/Title") or None
            except Exception:  # noqa: BLE001
                title = None
            for i, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001
                    skipped.append({"reason": f"pypdf failed on page {i}: {exc}"})
                    text = ""
                if text.strip():
                    blocks.extend(_text_to_blocks(text, page=i))
        except ImportError:
            skipped.append({"reason": "neither pdfplumber nor pypdf is installed"})
        except Exception as exc:  # noqa: BLE001
            skipped.append({"reason": f"pypdf failed on {path}: {exc}"})

    full_text = "\n\n".join(b["text"] for b in blocks)
    return {
        "text": full_text,
        "blocks": blocks,
        "meta": {"title": title, "pages": page_count or None},
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def _extract_docx(path: str) -> Dict[str, Any]:
    skipped: List[Dict[str, str]] = []
    try:
        import mammoth
    except ImportError:
        return _empty_result("mammoth is not installed")

    try:
        with open(path, "rb") as f:
            result = mammoth.convert_to_markdown(f)
    except Exception as exc:  # noqa: BLE001
        return _empty_result(f"mammoth failed on {path}: {exc}")

    text = result.value or ""
    for msg in getattr(result, "messages", None) or []:
        skipped.append({"reason": f"mammoth: {getattr(msg, 'message', str(msg))}"})

    blocks = _text_to_blocks(text, page=None)
    return {
        "text": text,
        "blocks": blocks,
        "meta": {"title": None, "pages": None},
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# HTML / URL
# ---------------------------------------------------------------------------


def _extract_html_or_url(path_or_url: str, *, is_url: bool) -> Dict[str, Any]:
    skipped: List[Dict[str, str]] = []

    try:
        import trafilatura
    except ImportError:
        return _empty_result("trafilatura is not installed")

    if is_url:
        from .security import check_url, safe_get, SecurityError

        try:
            check_url(path_or_url)
        except SecurityError as exc:
            return _empty_result(f"blocked by SSRF guard: {exc}")

        try:
            raw = safe_get(path_or_url)
        except SecurityError as exc:
            return _empty_result(f"blocked by SSRF guard during fetch: {exc}")
        except Exception as exc:  # noqa: BLE001 - network failure of any kind
            return _empty_result(f"failed to fetch {path_or_url}: {exc}")
        html_text = raw.decode("utf-8", errors="replace")
    else:
        try:
            with open(path_or_url, "r", encoding="utf-8", errors="replace") as f:
                html_text = f.read()
        except OSError as exc:
            return _empty_result(f"failed to read {path_or_url}: {exc}")

    title = None
    try:
        meta = trafilatura.extract_metadata(html_text)
        if meta is not None:
            title = getattr(meta, "title", None) or None
    except Exception:  # noqa: BLE001 - metadata extraction is best-effort
        logger.debug("trafilatura metadata extraction failed", exc_info=True)

    try:
        md_text = (
            trafilatura.extract(
                html_text,
                output_format="markdown",
                include_tables=True,
                include_formatting=True,
                favor_recall=True,
            )
            or ""
        )
    except Exception as exc:  # noqa: BLE001
        skipped.append({"reason": f"trafilatura extraction failed: {exc}"})
        md_text = ""

    if not md_text.strip():
        skipped.append({"reason": "trafilatura returned no content (paywall/JS-only/empty page?)"})

    blocks = _text_to_blocks(md_text, page=None)
    return {
        "text": md_text,
        "blocks": blocks,
        "meta": {"title": title, "pages": None},
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# MD / TXT
# ---------------------------------------------------------------------------


def _extract_plain_text(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as exc:
        return _empty_result(f"failed to read {path}: {exc}")

    blocks = _text_to_blocks(text, page=None)
    return {
        "text": text,
        "blocks": blocks,
        "meta": {"title": None, "pages": None},
        "skipped": [],
    }


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _extract_csv(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            raw = f.read()
    except OSError as exc:
        return _empty_result(f"failed to read {path}: {exc}")

    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(raw), dialect)
    rows = list(reader)
    if not rows:
        return _empty_result("CSV file has no rows")

    header = [h.strip() for h in rows[0]]
    header_line = " | ".join(header) if header else ""
    blocks: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for row_idx, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            skipped.append(
                {"reason": f"CSV row {row_idx} has {len(row)} fields, header has {len(header)} (kept anyway, positional)"}
            )
        pairs = []
        for col_idx, cell in enumerate(row):
            col_name = header[col_idx] if col_idx < len(header) else f"col{col_idx + 1}"
            pairs.append(f"{col_name}: {cell.strip()}")
        row_text = "\n".join(pairs)
        blocks.append(
            {
                "text": row_text,
                "page": None,
                "section": None,
                "is_code": False,
                "is_table_row": True,
                "table_header": header_line,
            }
        )

    full_text = "\n\n".join(b["text"] for b in blocks)
    return {
        "text": full_text,
        "blocks": blocks,
        "meta": {"title": None, "pages": None, "row_count": len(blocks)},
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {
    "pdf": lambda p: _extract_pdf(p),
    "docx": lambda p: _extract_docx(p),
    "html": lambda p: _extract_html_or_url(p, is_url=False),
    "url": lambda p: _extract_html_or_url(p, is_url=True),
    "md": lambda p: _extract_plain_text(p),
    "txt": lambda p: _extract_plain_text(p),
    "csv": lambda p: _extract_csv(p),
}


def extract(path_or_url: str, source_type: str) -> Dict[str, Any]:
    """Extract text/structure from *path_or_url*. Never raises — any failure
    is reported via the returned ``skipped`` list, with ``text``/``blocks``
    left empty (or partially filled, for partial failures like one bad PDF
    page among many good ones).
    """
    normalized_type = (source_type or "").strip().lower()
    handler = _DISPATCH.get(normalized_type)
    if handler is None:
        return _empty_result(f"unsupported source_type: {source_type!r}")

    try:
        return handler(path_or_url)
    except FileNotFoundError as exc:
        return _empty_result(f"file not found: {exc}")
    except Exception as exc:  # noqa: BLE001 - extract() must never raise
        logger.exception("extract() failed for %r (source_type=%r)", path_or_url, source_type)
        return _empty_result(f"unexpected extraction error: {exc}")
