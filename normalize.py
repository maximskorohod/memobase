"""Text normalization for memobase — the 11-step pipeline from DESIGN_v1.md:

    ftfy -> NFC -> ctrl-strip -> html.unescape -> code-tag resolve ->
    pdf boilerplate freq-heuristic -> hyphen repair -> ws collapse ->
    quote/dash (once) -> cross-block dedup -> per-block lang (py3langid)

DESIGN_v1.md's module interface:

    def normalize(doc: dict, profile: str = "default") -> dict
    # same shape, text cleaned; report counters in doc['norm_report']

This runs EXACTLY ONCE per document, between extract() and chunk() — the
stored/indexed text is the canonical, normalized text (stem.py's docstring
calls this out too: "normalize ONCE, stored text is canonical"). Nothing
downstream (chunk.py, embed.py, retrieve.py) re-normalizes.

Steps operate over ``doc['blocks']`` (extract.py's block list) rather than
the flat ``doc['text']`` string, because two steps genuinely need block-level
information:

  * step 6 (pdf boilerplate) needs each block's ``page`` number to spot
    lines that recur across many distinct pages (headers/footers/page
    numbers) — this works for ANY multi-page source, not only PDF, so it is
    applied whenever enough distinct page numbers are present.
  * step 11 ("per-CHUNK lang") is applied per BLOCK here, since at
    normalize() time these are the finest-grained retrievable units that
    exist (chunk.py has not run yet). ingest.py additionally calls the
    public :func:`detect_lang` helper on each FINAL chunk (after chunk.py
    merges/splits blocks) so ``chunks.lang`` in the DB always reflects the
    text that is actually stored, however chunk.py grouped it.

``doc['text']`` is rebuilt as ``"\\n\\n".join(block texts)`` after all steps
so the whole-document field stays in sync with the (possibly deduped/
boilerplate-stripped) block list.

Code blocks (``is_code=True``) only go through steps 1-3 (encoding fixes) —
html-unescape/code-tag-resolve/hyphen-repair/ws-collapse/quote-dash would
corrupt code syntax, so they are skipped for code units. This is a
deliberate, documented exception, not an oversight.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("memobase.normalize")

# ---------------------------------------------------------------------------
# Step 1: ftfy (mojibake / encoding-glitch repair)
# ---------------------------------------------------------------------------


def _step_ftfy(text: str, counters: Dict[str, int]) -> str:
    try:
        import ftfy
    except ImportError:
        return text
    try:
        fixed = ftfy.fix_text(text)
    except Exception:  # noqa: BLE001 - never let a text-hygiene step crash ingest
        logger.debug("ftfy.fix_text raised; leaving text as-is", exc_info=True)
        return text
    if fixed != text:
        counters["ftfy_fixes"] += 1
    return fixed


# ---------------------------------------------------------------------------
# Step 2: Unicode NFC
# ---------------------------------------------------------------------------


def _step_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Step 3: control-character strip (keep \n and \t)
# ---------------------------------------------------------------------------

_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _step_ctrl_strip(text: str, counters: Dict[str, int]) -> str:
    stripped, n = _CTRL_RE.subn("", text)
    if n:
        counters["ctrl_chars_stripped"] += n
    return stripped


# ---------------------------------------------------------------------------
# Step 4: html.unescape (decode leftover entities like &amp;, &nbsp;)
# ---------------------------------------------------------------------------


def _step_html_unescape(text: str, counters: Dict[str, int]) -> str:
    unescaped = html_lib.unescape(text)
    if unescaped != text:
        counters["html_entities_unescaped"] += 1
    return unescaped


# ---------------------------------------------------------------------------
# Step 5: code-tag resolve (leftover HTML code/pre tags -> markdown)
# ---------------------------------------------------------------------------

_PRE_TAG_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
_CODE_TAG_RE = re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL | re.IGNORECASE)
_KBD_TT_TAG_RE = re.compile(r"<(?:kbd|tt)[^>]*>(.*?)</(?:kbd|tt)>", re.DOTALL | re.IGNORECASE)
_GENERIC_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]*)?>")


def _step_resolve_code_tags(text: str, counters: Dict[str, int]) -> str:
    def _pre_sub(m: "re.Match[str]") -> str:
        counters["code_tags_resolved"] += 1
        return f"\n```\n{m.group(1).strip()}\n```\n"

    text = _PRE_TAG_RE.sub(_pre_sub, text)

    def _inline_sub(m: "re.Match[str]") -> str:
        counters["code_tags_resolved"] += 1
        return f"`{m.group(1)}`"

    text = _CODE_TAG_RE.sub(_inline_sub, text)
    text = _KBD_TT_TAG_RE.sub(_inline_sub, text)

    # Hygiene net: strip any other stray HTML tags left by imperfect
    # upstream extraction (not counted as "code tag resolves" — those are
    # tracked separately above).
    text = _GENERIC_TAG_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Step 6: PDF/multi-page boilerplate (repeated header/footer/page-number
# lines) — frequency heuristic across distinct page numbers.
# ---------------------------------------------------------------------------


def _strip_boilerplate(blocks: List[Dict[str, Any]], counters: Dict[str, int]) -> None:
    pages = {b.get("page") for b in blocks if b.get("page") is not None}
    if len(pages) < 3:
        return  # not enough distinct pages to trust a frequency heuristic

    line_pages: Dict[str, set] = defaultdict(set)
    for b in blocks:
        if b.get("is_code") or b.get("page") is None:
            continue
        for line in (b.get("text") or "").splitlines():
            s = line.strip()
            if not s or len(s) > 120:
                continue
            line_pages[s].add(b["page"])

    threshold = max(3, int(len(pages) * 0.4))
    boilerplate = {line for line, pgs in line_pages.items() if len(pgs) >= threshold}
    if not boilerplate:
        return

    for b in blocks:
        if b.get("is_code") or b.get("page") is None:
            continue
        kept_lines = []
        for line in (b.get("text") or "").splitlines():
            if line.strip() in boilerplate:
                counters["boilerplate_lines_removed"] += 1
                continue
            kept_lines.append(line)
        b["text"] = "\n".join(kept_lines)


# ---------------------------------------------------------------------------
# Step 7: hyphenation repair ("informa-\ntion" -> "information")
# ---------------------------------------------------------------------------

_HYPHEN_RE = re.compile(r"(\w)-\n\s*(\w)", re.UNICODE)


def _step_repair_hyphens(text: str, counters: Dict[str, int]) -> str:
    def _sub(m: "re.Match[str]") -> str:
        counters["hyphens_repaired"] += 1
        return m.group(1) + m.group(2)

    return _HYPHEN_RE.sub(_sub, text)


# ---------------------------------------------------------------------------
# Step 8: whitespace collapse
# ---------------------------------------------------------------------------


def _step_collapse_ws(text: str, counters: Dict[str, int]) -> str:
    before = text
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()
    if text != before:
        counters["ws_collapsed_blocks"] += 1
    return text


# ---------------------------------------------------------------------------
# Step 9: quote/dash canonicalization (once)
# ---------------------------------------------------------------------------

_QUOTE_DASH_MAP = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "′": "'",
    "–": "-", "—": "-", "−": "-", "‒": "-",
}
_QUOTE_DASH_TABLE = str.maketrans(_QUOTE_DASH_MAP)


def _step_normalize_quotes_dashes(text: str, counters: Dict[str, int]) -> str:
    hits = sum(text.count(k) for k in _QUOTE_DASH_MAP)
    if hits:
        counters["quotes_dashes_normalized"] += hits
    return text.translate(_QUOTE_DASH_TABLE)


# ---------------------------------------------------------------------------
# Step 10: cross-block dedup
# ---------------------------------------------------------------------------


def _dedup_blocks(blocks: List[Dict[str, Any]], counters: Dict[str, int]) -> List[Dict[str, Any]]:
    """Drop blocks that are exact-duplicate prose paragraphs (>= 8 words),
    keeping the first occurrence. Code blocks and table rows are exempt
    (legitimate repeats — e.g. a repeated code sample, or two rows that
    happen to render identically) and never deduped here. Empty blocks
    (nothing left after boilerplate stripping) are dropped too."""
    seen = set()
    kept: List[Dict[str, Any]] = []
    for b in blocks:
        text = (b.get("text") or "").strip()
        if not text:
            counters["empty_blocks_dropped"] += 1
            continue
        is_dedupe_candidate = (
            not b.get("is_code") and not b.get("is_table_row") and len(text.split()) >= 8
        )
        if is_dedupe_candidate:
            if text in seen:
                counters["blocks_deduped"] += 1
                continue
            seen.add(text)
        b["text"] = text
        kept.append(b)
    return kept


# ---------------------------------------------------------------------------
# Step 11: language detection (per block here; ingest.py also calls this
# per FINAL chunk — see module docstring)
# ---------------------------------------------------------------------------

_MIN_CHARS_FOR_LANGID = 20


def detect_lang(text: str) -> Optional[str]:
    """Best-effort ISO 639-1-ish language code for *text* via py3langid.
    Returns None if py3langid is unavailable, the text is too short, or
    detection fails for any reason. Never raises."""
    if not text or len(text.strip()) < _MIN_CHARS_FOR_LANGID:
        return None
    try:
        import py3langid as langid
    except ImportError:
        return None
    try:
        lang, _score = langid.classify(text[:2000])
        return lang
    except Exception:  # noqa: BLE001
        logger.debug("py3langid.classify failed", exc_info=True)
        return None


def _tag_langs(blocks: List[Dict[str, Any]], counters: Dict[str, int]) -> None:
    lang_counts: Dict[str, int] = defaultdict(int)
    for b in blocks:
        if b.get("is_code"):
            b["lang"] = None
            continue
        lang = detect_lang(b.get("text") or "")
        b["lang"] = lang
        if lang:
            lang_counts[lang] += 1
    for lang, count in lang_counts.items():
        counters[f"lang_{lang}_blocks"] = count


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def normalize(doc: Dict[str, Any], profile: str = "default") -> Dict[str, Any]:
    """Run the 11-step normalization pipeline over *doc* in place (and
    return it). ``doc['blocks']`` is cleaned/deduped/tagged; ``doc['text']``
    is rebuilt from the resulting blocks; ``doc['norm_report']`` carries
    step counters for observability/debugging.

    ``profile`` is accepted for forward-compatibility (DESIGN_v1.md's
    interface reserves it for a future per-collection normalization
    profile) but v1 only has one behavior — unrecognized profiles are
    treated the same as "default", never an error.
    """
    counters: Dict[str, int] = defaultdict(int)
    blocks = doc.get("blocks") or []

    # Steps 1-5, per block. Code blocks only get 1-3 (encoding hygiene).
    for b in blocks:
        text = b.get("text") or ""
        text = _step_ftfy(text, counters)
        text = _step_nfc(text)
        text = _step_ctrl_strip(text, counters)
        if b.get("is_code"):
            b["text"] = text
            continue
        text = _step_html_unescape(text, counters)
        text = _step_resolve_code_tags(text, counters)
        b["text"] = text

    # Step 6: cross-block boilerplate removal (needs the whole block list).
    _strip_boilerplate(blocks, counters)

    # Steps 7-9, per non-code block.
    for b in blocks:
        if b.get("is_code"):
            continue
        text = b.get("text") or ""
        text = _step_repair_hyphens(text, counters)
        text = _step_collapse_ws(text, counters)
        text = _step_normalize_quotes_dashes(text, counters)
        b["text"] = text

    # Step 10: cross-block dedup (also drops now-empty blocks).
    blocks = _dedup_blocks(blocks, counters)

    # Step 11: per-block language tag.
    _tag_langs(blocks, counters)

    doc["blocks"] = blocks
    doc["text"] = "\n\n".join(b["text"] for b in blocks if b.get("text"))
    doc["norm_report"] = dict(counters)
    return doc
