"""Structural chunker for memobase ("qmd-style" break scoring).

DESIGN_v1.md's module interface:

    def chunk(doc: dict, target_tokens: int, overlap_pct: float) -> list[dict]
    # [{text, seq, page_or_timecode, section, is_code}]

Algorithm (deliberately simple, stdlib-only, no ML model):

1. ``doc['blocks']`` (as produced by ``extract.py``, already normalized by
   ``normalize.py``) is flattened into a sequence of *units*. Each unit is
   either:
     * an atomic code-fence block (``is_code=True``) — NEVER split, per
       DESIGN_v1.md ("code fences never split"), or
     * an atomic table/CSV row (``is_table_row=True``) — also never split
       mid-row, or
     * one paragraph (blank-line-separated) of a prose block, tagged
       ``is_heading`` if it looks like a markdown heading (``# ...``) or a
       short ALL-CAPS line.

2. Units are assembled into chunks with a greedy sliding window: keep
   appending units while the running token estimate is below
   ``target_tokens * UPPER_MULT``. Once the running total crosses
   ``target_tokens * LOWER_MULT``, every unit boundary from then on is
   scored:

       score = structural_weight(boundary) - DECAY_WEIGHT * distance**2

   where ``distance = (tokens_so_far - target_tokens) / target_tokens``
   (the "quadratic distance decay" from the task) and
   ``structural_weight`` rewards boundaries right before a heading, right
   after a code fence, or at end-of-document, over a plain paragraph
   boundary. The highest-scoring boundary seen in the window is where the
   chunk is actually cut.

3. A single unit that alone exceeds the upper bound (e.g. a huge code
   block) is never split — it becomes its own chunk regardless of size,
   because atomicity is a harder constraint than the token target.

4. Overlap: the tail of each committed chunk (up to
   ``overlap_pct * target_tokens`` tokens, counted in whole units) is
   re-included at the start of the next chunk's window, by rewinding the
   cursor — never re-splitting a unit itself.

5. Table rows: if a chunk contains more than one ``is_table_row`` unit and
   a ``table_header`` was captured by extract.py, that header line is
   prepended once so a multi-row chunk still carries its column context
   even though each row is already independently self-describing
   ("key: value" text from extract.py).

This is a heuristic, not a learned model — tuned to be predictable and
cheap, matching the "qmd-style structural break scoring" wording in the
task (headings/code-fence/blank-line weights + quadratic decay), not a
byte-for-byte port of any specific project's chunker (qmd itself is an
external npm tool, not vendored into this repo).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Token approximation (shared with embed.py/ledger.py for cost estimation)
# ---------------------------------------------------------------------------

# Best-effort, language-agnostic approximation: no tokenizer dependency is in
# the install list (no tiktoken/transformers). Character-based counting is
# used (rather than word-count-based) because it degrades more gracefully for
# RU text, which tends to need MORE tokens per character than English in
# common BPE tokenizers (bge-m3/most LLM tokenizers) — 3.3 chars/token is a
# rough middle ground between typical English (~4) and Russian (~2.5-3)
# figures. This is only used for chunk-size *targeting* and $-cost
# estimation, never for anything that needs to be exact.
CHARS_PER_TOKEN_APPROX = 3.3


def approx_tokens(text: str) -> int:
    """Rough token-count estimate for *text*. Never zero for non-empty text."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN_APPROX))


# ---------------------------------------------------------------------------
# Heading / paragraph detection
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
# Short, punctuation-light, all-caps-ish line (RU or EN) — a common section
# title shape surviving PDF/OCR extraction that lost markdown formatting.
_ALLCAPS_HEADING_RE = re.compile(r"^[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9 \-:]{2,78}$")

_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _MD_HEADING_RE.match(stripped):
        return True
    if (
        len(stripped) <= 80
        and not stripped.endswith((".", ",", ";", ":"))
        and _ALLCAPS_HEADING_RE.match(stripped)
    ):
        return True
    return False


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def _build_units(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for block in blocks:
        text = block.get("text") or ""
        if not text.strip():
            continue
        page = block.get("page")
        section = block.get("section")
        is_code = bool(block.get("is_code"))
        is_table_row = bool(block.get("is_table_row"))
        table_header = block.get("table_header")

        if is_code or is_table_row:
            units.append(
                {
                    "text": text,
                    "tokens": approx_tokens(text),
                    "page": page,
                    "section": section,
                    "is_code": is_code,
                    "is_table_row": is_table_row,
                    "table_header": table_header,
                    "is_heading": False,
                }
            )
            continue

        for para in _split_paragraphs(text):
            units.append(
                {
                    "text": para,
                    "tokens": approx_tokens(para),
                    "page": page,
                    "section": section,
                    "is_code": False,
                    "is_table_row": False,
                    "table_header": None,
                    "is_heading": _is_heading(para),
                }
            )
    return units


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------

LOWER_MULT = 0.5     # start considering breaks once buf >= target * LOWER_MULT
UPPER_MULT = 1.6      # hard-stop once buf >= target * UPPER_MULT
OVERSHOOT_MULT = 1.9  # never let a non-atomic addition push buf past this
DECAY_WEIGHT = 4.0

_W_END_OF_DOC = 4.0
_W_BEFORE_HEADING = 10.0
_W_AFTER_CODE = 6.0
_W_PLAIN = 1.0


def _boundary_weight(units: List[Dict[str, Any]], j: int) -> float:
    """Structural weight of the boundary right AFTER unit index *j*."""
    weight = 0.0
    if units[j]["is_code"]:
        weight += _W_AFTER_CODE
    nxt = units[j + 1] if j + 1 < len(units) else None
    if nxt is None:
        weight += _W_END_OF_DOC
    elif nxt["is_heading"]:
        weight += _W_BEFORE_HEADING
    else:
        weight += _W_PLAIN
    return weight


def _render_chunk_text(commit_units: List[Dict[str, Any]]) -> str:
    table_rows = [u for u in commit_units if u.get("is_table_row")]
    parts = [u["text"] for u in commit_units]
    body = "\n\n".join(parts)
    if len(table_rows) > 1:
        header = next((u.get("table_header") for u in table_rows if u.get("table_header")), None)
        if header and not body.startswith(header):
            body = f"{header}\n{body}"
    return body


def _first_non_none(commit_units: List[Dict[str, Any]], key: str) -> Optional[Any]:
    for u in commit_units:
        if u.get(key) is not None:
            return u[key]
    return None


def chunk(doc: Dict[str, Any], target_tokens: int, overlap_pct: float) -> List[Dict[str, Any]]:
    """Split *doc* (post-extract, post-normalize) into retrieval chunks.

    ``doc`` is expected to carry ``blocks`` (see extract.py's return shape).
    Falls back to treating ``doc['text']`` as a single untagged block if
    ``blocks`` is missing/empty (defensive — should not happen in the normal
    extract -> normalize -> chunk pipeline).
    """
    target_tokens = max(1, int(target_tokens))
    overlap_pct = max(0.0, min(0.9, float(overlap_pct)))

    blocks = doc.get("blocks") or []
    if not blocks and doc.get("text"):
        blocks = [{"text": doc["text"], "page": None, "section": None, "is_code": False}]

    units = _build_units(blocks)
    if not units:
        return []

    n = len(units)
    lower = target_tokens * LOWER_MULT
    upper = target_tokens * UPPER_MULT
    overshoot_cap = target_tokens * OVERSHOOT_MULT
    overlap_token_budget = overlap_pct * target_tokens

    chunks: List[Dict[str, Any]] = []
    seq = 0
    i = 0

    while i < n:
        start_i = i
        buf_tokens = 0
        best_break: Optional[int] = None  # index j: break AFTER units[j]
        best_score = float("-inf")
        j = i

        while j < n:
            u = units[j]

            if buf_tokens == 0 and u["tokens"] >= upper:
                # Atomic unit already at/over the cap on its own: take it
                # alone, no further growth.
                buf_tokens += u["tokens"]
                j += 1
                break

            projected = buf_tokens + u["tokens"]
            if buf_tokens > 0 and projected > overshoot_cap:
                # Adding u would blow far past the cap — stop BEFORE adding
                # it; it starts the next window instead.
                break

            buf_tokens = projected
            j += 1  # j now points just past the unit we appended (index j-1)

            if buf_tokens >= lower:
                score = _boundary_weight(units, j - 1) - DECAY_WEIGHT * (
                    (buf_tokens - target_tokens) / target_tokens
                ) ** 2
                if score > best_score:
                    best_score = score
                    best_break = j - 1

            if buf_tokens >= upper:
                break

        if best_break is None:
            # Never crossed `lower` in this window (tiny doc / huge leading
            # unit) — commit everything accumulated so far.
            commit_end = max(j - 1, start_i)
        else:
            commit_end = best_break

        commit_units = units[start_i : commit_end + 1]
        if not commit_units:
            # Safety net: always make forward progress.
            commit_units = [units[start_i]]
            commit_end = start_i

        chunks.append(
            {
                "text": _render_chunk_text(commit_units),
                "seq": seq,
                "page_or_timecode": _first_non_none(commit_units, "page"),
                "section": _first_non_none(commit_units, "section"),
                "is_code": all(u["is_code"] for u in commit_units),
            }
        )
        seq += 1

        # Overlap: pull back up to overlap_token_budget worth of trailing
        # units from this commit into the next window's start.
        acc = 0
        overlap_count = 0
        for u in reversed(commit_units):
            if acc >= overlap_token_budget:
                break
            acc += u["tokens"]
            overlap_count += 1

        next_i = commit_end + 1
        if overlap_count and next_i < n:
            rewind_to = commit_end - overlap_count + 1
            if rewind_to > start_i:
                next_i = rewind_to
        if next_i <= start_i:
            next_i = start_i + 1  # guarantee progress
        i = next_i

    return chunks
