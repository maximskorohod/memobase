"""JIT contextual enrichment for memobase.

Contract (matches gbrain's "JIT wrapper" idea, per §1.8): a cheap LLM writes
a short (~50-100 token) "what/where is this chunk" context string per chunk
at INDEX time; that context is prepended ONLY to the text that gets sent to
the embedder, and is thrown away for storage purposes — the ``chunks.text``
row stored in the DB (and everything downstream: FTS indexing, citation
verification in answer.py) always stays the RAW, un-enriched text. This is
non-negotiable per the design doc: "в хранилище — сырой текст (иначе
сломается проверка цитат — верификация всегда идёт по сырому)".

The enrichment string itself is NOT discarded — it is persisted to a
separate debug side-table (``chunk_enrichment``, see db.py) purely for a
future ``--explain`` trace (§1.9 gap #24), never read back into the
retrieval/answer path.

Off by default (``memobase.enrich.enabled: false``) — enrichment costs one LLM
call per new chunk at ingest time; a user opts in via config once they want
the retrieval-quality bump (-35...49% retrieval misses, per Anthropic's
contextual retrieval writeup) enough to pay for it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("memobase.enrich")

DEFAULT_MAX_ENRICHMENT_CHARS = 600  # ~100-150 tokens; hard cap regardless of what the model returns
DEFAULT_MAX_CONTEXT_CHARS_SHOWN = 400  # how much surrounding doc text/title we show the model per chunk

_SYSTEM_PROMPT = (
    "You write a short (50-100 token) context note for one excerpt from a larger document, "
    "so the excerpt can be found by search even without its surrounding text. State ONLY what "
    "the excerpt is and where it sits in the document (e.g. document title/section, what topic "
    "it covers) — do not summarize its content, do not add opinions, do not repeat the excerpt "
    "verbatim. Answer in the SAME language as the excerpt. Output the context note only, "
    "nothing else."
)


class EnrichError(RuntimeError):
    """Raised only for programmer-error-shaped misuse (missing llm handle
    when enrichment is enabled). An individual chunk's enrichment CALL
    failing is never fatal to the ingest — see :func:`enrich_chunks_for_embedding`,
    which degrades a failed chunk to "no enrichment prefix" rather than
    aborting the whole batch."""


def is_enabled(memobase_cfg: Dict[str, Any]) -> bool:
    return bool(((memobase_cfg or {}).get("enrich") or {}).get("enabled", False))


def _build_prompt(chunk_text: str, doc_context: Dict[str, Any]) -> List[Dict[str, str]]:
    title = doc_context.get("title") or doc_context.get("source_uri") or "(untitled document)"
    section = doc_context.get("section")
    where = f'Document: "{title}"' + (f', section "{section}"' if section else "")
    excerpt = chunk_text[:DEFAULT_MAX_CONTEXT_CHARS_SHOWN]
    user_msg = f"{where}\n\nExcerpt:\n{excerpt}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def enrich_one_chunk(
    chunk_text: str, doc_context: Dict[str, Any], *, llm: Any, model: Optional[str] = None
) -> str:
    """Return a short context string for ONE chunk via ``llm.complete()``
    (the ``ctx.llm`` plugin façade). Returns
    ``""`` (never raises) on any LLM-call failure — a chunk that fails to
    enrich is simply embedded without an enrichment prefix, never blocks
    ingest."""
    if llm is None:
        return ""
    try:
        messages = _build_prompt(chunk_text, doc_context)
        kwargs: Dict[str, Any] = {"purpose": "kb_enrich"}
        if model:
            kwargs["model"] = model
        result = llm.complete(messages, **kwargs)
        text = (getattr(result, "text", "") or "").strip()
        return text[:DEFAULT_MAX_ENRICHMENT_CHARS]
    except Exception:  # noqa: BLE001 - enrichment is best-effort, never fatal to ingest
        logger.warning("chunk enrichment call failed; embedding without enrichment prefix", exc_info=True)
        return ""


def enrich_chunks_for_embedding(
    chunk_texts: List[str],
    doc_context: Dict[str, Any],
    *,
    llm: Any,
    model: Optional[str] = None,
) -> Tuple[List[str], List[Optional[str]]]:
    """Return ``(texts_for_embedding, enrichment_strings)`` — same length as
    *chunk_texts*. ``texts_for_embedding[i]`` is the enrichment string (if
    any) prepended to ``chunk_texts[i]``, for the embedder call ONLY;
    ``enrichment_strings[i]`` is the raw enrichment text (or ``None`` if
    enrichment failed/returned empty) for the caller to persist to the
    debug side-table via ``db.record_chunk_enrichment`` — see module
    docstring for why these two are kept separate from the stored chunk
    text.
    """
    texts_for_embedding: List[str] = []
    enrichment_strings: List[Optional[str]] = []
    for text in chunk_texts:
        note = enrich_one_chunk(text, doc_context, llm=llm, model=model)
        enrichment_strings.append(note or None)
        texts_for_embedding.append(f"{note}\n\n{text}" if note else text)
    return texts_for_embedding, enrichment_strings
