"""``memobase_selfcheck``: a retrieval-quality smoke test for one collection.

Верификация после загрузки: memobase_selfcheck задаёт базе контрольные
вопросы по случайным фрагментам и сверяет, находит ли их поиск (smoke-тест
качества индексации). This also feeds the calibration path: the collection's
``rrf_threshold``/``rerank_threshold`` columns start out NULL (see
``config.get_collection_cfg``'s fallback-default comment) and this module is
where a real, collection-specific number for them can come from.

Two question-generation modes:

  * **LLM mode** (``llm`` passed in, e.g. ``ctx.llm`` from ``tools.py``):
    asks the model to write one short, factual, answerable-ONLY-from-this-
    chunk RU question per sampled chunk — a genuine semantic retrieval
    test.
  * **Heuristic fallback** (no ``llm``, or the LLM call fails): the "control
    question" is a representative excerpt of the chunk's own text (skipping
    a likely leading heading). This is a weaker test (closer to "can FTS/
    vector find text that is literally present" than "can it answer a
    natural question about it"), but it needs zero external calls and never
    fails, so ``memobase_selfcheck`` always produces a real result.

Either way, the actual check is mechanical and identical: run
``retrieve.hybrid_search`` for the control query and see whether the chunk
it was generated from comes back in the top-k — never trust the LLM's own
opinion of whether retrieval "worked".
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from . import retrieve as retrieve_mod

logger = logging.getLogger("memobase.selfcheck")

DEFAULT_SAMPLE_SIZE = 8
MIN_CHUNKS_FOR_SELFCHECK = 5
RETRIEVE_K = 10
# Words skipped from the start of a chunk before taking the heuristic
# excerpt — the first few words are often a heading/label, not prose that
# makes a good standalone search probe.
_HEURISTIC_SKIP_WORDS = 3
_HEURISTIC_EXCERPT_WORDS = 12

_WORD_RE = re.compile(r"\S+")


def _heuristic_control_query(chunk_text: str) -> str:
    words = _WORD_RE.findall(chunk_text)
    if len(words) <= _HEURISTIC_SKIP_WORDS:
        return chunk_text[:200].strip()
    excerpt_words = words[_HEURISTIC_SKIP_WORDS : _HEURISTIC_SKIP_WORDS + _HEURISTIC_EXCERPT_WORDS]
    return " ".join(excerpt_words)


def _llm_control_questions(llm: Any, chunks: List[Dict[str, Any]]) -> Optional[Dict[int, str]]:
    """Ask *llm* for one control question per chunk. Returns
    ``{chunk_index: question}`` (index into *chunks*), or None on any
    failure (caller falls back to the heuristic per-chunk excerpt)."""
    numbered = "\n\n".join(f"[{i}] {c['text'][:1500]}" for i, c in enumerate(chunks))
    prompt = (
        "Для КАЖДОГО из пронумерованных фрагментов ниже придумай один короткий "
        "фактический вопрос на русском языке, ответ на который содержится "
        "ТОЛЬКО в этом фрагменте (не додумывай факты сверх текста). "
        "Верни СТРОГО JSON-массив вида "
        '[{"index": <номер фрагмента>, "question": "..."}], без пояснений вокруг.\n\n'
        f"{numbered}"
    )
    try:
        result = llm.complete(
            messages=[{"role": "user", "content": prompt}], temperature=0, purpose="memobase_selfcheck",
        )
        raw_text = (getattr(result, "text", "") or "").strip()
    except Exception:  # noqa: BLE001 - any LLM failure falls back to the heuristic
        logger.info("selfcheck: llm control-question generation failed; using heuristic excerpts", exc_info=True)
        return None

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw_text, re.DOTALL)
    if fence:
        raw_text = fence.group(1)
    brace_start, brace_end = raw_text.find("["), raw_text.rfind("]")
    if brace_start != -1 and brace_end > brace_start:
        raw_text = raw_text[brace_start : brace_end + 1]
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None

    out: Dict[int, str] = {}
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("index"), int) and item.get("question"):
            out[item["index"]] = str(item["question"])
    return out or None


def run_selfcheck(
    conn,
    collection_row: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    llm: Any = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Run the control-question retrieval smoke test for one collection.

    Returns a report dict: ``{status, collection, sample_size, checked,
    found, coverage_pct, mode_seen, suggested_thresholds, details}`` where
    ``status`` is ``"ok"`` or ``"skipped"`` (too few chunks — never
    ``"failed"``: every internal failure degrades to a lower-quality check
    rather than an error, matching this module's "always produce a real
    result" contract).
    """
    collection_id = collection_row["id"]
    total_row = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE collection_id = ? AND tombstoned_at IS NULL", (collection_id,)
    ).fetchone()
    total_chunks = total_row["n"] if total_row else 0

    if total_chunks < MIN_CHUNKS_FOR_SELFCHECK:
        return {
            "status": "skipped",
            "collection": collection_row.get("name"),
            "reason": (
                f"в коллекции всего {total_chunks} фрагмент(ов) — меньше порога "
                f"{MIN_CHUNKS_FOR_SELFCHECK}, самопроверка пропущена"
            ),
        }

    n = min(sample_size, total_chunks)
    sample_rows = conn.execute(
        "SELECT id, text FROM chunks WHERE collection_id = ? AND tombstoned_at IS NULL "
        "ORDER BY RANDOM() LIMIT ?",
        (collection_id, n),
    ).fetchall()
    chunks = [{"chunk_id": r["id"], "text": r["text"]} for r in sample_rows]

    questions_by_index: Dict[int, str] = {}
    if llm is not None:
        questions_by_index = _llm_control_questions(llm, chunks) or {}

    details: List[Dict[str, Any]] = []
    found_count = 0
    modes_seen: Dict[str, int] = {}
    found_scores_by_mode: Dict[str, List[float]] = {}

    for i, c in enumerate(chunks):
        query = questions_by_index.get(i) or _heuristic_control_query(c["text"])
        source_used = "llm" if i in questions_by_index else "heuristic"
        candidates = retrieve_mod.hybrid_search(conn, collection_id, query, RETRIEVE_K, cfg)
        mode = candidates[0]["mode"] if candidates else "none"
        modes_seen[mode] = modes_seen.get(mode, 0) + 1

        found = False
        rank = None
        score = None
        for r, cand in enumerate(candidates, start=1):
            if cand["chunk_id"] == c["chunk_id"]:
                found = True
                rank = r
                score = cand["score"]
                break

        if found:
            found_count += 1
            found_scores_by_mode.setdefault(mode, []).append(score)

        details.append({
            "chunk_id": c["chunk_id"], "query": query, "query_source": source_used,
            "found": found, "rank": rank, "score": score, "mode": mode,
        })

    coverage_pct = round(100.0 * found_count / len(chunks), 1) if chunks else 0.0

    # Conservative threshold suggestion: the minimum score among genuinely
    # found control questions, per mode — anything at/above this level did
    # successfully retrieve real content in THIS collection.
    suggested_thresholds = {
        mode: round(min(scores), 6) for mode, scores in found_scores_by_mode.items() if scores
    }

    return {
        "status": "ok",
        "collection": collection_row.get("name"),
        "sample_size": len(chunks),
        "checked": len(chunks),
        "found": found_count,
        "coverage_pct": coverage_pct,
        "modes_seen": modes_seen,
        "suggested_thresholds": suggested_thresholds,
        "details": details,
    }


def format_report(report: Dict[str, Any]) -> str:
    """Render a selfcheck report as an RU-facing plain-text summary."""
    if report.get("status") == "skipped":
        return f"Самопроверка пропущена: {report.get('reason')}"

    lines = [
        f"Самопроверка коллекции «{report['collection']}»: "
        f"{report['found']}/{report['checked']} контрольных вопросов нашли исходный фрагмент "
        f"({report['coverage_pct']}% покрытие).",
    ]
    if report.get("modes_seen"):
        modes = ", ".join(f"{m}: {n}" for m, n in report["modes_seen"].items())
        lines.append(f"Режимы поиска, встреченные при проверке: {modes}.")
    if report.get("suggested_thresholds"):
        lines.append("Рекомендованные пороги достаточности (минимум среди найденных):")
        for mode, thr in report["suggested_thresholds"].items():
            lines.append(f"  {mode}: {thr}")
    missed = [d for d in report.get("details", []) if not d["found"]]
    if missed:
        lines.append(f"Не найдено ({len(missed)}):")
        for d in missed[:10]:
            lines.append(f"  фрагмент #{d['chunk_id']}: запрос «{d['query'][:80]}»")
    return "\n".join(lines)
