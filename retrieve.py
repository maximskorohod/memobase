"""Hybrid retrieval for memobase: FTS5(BM25, RU-stemmed) + vec0 KNN, fused
via Reciprocal Rank Fusion, then blended with an optional reranker.

DESIGN_v1.md's module interface:

    def hybrid_search(collection_id: int, query: str, k: int, cfg: dict) -> list[dict]

Like ``ingest.ingest_source`` (which needs a live ``conn`` even though the
DESIGN_v1.md interface table omits it for brevity), this function cannot do
its job without direct DB access, so the actual signature is
``hybrid_search(conn, collection_id, query, k, cfg)`` — ``conn`` first,
matching every other DB-touching function in this codebase (db.py,
ingest.py). Same for :func:`rrf_fuse` and the module-private helpers.

Pipeline (DESIGN_v1.md "Retrieval detail"):

1. **FTS leg**: query hardened/split into RU-stemmed alpha terms (matched
   against ``chunks_fts.text_stem``) and "coded" terms — anything with a
   digit, hyphen, or dot, e.g. ``gpt-4``, ``2026.4.10`` — matched against
   the RAW ``chunks_fts.text`` column instead. This is necessary, not just
   stylistic: ``stem.stem_ru``'s tokenizer regex (``[^\\W\\d_]+``)
   deliberately excludes digits, so a purely-numeric/coded token produces
   **zero** stems and would silently vanish from ``text_stem`` — searching
   it against the raw ``text`` column is the only way to still find it.
2. **Vector leg**: embed the query (same embedder as the collection), KNN
   over ``vec_c{collection_id}`` via sqlite-vec's ``vec0`` MATCH/``k``
   syntax. Skipped entirely (not an error) if sqlite-vec/the vec table is
   unavailable, or the collection is mid-embedding-migration (see below).
3. **RRF fuse**: k=60, top-rank bonus (+0.05 rank #1 of either leg, +0.02
   ranks #2-3), each over-fetched at ``k*3``.
4. **Rerank + positional blend**: :func:`rerank.rerank` is called on the
   top of the fused list; when it returns ``mode == "cohere"``, blend the
   (min-max-normalized) RRF score with the reranker's ``relevance_score``
   using qmd's positional weights (rank 1-3: 75/25, 4-10: 60/40, 11+:
   40/60) keyed off each candidate's ORIGINAL RRF rank — so the reranker
   can refine ordering but cannot "dissolve" an exact top-RRF hit. When
   ``mode == "rrf-only"``, the
   RRF order is kept as-is (no blend) — answer.py is expected to apply the
   separate, more conservative ``rrf_threshold`` for that mode.
5. **Tombstone/supersede exclusion**: ``ingest.purge_removed_chunks``
   already deletes tombstoned chunks' rows from BOTH ``chunks_fts`` and the
   vec table at tombstone time, so in the normal case neither leg can even
   surface them. This module still re-checks ``chunks.tombstoned_at`` /
   ``documents.superseded_at`` when hydrating candidates, as defense in
   depth against any future code path that tombstones without cleaning the
   indexes (and because ``memobase_query`` calls this function directly, bypassing
   ``answer.py``'s own gates).

Every candidate dict returned carries: ``chunk_id, text, score, source
("fts"|"vector"|"both"), rrf_score, rerank_score (or None), mode
("cohere"|"rrf-only"), degraded, degraded_reason, document_id, source_uri,
title, page_or_timecode, section, lang``.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from . import db
from . import embed as embed_mod
from . import rerank as rerank_mod
from . import stem as stem_mod

logger = logging.getLogger("memobase.retrieve")

RRF_K = 60
OVERFETCH_MULT = 3
TOP_RANK_BONUS_1 = 0.05
TOP_RANK_BONUS_2_3 = 0.02
RERANK_INPUT_MAX = 20  # candidates handed to the (paid) reranker per query
MAX_K = 50  # model-supplied k is clamped: bounds FTS/vec overfetch and output size

# qmd positional blend weights: (max_rank_inclusive, weight_rrf, weight_rerank)
_BLEND_BUCKETS: List[Tuple[int, float, float]] = [
    (3, 0.75, 0.25),
    (10, 0.60, 0.40),
    (10**9, 0.40, 0.60),
]


# ---------------------------------------------------------------------------
# Query hardening
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\S+")
_CODED_SHAPE_RE = re.compile(r"[\d\-.]")  # contains a digit, hyphen, or dot
_STRIP_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def _escape_fts_string(s: str) -> str:
    """Escape a literal for use inside a double-quoted FTS5 MATCH string
    (FTS5's escape convention: a literal ``"`` is written as ``""``)."""
    return s.replace('"', '""')


def _build_match_expression(raw_query: str) -> Optional[str]:
    """Split *raw_query* into RU-stemmed alpha terms (searched against
    ``text_stem``) and coded/numeric terms (searched against the raw
    ``text`` column), then build an FTS5 MATCH expression OR-ing all of
    them together (OR, not AND — BM25 ranking rewards more term overlap,
    and OR keeps recall high for multi-word natural-language questions).

    Returns None if the query has no usable terms at all (e.g. only
    stopword-shaped punctuation) — caller must treat that as "no FTS leg".
    """
    parts: List[str] = []
    seen_stems: set = set()
    seen_coded: set = set()

    for raw_tok in _TOKEN_RE.findall(raw_query):
        tok = _STRIP_EDGE_PUNCT_RE.sub("", raw_tok)
        if not tok:
            continue
        if _CODED_SHAPE_RE.search(tok):
            key = tok.lower()
            if key and key not in seen_coded:
                seen_coded.add(key)
                parts.append(f'text:"{_escape_fts_string(key)}"')
        else:
            stemmed = stem_mod.stem_ru(tok)
            for s in stemmed.split():
                if s and s not in seen_stems:
                    seen_stems.add(s)
                    parts.append(f'text_stem:"{_escape_fts_string(s)}"')

    if not parts:
        return None
    return " OR ".join(parts)


# ---------------------------------------------------------------------------
# FTS leg
# ---------------------------------------------------------------------------


def _fts_search(conn: sqlite3.Connection, collection_id: int, query: str, limit: int) -> List[int]:
    """Return chunk ids in BM25-rank order (best first). Empty list on no
    usable query terms or any FTS syntax/runtime error — logged, never
    raised, so the vector leg can still carry the search."""
    match_expr = _build_match_expression(query)
    if not match_expr:
        return []
    try:
        rows = conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? AND collection_id = ? "
            "ORDER BY rank LIMIT ?",
            (match_expr, collection_id, limit),
        ).fetchall()
    except sqlite3.Error:
        logger.warning("retrieve: FTS query failed (match=%r); FTS leg empty for this query", match_expr, exc_info=True)
        return []
    return [r["chunk_id"] for r in rows]


# ---------------------------------------------------------------------------
# Vector leg
# ---------------------------------------------------------------------------


def _vec_ready(conn: sqlite3.Connection, collection_id: int) -> bool:
    """Return True iff the per-collection vec0 table exists AND the
    sqlite-vec extension is loaded (or loads successfully) on *conn*.

    Checks via a cheap ``SELECT vec_version()`` probe first rather than
    unconditionally calling ``db.load_sqlite_vec`` on every query — loading
    a SQLite extension twice on the same already-loaded connection is not
    guaranteed to be a harmless no-op, and this connection is very likely
    reused across many retrieval calls (hot path).
    """
    if not db.vec_table_exists(conn, collection_id):
        return False
    try:
        conn.execute("SELECT vec_version()")
        return True
    except sqlite3.Error:
        return db.load_sqlite_vec(conn)


def _vector_search(
    conn: sqlite3.Connection, collection_id: int, query: str, limit: int, collection_cfg: Dict[str, Any]
) -> List[int]:
    """Return chunk ids in nearest-first order via vec0 KNN. Empty list
    (never raises) if the vec leg is unavailable, the collection is
    mid-migration, or query embedding fails for any reason — all are
    "degrade to FTS-only for this call" conditions, not errors."""
    migration_state = collection_cfg.get("migration_state") or "idle"
    # 'failed' = a reindex aborted BEFORE its atomic swap: the live vec table
    # and the collection's embedder columns are still the consistent old pair.
    if migration_state not in ("idle", "failed", None):
        logger.info("retrieve: collection mid-migration (state=%r); skipping vector leg", migration_state)
        return []

    if not _vec_ready(conn, collection_id):
        return []

    try:
        vectors = embed_mod.embed_texts([query], collection_cfg, is_query=True)
    except embed_mod.EmbedError as exc:
        logger.warning("retrieve: query embedding failed (%s); vector leg empty for this query", exc)
        return []
    if not vectors:
        return []

    vec_table = db.vec_table_name(collection_id)
    packed = embed_mod.serialize_vector(vectors[0])
    try:
        rows = conn.execute(
            f"SELECT chunk_id FROM {vec_table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (packed, limit),
        ).fetchall()
    except sqlite3.Error:
        logger.warning("retrieve: vec0 KNN query failed; vector leg empty for this query", exc_info=True)
        return []
    return [r["chunk_id"] for r in rows]


# ---------------------------------------------------------------------------
# RRF fuse
# ---------------------------------------------------------------------------


def rrf_fuse(fts_ids: List[int], vector_ids: List[int], *, k: int = RRF_K) -> "Dict[int, Dict[str, Any]]":
    """Reciprocal Rank Fusion of two rank-ordered id lists, plus qmd's
    top-rank bonus. Returns ``{chunk_id: {"score": float, "source": str}}``
    — a pure function, independently unit-testable (no DB/network).

    ``source`` is ``"fts"``, ``"vector"``, or ``"both"``.
    """
    scores: Dict[int, float] = defaultdict(float)
    sources: Dict[int, set] = defaultdict(set)

    for rank, cid in enumerate(fts_ids, start=1):
        scores[cid] += 1.0 / (k + rank)
        if rank == 1:
            scores[cid] += TOP_RANK_BONUS_1
        elif rank in (2, 3):
            scores[cid] += TOP_RANK_BONUS_2_3
        sources[cid].add("fts")

    for rank, cid in enumerate(vector_ids, start=1):
        scores[cid] += 1.0 / (k + rank)
        if rank == 1:
            scores[cid] += TOP_RANK_BONUS_1
        elif rank in (2, 3):
            scores[cid] += TOP_RANK_BONUS_2_3
        sources[cid].add("vector")

    out: Dict[int, Dict[str, Any]] = {}
    for cid, score in scores.items():
        src = sources[cid]
        source_label = "both" if len(src) > 1 else next(iter(src))
        out[cid] = {"score": score, "source": source_label}
    return out


# ---------------------------------------------------------------------------
# Hydration (join chunk text + document metadata; drop tombstoned/superseded)
# ---------------------------------------------------------------------------


def _hydrate(conn: sqlite3.Connection, chunk_ids_in_order: List[int]) -> Dict[int, Dict[str, Any]]:
    if not chunk_ids_in_order:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids_in_order)
    rows = conn.execute(
        f"""
        SELECT c.id AS chunk_id, c.text, c.page_or_timecode, c.section, c.lang,
               c.document_id, c.tombstoned_at,
               d.source_uri, d.title, d.superseded_at
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids_in_order,
    ).fetchall()
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if r["tombstoned_at"] is not None or r["superseded_at"] is not None:
            continue  # defense in depth (see module docstring)
        out[r["chunk_id"]] = dict(r)
    return out


# ---------------------------------------------------------------------------
# Positional blend
# ---------------------------------------------------------------------------


def _blend_weights(rrf_rank: int) -> Tuple[float, float]:
    for max_rank, w_rrf, w_rerank in _BLEND_BUCKETS:
        if rrf_rank <= max_rank:
            return w_rrf, w_rerank
    return _BLEND_BUCKETS[-1][1], _BLEND_BUCKETS[-1][2]  # unreachable in practice


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def hybrid_search(
    conn: sqlite3.Connection,
    collection_id: int,
    query: str,
    k: int,
    cfg: Dict[str, Any],
    *,
    user_id: Optional[str] = None,
    memobase_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return up to *k* fused, (optionally reranked+blended) candidates for
    *query* in *collection_id*. Never raises for ordinary degradation
    (empty query, no candidates, reranker unavailable, vector leg
    unavailable) — returns ``[]`` only when there is truly nothing to show.

    *cfg* is the effective per-collection config (``config.get_collection_cfg``
    output) — needs ``embedder`` (for query embedding), ``rerank`` (for the
    reranker step), and ``migration_state``.

    *user_id*: a GUEST identity (None for the owner) — the paid rerank call
    is then checked against and charged to that guest's daily budget.
    """
    from . import normalize as normalize_mod

    query = normalize_mod.normalize_query(query or "")
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 8
    k = min(k, MAX_K)
    if not query or k <= 0:
        return []

    overfetch = max(k * OVERFETCH_MULT, k)
    fts_ids = _fts_search(conn, collection_id, query, overfetch)
    vector_ids = _vector_search(conn, collection_id, query, overfetch, cfg)

    fused = rrf_fuse(fts_ids, vector_ids)
    if not fused:
        return []

    fused_order = sorted(fused.keys(), key=lambda cid: fused[cid]["score"], reverse=True)

    hydrated = _hydrate(conn, fused_order)
    # Re-filter the order to only ids that survived hydration (dropped
    # tombstoned/superseded ones), preserving RRF rank among the rest.
    fused_order = [cid for cid in fused_order if cid in hydrated]
    if not fused_order:
        return []

    migration_state = cfg.get("migration_state") or "idle"
    degraded = migration_state not in ("idle", "failed", None)
    degraded_reason = "migrating" if degraded else None

    rerank_input_ids = fused_order[:RERANK_INPUT_MAX]
    rerank_candidates = [
        {
            "chunk_id": cid,
            "text": hydrated[cid]["text"],
            "rrf_score": fused[cid]["score"],
            "source": fused[cid]["source"],
            "_rrf_rank": rank,  # 1-indexed rank BEFORE reranking; private to this module
        }
        for rank, cid in enumerate(rerank_input_ids, start=1)
    ]

    ranked, mode = rerank_mod.rerank(
        query, rerank_candidates, cfg, conn=conn, collection_id=collection_id,
        user_id=user_id, memobase_cfg=memobase_cfg,
    )

    if mode == "cohere":
        rrf_scores = [c["rrf_score"] for c in ranked]
        norm_rrf = dict(zip((c["chunk_id"] for c in ranked), _min_max_normalize(rrf_scores)))
        for c in ranked:
            w_rrf, w_rerank = _blend_weights(c["_rrf_rank"])
            c["score"] = w_rrf * norm_rrf[c["chunk_id"]] + w_rerank * c.get("rerank_score", 0.0)
        ranked.sort(key=lambda c: c["score"], reverse=True)
    else:
        # rrf-only: keep RRF order as-is; expose the RRF score itself as the
        # gating "score" (answer.py applies the collection's rrf_threshold).
        for c in ranked:
            c["score"] = c["rrf_score"]

    # Candidates beyond RERANK_INPUT_MAX (if k*3 overfetch produced more than
    # we sent to the reranker) are appended in their original RRF order,
    # scored purely by RRF — they were never going to outrank the reranked
    # head, but the caller may still want up to k results.
    tail_ids = fused_order[RERANK_INPUT_MAX:]
    for cid in tail_ids:
        ranked.append({
            "chunk_id": cid,
            "text": hydrated[cid]["text"],
            "rrf_score": fused[cid]["score"],
            "score": fused[cid]["score"],
            "source": fused[cid]["source"],
        })

    results: List[Dict[str, Any]] = []
    for c in ranked[:k]:
        cid = c["chunk_id"]
        meta = hydrated[cid]
        results.append({
            "chunk_id": cid,
            "text": meta["text"],
            "score": c["score"],
            "source": c.get("source", "fts"),
            "rrf_score": c.get("rrf_score"),
            "rerank_score": c.get("rerank_score"),
            "mode": mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "document_id": meta["document_id"],
            "source_uri": meta["source_uri"],
            "title": meta["title"],
            "page_or_timecode": meta["page_or_timecode"],
            "section": meta["section"],
            "lang": meta["lang"],
        })
    return results
