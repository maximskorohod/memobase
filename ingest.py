"""Ingestion orchestrator for memobase: extract -> normalize -> chunk ->
BLOCKING secret scan (quarantine) -> embed -> index (FTS + vec).

Also owns:
  * content-hash document dedup (skip re-processing an unchanged source),
  * the RE-INGEST PURGE flow (gap #6): diff the new
    chunk-content-hash set against the document's existing chunk hashes,
    reuse (don't re-embed) unchanged chunks, tombstone chunks whose hash is
    no longer present,
  * an ``ingestion_jobs`` row per call, for observability/resume-scaffolding
    (see gap #10 — v1 does not implement true mid-job resume for local/URL
    ingests, since unlike the v1.x Apify/YouTube ladder there is no
    multi-hour paginated external listing step here; the row still lets a
    caller/UI poll progress and see a clear terminal state),
  * a pre-embed size/cost gate honoring ``memobase.confirm_over_chunks`` and the
    ledger's monthly spend ceiling, BEFORE any money is spent.

Design note on the "pre-ingest size/cost estimate" requirement
(gap #1 calls out that the estimate phase reads the
source too, so it needs the same SSRF/size guard as the real fetch): this
module does NOT do a separate blind pre-fetch. For URL sources,
``extract.extract()`` already performs the one and only network fetch
through ``security.check_url``/``security.safe_get`` (SSRF-guarded,
size-capped, browser UA). The cost/size estimate used below is computed
from that already-fetched, already-guarded text — avoiding a second network
round-trip while still satisfying "same guard for the estimate". The only
NEW external network calls this module itself makes are the embedding
calls, and those are gated on the confirm/budget checks below, applied
strictly after extraction (which costs nothing in $ terms) has completed.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from . import chunk as chunk_mod
from . import config as kb_config
from . import db
from . import embed as embed_mod
from . import enrich as enrich_mod
from . import extract as extract_mod
from . import ledger
from . import normalize as normalize_mod
from . import obsidian as obsidian_mod
from . import security
from . import stem as stem_mod
from . import stt as stt_mod
from . import youtube as youtube_mod

logger = logging.getLogger("memobase.ingest")


class _EmbedderChanged(Exception):
    """The collection's embedder changed (reindex) while this ingest was
    embedding — its vectors belong to the old model; abort the write."""


class IngestError(RuntimeError):
    """Raised only for programmer-error-shaped misuse (malformed
    ``collection_row``). Normal ingest failures (bad source, budget
    refusal, quarantine) are reported via the returned result dict's
    ``status``/``error`` fields, never by raising."""


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Ingestion-source dispatch (pipeline table): every
# source_type is reduced to the same extract.py-shaped {text, blocks, meta,
# skipped} doc before entering the shared normalize -> chunk -> ... pipeline
# below. 'youtube'/'audio'/'video'/'obsidian' are each ONE item here (one
# video / one media file / one note) — whole-channel/whole-vault ingestion
# is a SEPARATE multi-document orchestration living in youtube.ingest_channel
# / obsidian.ingest_vault, which call ingest_source() once per item (see
# tools.py's memobase_ingest handler for the routing between the two).
# ---------------------------------------------------------------------------


def _extract_dispatch(
    source: str, source_type: str, *, memobase_cfg: Dict[str, Any], meta_hint: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """*meta_hint* is an OPTIONAL caller-supplied metadata hint (currently
    only a ``"title"`` key is read anywhere) for source types whose extractor
    can use a piece of metadata the caller already has instead of a worse
    fallback (``youtube.extract_video`` falling back to the bare video id
    when it doesn't know the title). Every other source_type ignores it —
    kept optional/None-default so their extractors' signatures
    (pdf/obsidian/audio/...) are untouched."""
    normalized = (source_type or "").strip().lower()
    if normalized == "youtube":
        return youtube_mod.extract_video(source, memobase_cfg=memobase_cfg, meta_hint=meta_hint)
    if normalized in ("audio", "video"):
        return stt_mod.extract_media(source, source_type=normalized, memobase_cfg=memobase_cfg)
    if normalized == "obsidian":
        return obsidian_mod.extract_note(source)
    return extract_mod.extract(source, source_type)


# ---------------------------------------------------------------------------
# Re-ingest purge (independently testable — see task's smoke-test list)
# ---------------------------------------------------------------------------


def diff_chunk_hashes(old_hashes: Set[str], new_hashes: Set[str]) -> Dict[str, Set[str]]:
    """Pure set-diff helper: ``{'unchanged', 'added', 'removed'}`` hash
    sets, used to decide which chunks to reuse/embed/tombstone on re-ingest."""
    return {
        "unchanged": old_hashes & new_hashes,
        "added": new_hashes - old_hashes,
        "removed": old_hashes - new_hashes,
    }


def purge_removed_chunks(
    conn, *, collection_id: int, document_id: int, keep_hashes: Set[str]
) -> List[int]:
    """Tombstone every non-tombstoned chunk of *document_id* whose
    ``content_sha256`` is NOT in *keep_hashes*; also delete their
    ``chunks_fts`` rows and (if the vec table exists) their vec0 rows.

    Returns the list of tombstoned chunk ids. A pure DB operation — safe to
    unit-test directly against a temp ``memobase.db`` without touching extract/
    normalize/chunk/embed at all.
    """
    with conn:
        return _purge_removed_chunks_in_tx(
            conn, collection_id=collection_id, document_id=document_id, keep_hashes=keep_hashes
        )


def _purge_removed_chunks_in_tx(
    conn, *, collection_id: int, document_id: int, keep_hashes: Set[str]
) -> List[int]:
    """Body of :func:`purge_removed_chunks` WITHOUT its own commit — for use
    inside a caller-owned transaction (ingest_source's single write tx)."""
    rows = conn.execute(
        "SELECT id, content_sha256 FROM chunks "
        "WHERE document_id = ? AND collection_id = ? AND tombstoned_at IS NULL",
        (document_id, collection_id),
    ).fetchall()
    to_remove = [r["id"] for r in rows if r["content_sha256"] not in keep_hashes]
    if not to_remove:
        return []

    placeholders = ",".join("?" for _ in to_remove)
    now = db.now()
    conn.execute(
        f"UPDATE chunks SET tombstoned_at = ? WHERE id IN ({placeholders})",
        (now, *to_remove),
    )
    conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", to_remove)
    if db.vec_table_exists(conn, collection_id):
        vec_table = db.vec_table_name(collection_id)
        conn.execute(f"DELETE FROM {vec_table} WHERE chunk_id IN ({placeholders})", to_remove)
    return to_remove


# ---------------------------------------------------------------------------
# Blocking secret scan (gap #12)
# ---------------------------------------------------------------------------


def _quarantine_scan(chunk_texts: List[str]) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Return ``(blocked_indices, quarantine_report)``. A chunk is blocked
    (quarantined — never embedded, never indexed) if ``scan_secrets`` finds
    any ``high``/``medium``-confidence finding; ``low``-confidence
    (high-entropy-string) findings do not block, per security.py's own
    documented contract."""
    blocked: List[int] = []
    report: List[Dict[str, Any]] = []
    for i, text in enumerate(chunk_texts):
        findings = security.scan_secrets(text)
        blocking = [f for f in findings if f.get("confidence") in ("high", "medium")]
        if blocking:
            blocked.append(i)
            report.append({"chunk_index": i, "findings": blocking})
    return blocked, report


# ---------------------------------------------------------------------------
# MULTIUSER: guest-upload injection quarantine ("Гостевые загрузки =
# недоверенный контент по определению" + gap #24
# "квота гостя на флаг инъекции = карантин с очередью ревью владельца, не
# «проиндексировать с пометкой»"). Distinct from `_quarantine_scan` above
# (secrets — always a hard drop for EVERYONE): this gate only runs for a
# GUEST uploader, and a hit routes the chunk to the `quarantine` DB table
# (stored, pending owner review) instead of a silent drop.
# ---------------------------------------------------------------------------


def _guest_injection_scan(chunk_texts: List[str]) -> List[int]:
    """Return indices of *chunk_texts* whose injection scan found any hit.
    Pure text-in/indices-out — the caller decides what happens to a hit
    (route to ``db.quarantine_insert``); this function never touches the DB
    itself, matching this module's convention of keeping DB writes in the
    orchestration body."""
    flagged: List[int] = []
    for i, text in enumerate(chunk_texts):
        if security.scan_injections(text):
            flagged.append(i)
    return flagged


def estimate_embed_cost_usd(chunk_texts: List[str], collection_cfg: Dict[str, Any]) -> float:
    """Rough $ estimate for embedding *chunk_texts*, using chunk.py's
    approx-token counter. Exposed so tools.py can show a cost estimate
    proactively (e.g. before even calling :func:`ingest_source`), in
    addition to the estimate embedded in a ``needs_confirmation`` result.
    """
    provider = (collection_cfg.get("embedder", {}).get("provider") or "cloudflare").lower()
    total_tokens = sum(chunk_mod.approx_tokens(t) for t in chunk_texts)
    return ledger.estimate_cost_usd(provider, "embed", total_tokens / 1000.0)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def ingest_source(
    conn,
    collection_row: Dict[str, Any],
    source: str,
    source_type: str,
    *,
    memobase_cfg: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
    llm: Optional[Any] = None,
    uploader_user_id: Optional[str] = None,
    meta_hint: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Ingest *source* (a file path or, for ``source_type == "url"``, a
    URL) into the collection described by *collection_row* (a DB row dict
    from ``db.get_collection_by_id``/``by_name``).

    *meta_hint*: see :func:`_extract_dispatch` — an optional metadata hint
    forwarded to the extractor (currently only used by ``source_type ==
    "youtube"``). ``None`` (the default) is a no-op for every source_type,
    including youtube (falls back to its own id/title logic).

    *llm* is an OPTIONAL ``ctx.llm``-shaped handle, used ONLY for JIT
    contextual enrichment (enrich.py) when ``memobase.enrich.enabled`` is
    true — see the embed step below. Omitted
    (``None``, the default), enrichment is silently skipped even if enabled
    in config (no LLM handle, nothing to call), never an error.

    *uploader_user_id*: pass a non-None guest identity to mark this as a
    GUEST upload (gap #24) — the caller
    (tools.py's ``memobase_ingest``, which already knows the requester's
    privilege) is the ONLY thing that decides this; ``ingest_source`` itself
    trusts the flag unconditionally and applies the STRICT, non-optional
    injection-quarantine gate (see ``_guest_injection_scan`` below) whenever
    it is set. Leave it ``None`` for an owner/privileged-operator upload —
    normal behavior, unchanged from v1. Also attributes any embed spend
    recorded during this call to this user_id (ledger, §1.9 gap #8).

    *force*: bypass the content-hash "unchanged" short-circuit below and
    re-run normalize/chunk/embed even when the document's overall text is
    byte-identical to what's already indexed. Does NOT duplicate the
    ``CHUNKER_VERSION`` salt already baked into ``content_hash`` (step 1
    below) — that salt exists so a chunker upgrade auto-invalidates every
    "unchanged" verdict on its own; ``force`` is the MANUAL escape hatch for
    every other reason a caller might want a clean re-run (troubleshooting,
    or a collection-level ``chunk_target_tokens``/``overlap_pct`` change,
    which the document-level content hash cannot see on its own). The
    per-CHUNK content-hash dedup at step 5 still applies regardless — chunk
    text that genuinely didn't change is still reused, never re-embedded, so
    ``force`` never pays to re-embed something that didn't actually change.
    Never bypasses the guest quota/budget gates elsewhere in this function
    (those are independent checks, unaffected by this flag) — a guest
    forcing a re-ingest of a document they're already over-budget for is
    still refused exactly like a normal ingest.

    Returns a plain result dict, always with a ``status`` key, one of:
    ``"done"``, ``"unchanged"`` (content-hash identical to what's already
    indexed — nothing re-processed), ``"needs_confirmation"`` (chunk count
    over ``memobase.confirm_over_chunks``; re-call with ``confirm=True`` to
    proceed), ``"quarantined"`` (every chunk was blocked by the secret
    scanner), or ``"failed"`` (with an ``"error"`` message). A ``"done"``
    result additionally carries best-effort observability extras (never an
    extra network/paid call, just data already computed along the way):
    ``transcript_provider`` (youtube/STT source only, else ``None``),
    ``stt_chunk_count``/``stt_duration_s`` (audio/video STT source only,
    else ``None``), and ``cost_estimate_usd`` (this call's own actual embed
    spend, ``0.0`` if nothing new was embedded).

    Never raises for ordinary failure modes — only :class:`IngestError` for
    a malformed *collection_row* (missing ``id``/``name``), which indicates
    a caller bug, not a data/network problem.
    """
    if not isinstance(collection_row, dict) or "id" not in collection_row or "name" not in collection_row:
        raise IngestError(f"collection_row missing required keys (id/name): {collection_row!r}")

    memobase_cfg = memobase_cfg if memobase_cfg is not None else kb_config.get_memobase_config_readonly()
    collection_cfg = kb_config.get_collection_cfg(collection_row, memobase_cfg=memobase_cfg)
    collection_id = collection_row["id"]

    job_id = db.create_ingestion_job(conn, collection_id=collection_id, kind="ingest", stage="extract")

    def _fail(reason: str, **extra: Any) -> Dict[str, Any]:
        db.update_ingestion_job(conn, job_id, status="failed", stage="failed")
        logger.warning("ingest_source(%r, %r) failed: %s", source, source_type, reason)
        return {"status": "failed", "error": reason, "job_id": job_id, **extra}

    migration_state = collection_cfg.get("migration_state") or "idle"
    if migration_state not in ("idle", "failed", None):
        return _fail(
            f"collection {collection_row['name']!r} is mid-migration "
            f"(migration_state={migration_state!r}); ingest refused until it completes"
        )

    # --- 1. Extract --------------------------------------------------------
    # Paid external calls inside the extractors (Apify / ScrapeCreators /
    # STT) are noted into a bucket and recorded here, attributed to the
    # uploader — whether or not extraction then produced usable text.
    with ledger.collect_external_spend() as spend_bucket:
        doc = _extract_dispatch(source, source_type, memobase_cfg=memobase_cfg, meta_hint=meta_hint)
    external_cost_usd = ledger.flush_external_spend(
        conn, spend_bucket, collection_id=collection_id, user_id=uploader_user_id,
    )
    if uploader_user_id is not None and external_cost_usd:
        stt_seconds = float(((doc.get("meta") or {}).get("duration_s")) or 0.0)
        db.record_guest_usage(conn, uploader_user_id, usd_spent=external_cost_usd, stt_seconds=stt_seconds)
    if not (doc.get("text") or "").strip():
        reasons = "; ".join(s.get("reason", "?") for s in doc.get("skipped", [])) or "no text extracted"
        return _fail(f"extraction produced no text: {reasons}", skipped=doc.get("skipped", []))

    # CHUNKER_VERSION salt: a chunking-algorithm upgrade must invalidate the
    # "unchanged" short-circuit below, otherwise already-ingested documents
    # would never get re-chunked (per-chunk hash dedup further down keeps the
    # re-embed cost limited to chunks whose text actually changed).
    content_hash = _sha256(doc["text"] + "|chunker:" + chunk_mod.CHUNKER_VERSION)

    existing_row = conn.execute(
        "SELECT * FROM documents WHERE collection_id = ? AND source_uri = ?",
        (collection_id, source),
    ).fetchone()
    existing_doc = dict(existing_row) if existing_row is not None else None

    if not force and existing_doc is not None and existing_doc.get("content_sha256") == content_hash:
        db.update_ingestion_job(conn, job_id, status="done", stage="unchanged", items_total=0, items_done=0)
        return {
            "status": "unchanged",
            "job_id": job_id,
            "document_id": existing_doc["id"],
            "chunks_added": 0,
            "chunks_tombstoned": 0,
        }

    # --- 2. Normalize --------------------------------------------------------
    db.update_ingestion_job(conn, job_id, stage="normalize")
    doc = normalize_mod.normalize(doc, profile="default")

    # --- 3. Chunk --------------------------------------------------------
    db.update_ingestion_job(conn, job_id, stage="chunk")
    target_tokens = collection_cfg["chunk"]["target_tokens"]
    overlap_pct = collection_cfg["chunk"]["overlap_pct"]
    raw_chunks = chunk_mod.chunk(doc, target_tokens, overlap_pct)
    if not raw_chunks:
        return _fail("chunking produced zero chunks from non-empty extracted text")

    # --- 4. BLOCKING secret scan (quarantine) --------------------------------------------------------
    db.update_ingestion_job(conn, job_id, stage="secret_scan")
    all_texts = [c["text"] for c in raw_chunks]
    blocked_indices, quarantine_report = _quarantine_scan(all_texts)
    blocked_set = set(blocked_indices)
    usable = [(i, c) for i, c in enumerate(raw_chunks) if i not in blocked_set]
    if not usable:
        db.update_ingestion_job(conn, job_id, status="failed", stage="quarantined")
        if uploader_user_id is not None:
            # §1.9 gap #8: a fully-blocked upload must still count against
            # the guest's daily CALL quota — otherwise a guest could spam
            # secret/injection-triggering uploads for free (no $ spent, but
            # still real work: extraction, chunking, scanning).
            db.record_guest_usage(conn, uploader_user_id, calls=1)
        return {
            "status": "quarantined",
            "job_id": job_id,
            "quarantine": quarantine_report,
            "error": "все фрагменты заблокированы сканером секретов; ничего не загружено, нужна проверка владельцем",
        }

    # --- 4b. GUEST-only STRICT injection quarantine (owner-review queue) --------------------------------------------------------
    # Guest uploads are untrusted content by
    # definition — this gate is NOT configurable off (uploader_user_id is
    # set by tools.py ONLY for a resolved, non-privileged identity). A hit
    # here does not drop the chunk silently (unlike the secret scan above):
    # it is STORED in the `quarantine` table for the owner to approve/reject
    # (`memobase_quarantine_list`/`memobase_quarantine_review`), and simply excluded from
    # `usable` for this run — never embedded, never indexed, until approved.
    guest_quarantined_count = 0
    owner_flagged_injection = 0
    if uploader_user_id is None:
        # Owner upload: trusted enough to index, but still scanned — flagged
        # chunks are reported (and get the warning banner at read time via
        # security.fence_untrusted), never silently passed.
        owner_flagged_injection = sum(1 for _, c in usable if security.scan_injections(c["text"]))
    if uploader_user_id is not None:
        # `_guest_injection_scan` returns positions in the list it is given;
        # map them back to the ORIGINAL raw_chunks indices carried in `usable`
        # (after the secret scan dropped some chunks the two diverge, and
        # comparing compacted positions with raw indices let a flagged chunk
        # slip through unquarantined).
        hit_positions = _guest_injection_scan([c["text"] for _, c in usable])
        injection_hits = {usable[pos][0] for pos in hit_positions}
        if injection_hits:
            still_usable = []
            for i, c in usable:
                if i in injection_hits:
                    db.quarantine_insert(
                        conn,
                        collection_id=collection_id,
                        uploader_user_id=uploader_user_id,
                        source_uri=source,
                        chunk_index=i,
                        text=c["text"],
                        findings=security.scan_injections(c["text"]),
                    )
                    guest_quarantined_count += 1
                else:
                    still_usable.append((i, c))
            usable = still_usable
        if not usable:
            db.update_ingestion_job(conn, job_id, status="failed", stage="quarantined")
            db.record_guest_usage(conn, uploader_user_id, calls=1)
            return {
                "status": "quarantined",
                "job_id": job_id,
                "quarantine_injection_count": guest_quarantined_count,
                "error": (
                    "все фрагменты этой гостевой загрузки заблокированы сканером инъекций; "
                    "ничего не загружено, нужна проверка владельцем (memobase_quarantine_list)"
                ),
            }

    # --- 5. Chunk-level content-hash dedup against existing chunks --------------------------------------------------------
    # (re-ingest of an edited document: unchanged paragraphs are reused —
    # not re-embedded, not re-inserted — new paragraphs are embedded, and
    # step 8 below tombstones whatever old chunk hash is no longer present.)
    new_hash_by_index: Dict[int, str] = {i: _sha256(c["text"]) for i, c in usable}
    existing_chunk_rows: List[Dict[str, Any]] = []
    if existing_doc is not None:
        existing_chunk_rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? AND collection_id = ? AND tombstoned_at IS NULL",
                (existing_doc["id"], collection_id),
            ).fetchall()
        ]
    existing_hash_to_row = {
        r["content_sha256"]: r for r in existing_chunk_rows if r.get("content_sha256")
    }

    to_embed = [(i, c) for i, c in usable if new_hash_by_index[i] not in existing_hash_to_row]
    reused_hashes = {new_hash_by_index[i] for i, _ in usable} & set(existing_hash_to_row.keys())

    # --- 6. Pre-embed size/cost gate --------------------------------------------------------
    new_chunk_count = len(to_embed)
    confirm_threshold = memobase_cfg.get("confirm_over_chunks", 500)
    if new_chunk_count > confirm_threshold and not confirm:
        estimated_cost = estimate_embed_cost_usd([c["text"] for _, c in to_embed], collection_cfg)
        db.update_ingestion_job(
            conn, job_id, status="done", stage="needs_confirmation", items_total=new_chunk_count
        )
        return {
            "status": "needs_confirmation",
            "job_id": job_id,
            "new_chunk_count": new_chunk_count,
            "estimated_cost_usd": round(estimated_cost, 6),
            "message": (
                f"{new_chunk_count} новых фрагментов нужно проиндексировать "
                f"(порог подтверждения: {confirm_threshold}, оценка стоимости: "
                f"${estimated_cost:.4f}). Повторите запрос с подтверждением, чтобы продолжить."
            ),
        }

    provider = (collection_cfg["embedder"].get("provider") or "cloudflare").lower()
    try:
        ledger.ensure_within_ceiling(
            conn, provider, memobase_cfg,
            estimated_usd=estimate_embed_cost_usd([c["text"] for _, c in to_embed], collection_cfg),
        )
    except ledger.LedgerError as exc:
        return _fail(str(exc))

    # --- 6b. GUEST daily-$-budget gate, BEFORE the paid embed call --------------------------------------------------------
    # Gap #8: "проверяется до отправки в Apify/Groq/
    # embed" — tools.py's memobase_ingest already does a coarser pre-dispatch
    # check on a rough size estimate; this one re-checks against the actual
    # post-chunking cost, right before the money is spent, for the same
    # (uploader_user_id is not None ⇒ guest) requests.
    if uploader_user_id is not None and to_embed:
        estimated_now = estimate_embed_cost_usd([c["text"] for _, c in to_embed], collection_cfg)
        guest_quota = security.effective_guest_quota(memobase_cfg, db.get_guest_quota(conn, uploader_user_id))
        spent_today = db.get_guest_usage_today(conn, uploader_user_id)["usd_spent"]
        budget_check = security.check_daily_budget_quota(
            guest_quota, used_usd_today=spent_today, estimated_usd=estimated_now
        )
        if not budget_check.ok:
            return _fail(f"дневной бюджет гостя исчерпан: {budget_check.reason}")

    # --- 7. Embed only the genuinely-new chunks --------------------------------------------------------
    db.update_ingestion_job(conn, job_id, stage="embed", items_total=new_chunk_count)
    embed_signature = embed_mod.embedding_signature(collection_cfg)
    new_vectors: Dict[int, List[float]] = {}
    enrichment_by_index: Dict[int, Optional[str]] = {}
    actual_embed_cost_usd = 0.0
    if to_embed:
        texts_to_embed = [c["text"] for _, c in to_embed]
        texts_for_embedder = texts_to_embed
        # JIT contextual enrichment (enrich.py): the enrichment prefix is used
        # ONLY for this embedder call —
        # `texts_to_embed` (raw) is still what gets stored in `chunks.text`
        # below (step 9), never the enriched variant.
        if enrich_mod.is_enabled(memobase_cfg) and llm is not None:
            doc_meta_preview = doc.get("meta") or {}
            doc_context = {"title": doc_meta_preview.get("title"), "source_uri": source}
            enrich_model = ((memobase_cfg.get("enrich") or {}).get("model") or None)
            try:
                texts_for_embedder, enrichment_strings = enrich_mod.enrich_chunks_for_embedding(
                    texts_to_embed, doc_context, llm=llm, model=enrich_model
                )
            except Exception:  # noqa: BLE001 - enrichment must never block an otherwise-good ingest
                logger.warning("chunk enrichment batch failed; embedding without enrichment", exc_info=True)
                texts_for_embedder = texts_to_embed
                enrichment_strings = [None] * len(texts_to_embed)
            for (i, _c), note in zip(to_embed, enrichment_strings):
                enrichment_by_index[i] = note
        try:
            vectors = embed_mod.embed_texts(texts_for_embedder, collection_cfg)
        except embed_mod.EmbedError as exc:
            return _fail(f"embedding failed: {exc}")
        total_tokens = sum(chunk_mod.approx_tokens(t) for t in texts_for_embedder)
        actual_embed_cost_usd = ledger.estimate_cost_usd(provider, "embed", total_tokens / 1000.0)
        ledger.record_call(
            conn, provider=provider, op="embed", est_usd=actual_embed_cost_usd,
            units=total_tokens / 1000.0, collection_id=collection_id, user_id=uploader_user_id,
        )
        for (i, _c), vec in zip(to_embed, vectors):
            new_vectors[i] = vec

    # --- 8-10. ONE write transaction: document row + chunks/FTS/vec + purge ---
    # The document's content_sha256 is what the "unchanged" short-circuit
    # (step 1) trusts. It used to be committed BEFORE the chunk inserts, so
    # any failure in between left a document that looked "unchanged" but had
    # no chunks — forever. Now the hash, the chunks and the purge commit
    # together or not at all (db.immediate_transaction: explicit BEGIN
    # IMMEDIATE, rolled back on any exception).
    now = db.now()
    meta = doc.get("meta") or {}
    dims = collection_cfg["embedder"].get("dims")
    inserted_chunk_ids: List[int] = []
    tombstoned_ids: List[int] = []
    try:
        # DDL (CREATE VIRTUAL TABLE) stays outside the data transaction.
        vec_ready = bool(to_embed) and bool(dims) and db.ensure_vec_table(conn, collection_id, dims)
        with db.immediate_transaction(conn):
            # Re-check under the write lock: a reindex that swapped the
            # vector table while we were embedding would otherwise receive
            # vectors from the OLD model.
            live = conn.execute(
                "SELECT embedder_provider, embedder_model, embedder_dims, migration_state "
                "FROM collections WHERE id = ?", (collection_id,),
            ).fetchone()
            if live is None or (live["migration_state"] or "idle") not in ("idle", "failed") or (
                kb_config.get_collection_cfg(dict(collection_row, **dict(live)), memobase_cfg=memobase_cfg)["embedder"]
                != collection_cfg["embedder"]
            ):
                raise _EmbedderChanged()
            if existing_doc is not None:
                document_id = existing_doc["id"]
                conn.execute(
                    """
                    UPDATE documents
                       SET content_sha256 = ?, title = ?, page_count = ?, ingested_at = ?, superseded_at = NULL
                     WHERE id = ?
                    """,
                    (content_hash, meta.get("title"), meta.get("pages"), now, document_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO documents(collection_id, source_uri, source_type, content_sha256,
                                           title, page_count, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (collection_id, source, source_type, content_hash, meta.get("title"), meta.get("pages"), now),
                )
                document_id = int(cur.lastrowid)

            for i, c in to_embed:
                content_sha = new_hash_by_index[i]
                lang = normalize_mod.detect_lang(c["text"])
                page_val = c.get("page_or_timecode")
                cur = conn.execute(
                    """
                    INSERT INTO chunks(collection_id, document_id, seq, text, content_sha256,
                                        page_or_timecode, section, lang, embed_signature, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        collection_id,
                        document_id,
                        c["seq"],
                        c["text"],
                        content_sha,
                        str(page_val) if page_val is not None else None,
                        c.get("section"),
                        lang,
                        embed_signature,
                        now,
                    ),
                )
                chunk_id = int(cur.lastrowid)
                inserted_chunk_ids.append(chunk_id)

                text_stem = stem_mod.stem_ru(c["text"])
                conn.execute(
                    "INSERT INTO chunks_fts(text, text_stem, chunk_id, collection_id) VALUES (?, ?, ?, ?)",
                    (c["text"], text_stem, chunk_id, collection_id),
                )
                if vec_ready and i in new_vectors:
                    vec_table = db.vec_table_name(collection_id)
                    conn.execute(
                        f"INSERT OR REPLACE INTO {vec_table}(chunk_id, embedding) VALUES (?, ?)",
                        (chunk_id, embed_mod.serialize_vector(new_vectors[i])),
                    )
                if enrichment_by_index.get(i):
                    # Inlined (not db.record_chunk_enrichment, which opens its
                    # own `with conn:` and would commit this transaction early).
                    conn.execute(
                        "INSERT INTO chunk_enrichment(chunk_id, collection_id, enrichment_text, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (chunk_id, collection_id, enrichment_by_index[i], now),
                    )

            if existing_doc is not None:
                keep_hashes = {new_hash_by_index[i] for i, _ in usable}
                tombstoned_ids = _purge_removed_chunks_in_tx(
                    conn, collection_id=collection_id, document_id=document_id, keep_hashes=keep_hashes
                )
    except _EmbedderChanged:
        return _fail("коллекция переиндексируется (сменилась модель эмбеддингов) — повторите загрузку позже")
    except (sqlite3.Error, db.DbError) as exc:
        # Nothing of this document was committed (rolled back above); the
        # embed spend already recorded stays in the ledger (it was paid).
        logger.error("ingest_source: index write failed for %r", source, exc_info=True)
        return _fail(f"запись в индекс не удалась ({type(exc).__name__}); документ не изменён")

    db.update_ingestion_job(
        conn, job_id, status="done", stage="done", items_done=len(inserted_chunk_ids), items_total=new_chunk_count
    )

    if uploader_user_id is not None:
        # §1.9 gap #8: today's usage must reflect THIS call before the NEXT
        # one's pre-checks run — bytes_uploaded is approximated from the
        # newly embedded chunks' raw text size (what actually got processed),
        # not the whole source (which may include reused/unchanged content).
        uploaded_bytes = sum(len((c["text"] or "").encode("utf-8")) for _, c in to_embed)
        db.record_guest_usage(
            conn, uploader_user_id, bytes_uploaded=uploaded_bytes, calls=1, usd_spent=actual_embed_cost_usd,
        )

    return {
        "status": "done",
        "job_id": job_id,
        "document_id": document_id,
        "chunks_added": len(inserted_chunk_ids),
        "chunks_reused": len(reused_hashes),
        "chunks_tombstoned": len(tombstoned_ids),
        "chunks_quarantined": len(blocked_indices),
        "chunks_quarantined_injection": guest_quarantined_count,
        "chunks_flagged_injection": owner_flagged_injection,
        "quarantine": quarantine_report,
        "vector_index_ready": vec_ready,
        # Observability extras (best-effort, read from data this call already
        # computed/paid for -- never a new network/paid call). `meta` (step 8
        # above) carries transcript_provider (youtube.py's build_video_doc)
        # or stt_provider (stt.py's extract_media) -- different key names at
        # the source because they're different extractors, unified here
        # under one name for the caller. Renamed to stt_chunk_count/
        # stt_duration_s (not chunk_count/duration_s, stt.py's own internal
        # names) specifically to avoid colliding with this SAME dict's
        # chunks_added/chunks_reused/... family above, which count TEXT
        # chunks -- a wholly different thing from audio chunks.
        "transcript_provider": meta.get("transcript_provider") or meta.get("stt_provider"),
        "stt_chunk_count": meta.get("chunk_count"),
        "stt_duration_s": meta.get("duration_s"),
        "cost_estimate_usd": round(actual_embed_cost_usd + external_cost_usd, 6),
        "external_cost_usd": round(external_cost_usd, 6),
    }


# ---------------------------------------------------------------------------
# MULTIUSER: owner approval of a quarantined guest chunk (tools.py's
# `memobase_quarantine_review`, gap #24). A single-
# chunk analog of the main pipeline's steps 4/7/9 — deliberately NOT routed
# back through `ingest_source` (there is no "source" to re-extract; the text
# was already extracted, chunked, and stored verbatim in `quarantine.text`
# at upload time) — instead re-runs just the parts that still apply: a
# defense-in-depth secret re-scan, embed, and chunk/FTS/vec insert, reusing
# an existing `documents` row for the same source_uri when there is one so
# this chunk is attributed to the right document rather than orphaned.
# ---------------------------------------------------------------------------


def approve_quarantined_chunk(
    conn, collection_row: Dict[str, Any], quarantine_row: Dict[str, Any], memobase_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Owner-approved: embed + index *quarantine_row*'s text into
    *collection_row*. Returns ``{"status": "done", "chunk_id": ..., "reused":
    bool}`` on success (``reused=True`` if an identical live chunk already
    existed — content-addressable dedup, same rule as the main pipeline) or
    ``{"status": "failed", "error": ...}``. Never marks the quarantine row
    itself reviewed — the caller (tools.py) does that only after this
    succeeds, so a failed approval leaves the item ``pending`` for a retry.
    """
    collection_id = collection_row["id"]
    text = quarantine_row["text"] or ""
    source_uri = quarantine_row.get("source_uri") or "quarantine-review"
    uploader_user_id = quarantine_row.get("uploader_user_id")

    # Defense in depth: the secret scanner already ran once at the original
    # upload attempt (this chunk only got here because it was flagged by the
    # INJECTION scanner, not the secret one) — re-check anyway in case the
    # two were somehow bypassed independently; never skip this gate just
    # because it's a "re-approval" path.
    secret_findings = security.scan_secrets(text)
    if any(f.get("confidence") in ("high", "medium") for f in secret_findings):
        return {"status": "failed", "error": "текст всё ещё блокируется сканером секретов"}

    content_hash = _sha256(text)
    dup = conn.execute(
        "SELECT id FROM chunks WHERE collection_id = ? AND content_sha256 = ? AND tombstoned_at IS NULL",
        (collection_id, content_hash),
    ).fetchone()
    if dup is not None:
        return {"status": "done", "chunk_id": int(dup["id"]), "reused": True}

    collection_cfg = kb_config.get_collection_cfg(collection_row, memobase_cfg=memobase_cfg)
    provider = (collection_cfg["embedder"].get("provider") or "cloudflare").lower()
    try:
        ledger.ensure_within_ceiling(conn, provider, memobase_cfg)
        vectors = embed_mod.embed_texts([text], collection_cfg)
    except (ledger.LedgerError, embed_mod.EmbedError) as exc:
        return {"status": "failed", "error": str(exc)}

    total_tokens = chunk_mod.approx_tokens(text)
    embed_cost = ledger.estimate_cost_usd(provider, "embed", total_tokens / 1000.0)
    ledger.record_call(
        conn, provider=provider, op="embed", est_usd=embed_cost, units=total_tokens / 1000.0,
        collection_id=collection_id, user_id=uploader_user_id,
    )

    now = db.now()
    existing_doc = conn.execute(
        "SELECT * FROM documents WHERE collection_id = ? AND source_uri = ?",
        (collection_id, source_uri),
    ).fetchone()
    embed_signature = embed_mod.embedding_signature(collection_cfg)
    dims = collection_cfg["embedder"].get("dims")
    vec_ready = bool(dims) and db.ensure_vec_table(conn, collection_id, dims)

    with conn:
        if existing_doc is not None:
            document_id = int(existing_doc["id"])
        else:
            cur = conn.execute(
                "INSERT INTO documents(collection_id, source_uri, source_type, content_sha256, ingested_at) "
                "VALUES (?, ?, 'quarantine_review', ?, ?)",
                (collection_id, source_uri, content_hash, now),
            )
            document_id = int(cur.lastrowid)

        seq_row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchone()
        seq = int(seq_row["next_seq"])

        cur = conn.execute(
            """
            INSERT INTO chunks(collection_id, document_id, seq, text, content_sha256,
                                page_or_timecode, section, lang, embed_signature, created_at)
            VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
            """,
            (collection_id, document_id, seq, text, content_hash, normalize_mod.detect_lang(text), embed_signature, now),
        )
        chunk_id = int(cur.lastrowid)

        text_stem = stem_mod.stem_ru(text)
        conn.execute(
            "INSERT INTO chunks_fts(text, text_stem, chunk_id, collection_id) VALUES (?, ?, ?, ?)",
            (text, text_stem, chunk_id, collection_id),
        )
        if vec_ready:
            vec_table = db.vec_table_name(collection_id)
            conn.execute(
                f"INSERT OR REPLACE INTO {vec_table}(chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, embed_mod.serialize_vector(vectors[0])),
            )

    if uploader_user_id is not None:
        db.record_guest_usage(conn, uploader_user_id, usd_spent=embed_cost)

    return {"status": "done", "chunk_id": chunk_id, "reused": False}
