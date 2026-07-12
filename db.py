"""SQLite schema and connection management for memobase.

Single database file: ``<HERMES_HOME>/memobase/memobase.db`` (WAL mode). Collection is a
COLUMN, not a directory — DESIGN_v1.md calls this out explicitly as safer
than per-directory collections, since there is no path-traversal surface on
the DB path itself (only on per-collection *file* operations elsewhere, which
is what ``security.valid_collection_name``/``safe_collection_path`` guard).

Vector search uses one ``vec_c{collection_id}`` virtual table PER collection
(sqlite-vec's ``vec0`` extension), created lazily once a collection's
embedding dimensionality is known. sqlite-vec is an OPTIONAL dependency here
by design — every function that touches it degrades to "vector search
unavailable" rather than raising, so FTS5-only search keeps working even
before ``install.ps1`` has run or if the extension fails to load.

Defensiveness contract (per task's non-negotiables — "all wrapped
defensively"):
  * Schema/DDL and connection setup raise :class:`DbError` with context on
    failure — callers (ingest jobs, tool handlers) are expected to catch
    this and report an honest failure, not crash silently.
  * ``ensure_vec_table`` never raises for a *missing/unloadable extension* —
    only for a genuinely corrupt schema state after the extension loaded
    successfully. Its return value (bool) is the "is vector search ready"
    signal.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("memobase.db")

# Bump this and add a migration step to `_MIGRATIONS` when the schema changes.
#
# v2 (MULTIUSER + OPS round): adds `spend.user_id` (per-guest budget
# attribution — gap #8) via an ALTER TABLE migration
# below, since `CREATE TABLE IF NOT EXISTS` is a no-op against an
# already-existing `spend` table on upgraded installs. The new
# collection_shares/guest_quotas/guest_usage_daily/quarantine tables need NO
# migration entry — they are brand-new tables, so their `CREATE TABLE IF NOT
# EXISTS` in DDL_STATEMENTS already covers both fresh and upgraded installs.
SCHEMA_VERSION = 2

DB_BUSY_TIMEOUT_MS = 5000
CONNECT_TIMEOUT_S = 5.0


class DbError(RuntimeError):
    """Raised for memobase database failures the caller must not ignore."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def get_kb_dir() -> Path:
    """Return ``<HERMES_HOME>/memobase`` (created on first connect, not here)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "memobase"


def get_db_path() -> Path:
    """Return ``<HERMES_HOME>/memobase/memobase.db``."""
    return get_kb_dir() / "memobase.db"


# ---------------------------------------------------------------------------
# Schema DDL — verbatim from DESIGN_v1.md "DB schema" section
# ---------------------------------------------------------------------------

DDL_STATEMENTS: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS collections (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        owner_user_id TEXT,
        visibility TEXT NOT NULL DEFAULT 'private',
        embedder_provider TEXT,
        embedder_model TEXT,
        embedder_dims INTEGER,
        chunk_target_tokens INTEGER NOT NULL DEFAULT 900,
        chunk_overlap_pct REAL NOT NULL DEFAULT 0.15,
        rrf_threshold REAL,
        rerank_threshold REAL,
        migration_state TEXT NOT NULL DEFAULT 'idle',
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL REFERENCES collections(id),
        source_uri TEXT NOT NULL,
        source_type TEXT,
        content_sha256 TEXT,
        title TEXT,
        page_count INTEGER,
        ingested_at REAL,
        superseded_at REAL,
        UNIQUE(collection_id, source_uri)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL REFERENCES collections(id),
        document_id INTEGER NOT NULL REFERENCES documents(id),
        seq INTEGER NOT NULL,
        text TEXT NOT NULL,
        content_sha256 TEXT,
        page_or_timecode TEXT,
        section TEXT,
        lang TEXT,
        embed_signature TEXT,
        tombstoned_at REAL,
        created_at REAL NOT NULL
    )
    """,
    # NOTE on "external-content": DESIGN_v1.md's comment calls this
    # "FTS5 external-content over chunks.text" loosely, but the literal
    # CREATE below has no `content='chunks'` clause — it CANNOT, because
    # `text_stem` is a derived column with no counterpart in `chunks`
    # (true FTS5 external-content tables require every indexed column to
    # map to a real column of the content table). This is therefore a
    # normal (self-contained) FTS5 table that ingest.py populates
    # explicitly alongside `chunks` inserts/deletes/tombstones, keyed by
    # the UNINDEXED `chunk_id` column, not by matching rowids. Downstream
    # modules must keep the two in sync manually (insert into both in the
    # same transaction; on tombstone, DELETE FROM chunks_fts WHERE chunk_id=?).
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text,
        text_stem,
        chunk_id UNINDEXED,
        collection_id UNINDEXED,
        tokenize='unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_jobs (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL REFERENCES collections(id),
        kind TEXT NOT NULL,
        external_run_id TEXT,
        stage TEXT,
        items_total INTEGER,
        items_done INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        started_at REAL,
        updated_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS spend (
        id INTEGER PRIMARY KEY,
        ts REAL NOT NULL,
        provider TEXT NOT NULL,
        op TEXT NOT NULL,
        units REAL,
        est_usd REAL,
        collection_id INTEGER REFERENCES collections(id),
        user_id TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS _meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    # enrich.py's module docstring: the
    # JIT contextual-enrichment string is persisted here for a future
    # `--explain` trace, kept OUT of `chunks.text` on purpose (citation
    # verification always reads the raw stored chunk text).
    """
    CREATE TABLE IF NOT EXISTS chunk_enrichment (
        id INTEGER PRIMARY KEY,
        chunk_id INTEGER NOT NULL REFERENCES chunks(id),
        collection_id INTEGER NOT NULL REFERENCES collections(id),
        enrichment_text TEXT,
        created_at REAL NOT NULL
    )
    """,
    # --- MULTIUSER ----------------------------------------------------------
    # ACL grants a non-owner identity (`user_id`, resolved out-of-band from
    # gateway identity, never from a model-supplied argument — see tools.py's
    # `_resolve_identity`) read or write access to a collection they do not
    # own. Absence of a row (and not being the owner) means NO access — the
    # default-deny documented in §1.4 ("ваши private-коллекции для гостей не
    # существуют").
    """
    CREATE TABLE IF NOT EXISTS collection_shares (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL REFERENCES collections(id),
        user_id TEXT NOT NULL,
        permission TEXT NOT NULL DEFAULT 'read',
        granted_by TEXT,
        created_at REAL NOT NULL,
        UNIQUE(collection_id, user_id)
    )
    """,
    # Per-guest quota OVERRIDES (§1.9 gap #8: storage quota is not enough --
    # a guest also needs a daily $/upload/STT ceiling, checked BEFORE the
    # costly external call, not just before the chunk write). A missing row
    # for a given user_id means "use the config's memobase.guest_defaults.*"
    # (security.py's `effective_guest_quota` merges the two).
    """
    CREATE TABLE IF NOT EXISTS guest_quotas (
        id INTEGER PRIMARY KEY,
        user_id TEXT NOT NULL UNIQUE,
        max_mb REAL,
        max_chunks INTEGER,
        daily_upload_mb REAL,
        daily_budget_usd REAL,
        daily_calls INTEGER,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    # Rolling per-UTC-day usage counters a guest's quota check compares
    # against. One row per (user_id, day); `record_guest_usage` upserts by
    # adding to the existing row rather than overwriting.
    """
    CREATE TABLE IF NOT EXISTS guest_usage_daily (
        id INTEGER PRIMARY KEY,
        user_id TEXT NOT NULL,
        usage_date TEXT NOT NULL,
        bytes_uploaded INTEGER NOT NULL DEFAULT 0,
        calls INTEGER NOT NULL DEFAULT 0,
        usd_spent REAL NOT NULL DEFAULT 0,
        stt_seconds REAL NOT NULL DEFAULT 0,
        UNIQUE(user_id, usage_date)
    )
    """,
    # Owner-review queue for guest uploads flagged by the injection scanner
    # in STRICT mode (§1.9 gap #13 / §1.4: "квота гостя на флаг инъекции =
    # карантин с очередью ревью владельца, не «проиндексировать с
    # пометкой»"). Distinct from the SECRET-scanner quarantine in ingest.py
    # (which is a hard drop, never stored anywhere) -- a chunk landing here
    # IS stored (so the owner can actually review it) but is NEVER embedded
    # or indexed until approved.
    """
    CREATE TABLE IF NOT EXISTS quarantine (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL REFERENCES collections(id),
        uploader_user_id TEXT,
        source_uri TEXT,
        chunk_index INTEGER,
        text TEXT NOT NULL,
        findings_json TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at REAL NOT NULL,
        reviewed_at REAL,
        reviewed_by TEXT
    )
    """,
]

# Additive, non-spec indexes for the query patterns every downstream module
# needs (retrieval by collection, job polling by collection, spend rollups
# by time). These do not change any spec'd column/table shape.
INDEX_STATEMENTS: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_collection_document ON chunks(collection_id, document_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(collection_id, content_sha256)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_tombstoned ON chunks(collection_id, tombstoned_at)",
    "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_collection ON ingestion_jobs(collection_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_spend_ts ON spend(ts)",
    "CREATE INDEX IF NOT EXISTS idx_spend_collection ON spend(collection_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunk_enrichment_chunk ON chunk_enrichment(chunk_id)",
    "CREATE INDEX IF NOT EXISTS idx_collection_shares_collection ON collection_shares(collection_id)",
    "CREATE INDEX IF NOT EXISTS idx_collection_shares_user ON collection_shares(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_guest_usage_daily_user_date ON guest_usage_daily(user_id, usage_date)",
    "CREATE INDEX IF NOT EXISTS idx_quarantine_collection_status ON quarantine(collection_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_spend_user ON spend(user_id)",
]


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def _configure_connection(conn: sqlite3.Connection) -> None:
    """Apply the project's mandatory PRAGMAs and row factory.

    WAL + busy_timeout=5000 + synchronous=NORMAL per the task's exact spec
    (readers do not block on a writer holding the WAL; writers wait up to
    5s instead of raising "database is locked" immediately).
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row


def connect(db_path: Optional[Path] = None, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a raw connection to ``memobase.db``.

    ``read_only=True`` opens via a ``file:...?mode=ro`` URI — for
    status/polling call sites that must never become the WAL writer by
    accident (mirrors the ``hermes_state.SessionDB(read_only=True)``
    pattern). WAL already lets ordinary readers run concurrently with a
    writer; ``read_only`` is an extra guarantee, not a workaround for lock
    contention.

    Raises :class:`DbError` on any failure to open/configure — never lets a
    raw ``sqlite3.Error`` escape this module.
    """
    path = db_path or get_db_path()

    if read_only:
        if not path.exists():
            raise DbError(
                f"memobase.db not found at {path} (read-only open requested before "
                f"any write has created it)"
            )
        uri = f"file:{path.as_posix()}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=CONNECT_TIMEOUT_S)
        except sqlite3.Error as exc:
            raise DbError(f"failed to open memobase.db read-only at {path}: {exc}") from exc
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DbError(f"failed to create kb dir {path.parent}: {exc}") from exc
        try:
            conn = sqlite3.connect(str(path), timeout=CONNECT_TIMEOUT_S)
        except sqlite3.Error as exc:
            raise DbError(f"failed to open memobase.db at {path}: {exc}") from exc

    try:
        _configure_connection(conn)
    except sqlite3.Error as exc:
        conn.close()
        raise DbError(f"failed to configure memobase.db connection: {exc}") from exc

    # Best-effort: try to load sqlite-vec on EVERY new connection, not just
    # the ones that happen to call ensure_vec_table()/embed a query first.
    # sqlite-vec is a runtime-loaded extension (not a persisted file format
    # feature) -- loading it on the connection that CREATED a vec_c{id}
    # table does not make it available on a different, later connection to
    # the same memobase.db. Without this, any fresh connection that touches an
    # EXISTING vec0 table (e.g. delete_collection()'s "DROP TABLE
    # vec_c{id}", or swap_vec_table()'s rename) fails with sqlite3's
    # "no such module: vec0", even though the table genuinely exists and
    # was created successfully by an earlier connection. Harmless/no-op
    # when the sqlite_vec package isn't installed (load_sqlite_vec never
    # raises); retrieve.py's _vec_ready() probe-then-load stays as a cheap
    # per-call confirmation, not a redundant second load, since this call
    # already succeeded (or already failed) once per connection.
    load_sqlite_vec(conn)
    return conn


def get_connection(*, read_only: bool = False, ensure_schema: bool = True, db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Convenience wrapper: :func:`connect` + (optionally) :func:`init_schema`.

    ``ensure_schema`` is ignored (forced off) when ``read_only=True`` — a
    read-only connection cannot CREATE TABLE, and callers on that path are
    expected to run against an already-initialized database (the writer
    side calls this with defaults at plugin/tool-handler startup).
    """
    conn = connect(db_path=db_path, read_only=read_only)
    if ensure_schema and not read_only:
        init_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Schema init / migrations
# ---------------------------------------------------------------------------


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables/indexes/FTS if missing, then run any pending
    migrations. Idempotent — safe to call on every plugin/tool-handler
    startup.
    """
    try:
        with conn:
            for stmt in DDL_STATEMENTS:
                conn.execute(stmt)
            for stmt in INDEX_STATEMENTS:
                conn.execute(stmt)
        _ensure_meta_defaults(conn)
        migrate(conn)
    except sqlite3.Error as exc:
        raise DbError(f"failed to initialize kb schema: {exc}") from exc


def _ensure_meta_defaults(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        with conn:
            conn.execute(
                "INSERT INTO _meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )


def get_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def _add_spend_user_id_column(conn: sqlite3.Connection) -> None:
    """v1 -> v2: add ``spend.user_id`` (per-guest budget attribution).

    Defensive column-existence check even though this only ever runs once
    per DB (gated by ``_meta.schema_version``) — cheap, and matches this
    project's "never crash on an unexpected repeat" convention rather than
    relying solely on the version gate.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(spend)").fetchall()}
    if "user_id" not in cols:
        conn.execute("ALTER TABLE spend ADD COLUMN user_id TEXT")


# Migration steps: (target_version, list_of_statements_or_callables).
# Callables receive the connection and run inside the same transaction.
_MIGRATIONS: List[tuple] = [
    (2, [_add_spend_user_id_column]),
]


def migrate(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations in ``_MIGRATIONS``, in order, updating
    ``_meta.schema_version`` after each. Never raises for "nothing to do".
    """
    current = get_schema_version(conn)
    for target_version, steps in _MIGRATIONS:
        if target_version <= current:
            continue
        try:
            with conn:
                for step in steps:
                    if callable(step):
                        step(conn)
                    else:
                        conn.execute(step)
                conn.execute(
                    "INSERT INTO _meta(key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(target_version),),
                )
            current = target_version
        except sqlite3.Error as exc:
            raise DbError(f"migration to schema version {target_version} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Per-collection vec0 (sqlite-vec) table — optional dependency
# ---------------------------------------------------------------------------

_VEC_TABLE_NAME_RE = re.compile(r"^vec_c\d+(?:_v2)?$")


def vec_table_name(collection_id: int, *, shadow: bool = False) -> str:
    """Return ``vec_c{id}`` (or ``vec_c{id}_v2`` for the shadow-table
    migration path).

    Validates ``collection_id`` is a plain non-negative int before
    interpolating it into SQL — this name is used in ``CREATE VIRTUAL
    TABLE``/``DROP TABLE`` statements built by string formatting (sqlite3
    does not support parameterized identifiers), so a bad id here is an SQL
    injection surface, not just a cosmetic bug.
    """
    if not isinstance(collection_id, int) or isinstance(collection_id, bool) or collection_id < 0:
        raise DbError(f"invalid collection_id for vec table name: {collection_id!r}")
    name = f"vec_c{collection_id}_v2" if shadow else f"vec_c{collection_id}"
    if not _VEC_TABLE_NAME_RE.match(name):  # defense in depth; should be unreachable
        raise DbError(f"generated vec table name failed validation: {name!r}")
    return name


def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension into *conn*. Returns False (never
    raises) if the ``sqlite_vec`` Python package is not installed or the
    native extension fails to load for any reason — callers must treat that
    as "vector search unavailable, use FTS-only" per DESIGN_v1.md's
    degradation contract, not as a fatal error.
    """
    try:
        import sqlite_vec  # optional dependency; heavy/native import kept local
    except ImportError:
        logger.warning("sqlite-vec package not installed; vector search disabled (FTS-only)")
        return False

    try:
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
    except (sqlite3.Error, AttributeError, OSError) as exc:
        logger.warning("sqlite-vec extension failed to load: %s", exc)
        return False
    return True


def ensure_vec_table(conn: sqlite3.Connection, collection_id: int, dims: int, *, shadow: bool = False) -> bool:
    """Create the per-collection ``vec0`` virtual table if missing.

    Returns True iff the table exists and is ready for use after this call.
    Returns False (never raises) when sqlite-vec cannot be loaded — the
    caller falls back to FTS-only search. Raises :class:`DbError` only for a
    genuinely corrupt schema state (extension loaded fine, but CREATE
    VIRTUAL TABLE itself failed for a reason other than "already exists").
    """
    if not isinstance(dims, int) or dims <= 0:
        raise DbError(f"invalid dims for vec table: {dims!r}")
    table = vec_table_name(collection_id, shadow=shadow)

    if not load_sqlite_vec(conn):
        return False

    try:
        with conn:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
                f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dims}])"
            )
    except sqlite3.Error as exc:
        raise DbError(f"failed to create vec table {table}: {exc}") from exc
    return True


def vec_table_exists(conn: sqlite3.Connection, collection_id: int, *, shadow: bool = False) -> bool:
    table = vec_table_name(collection_id, shadow=shadow)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def swap_vec_table(conn: sqlite3.Connection, collection_id: int) -> None:
    """Atomically promote the shadow table (``vec_c{id}_v2``) to live
    (``vec_c{id}``), for the embedding-migration flow. Caller (embed.py) is
    responsible for having fully re-embedded into the shadow table and
    updating ``collections.migration_state``/embed_signature BEFORE calling
    this — this function only does the rename, inside one transaction so
    readers never observe a half-swapped state.
    """
    live = vec_table_name(collection_id, shadow=False)
    shadow = vec_table_name(collection_id, shadow=True)
    if not vec_table_exists(conn, collection_id, shadow=True):
        raise DbError(f"cannot swap: shadow table {shadow} does not exist")
    try:
        with conn:
            if vec_table_exists(conn, collection_id, shadow=False):
                conn.execute(f"DROP TABLE {live}")
            conn.execute(f"ALTER TABLE {shadow} RENAME TO {live}")
    except sqlite3.Error as exc:
        raise DbError(f"failed to swap vec tables for collection {collection_id}: {exc}") from exc


# ---------------------------------------------------------------------------
# Small CRUD helpers other modules will need immediately
# ---------------------------------------------------------------------------


def now() -> float:
    """Unix timestamp helper — single spot so every writer uses the same
    clock source (makes it trivial to monkeypatch in tests)."""
    return time.time()


def create_collection(conn: sqlite3.Connection, name: str, *, owner_user_id: Optional[str] = None,
                       visibility: str = "private", embedder_provider: Optional[str] = None,
                       embedder_model: Optional[str] = None, embedder_dims: Optional[int] = None,
                       chunk_target_tokens: int = 900, chunk_overlap_pct: float = 0.15) -> int:
    """Insert a new collection row. Caller MUST have already validated
    ``name`` via ``security.valid_collection_name`` — this function does not
    re-validate the string shape, only enforces the DB-level UNIQUE
    constraint (raises DbError on duplicate name).
    """
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO collections
                    (name, owner_user_id, visibility, embedder_provider, embedder_model,
                     embedder_dims, chunk_target_tokens, chunk_overlap_pct, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, owner_user_id, visibility, embedder_provider, embedder_model,
                 embedder_dims, chunk_target_tokens, chunk_overlap_pct, now()),
            )
            return int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise DbError(f"collection {name!r} already exists: {exc}") from exc
    except sqlite3.Error as exc:
        raise DbError(f"failed to create collection {name!r}: {exc}") from exc


def get_collection_by_name(conn: sqlite3.Connection, name: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
    return dict(row) if row is not None else None


def get_collection_by_id(conn: sqlite3.Connection, collection_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM collections WHERE id = ?", (collection_id,)).fetchone()
    return dict(row) if row is not None else None


def list_collections(conn: sqlite3.Connection, *, owner_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if owner_user_id is not None:
        rows = conn.execute(
            "SELECT * FROM collections WHERE owner_user_id = ? ORDER BY name", (owner_user_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM collections ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def delete_collection(conn: sqlite3.Connection, collection_id: int) -> None:
    """Delete a collection and everything under it (documents/chunks/FTS
    rows/jobs/spend rows referencing it), plus its vec tables if present.
    Runs as one transaction — either the whole collection is gone or none
    of it is.
    """
    try:
        with conn:
            conn.execute("DELETE FROM chunks_fts WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM chunk_enrichment WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM chunks WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM ingestion_jobs WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM spend WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM collection_shares WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM quarantine WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            for shadow in (False, True):
                if vec_table_exists(conn, collection_id, shadow=shadow):
                    conn.execute(f"DROP TABLE {vec_table_name(collection_id, shadow=shadow)}")
    except sqlite3.Error as exc:
        raise DbError(f"failed to delete collection {collection_id}: {exc}") from exc


def record_spend(conn: sqlite3.Connection, *, provider: str, op: str, units: Optional[float] = None,
                  est_usd: Optional[float] = None, collection_id: Optional[int] = None,
                  user_id: Optional[str] = None) -> int:
    """Append one row to the KB external-spend ledger — KB's own
    Cloudflare/Cohere/Apify/Groq calls are invisible to hermes' own
    LLM-cost tracking, so this ledger is the only place their $ cost is
    tracked. ``user_id`` attributes the spend to a specific guest for the
    per-guest daily-budget check — NULL for owner-initiated spend, which is
    never budget-limited."""
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO spend(ts, provider, op, units, est_usd, collection_id, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now(), provider, op, units, est_usd, collection_id, user_id),
            )
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DbError(f"failed to record spend ({provider}/{op}): {exc}") from exc


def user_spend_since(conn: sqlite3.Connection, user_id: str, *, since_ts: float) -> float:
    """Sum ``est_usd`` for *user_id* since *since_ts* — the per-guest analog
    of :func:`monthly_spend` (which sums per-PROVIDER across everyone)."""
    row = conn.execute(
        "SELECT COALESCE(SUM(est_usd), 0) AS total FROM spend WHERE user_id = ? AND ts >= ?",
        (user_id, since_ts),
    ).fetchone()
    return float(row["total"]) if row is not None else 0.0


def monthly_spend(conn: sqlite3.Connection, provider: str, *, since_ts: Optional[float] = None) -> float:
    """Sum ``est_usd`` for a provider since ``since_ts`` (defaults to 30
    days ago) — used to enforce ``memobase.monthly_ceiling_usd.<provider>``
    before starting a costly job."""
    cutoff = since_ts if since_ts is not None else now() - 30 * 24 * 3600
    row = conn.execute(
        "SELECT COALESCE(SUM(est_usd), 0) AS total FROM spend WHERE provider = ? AND ts >= ?",
        (provider, cutoff),
    ).fetchone()
    return float(row["total"]) if row is not None else 0.0


def create_ingestion_job(conn: sqlite3.Connection, *, collection_id: int, kind: str,
                          external_run_id: Optional[str] = None, stage: Optional[str] = None,
                          items_total: Optional[int] = None) -> int:
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO ingestion_jobs
                    (collection_id, kind, external_run_id, stage, items_total, items_done,
                     status, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, 'pending', ?, ?)
                """,
                (collection_id, kind, external_run_id, stage, items_total, now(), now()),
            )
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DbError(f"failed to create ingestion job: {exc}") from exc


def update_ingestion_job(conn: sqlite3.Connection, job_id: int, **fields: Any) -> None:
    """Update arbitrary columns on an ``ingestion_jobs`` row (e.g. stage=,
    items_done=, status=). Always stamps ``updated_at``. Restricted to a
    fixed column allowlist to keep this safe against accidental misuse with
    untrusted keys."""
    allowed = {"kind", "external_run_id", "stage", "items_total", "items_done", "status"}
    unknown = set(fields) - allowed
    if unknown:
        raise DbError(f"update_ingestion_job: unknown column(s) {sorted(unknown)}")
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [now(), job_id]
    try:
        with conn:
            conn.execute(
                f"UPDATE ingestion_jobs SET {set_clause}, updated_at = ? WHERE id = ?",
                values,
            )
    except sqlite3.Error as exc:
        raise DbError(f"failed to update ingestion job {job_id}: {exc}") from exc


def record_chunk_enrichment(conn: sqlite3.Connection, *, chunk_id: int, collection_id: int,
                             enrichment_text: Optional[str]) -> int:
    """Persist one chunk's JIT contextual-enrichment string (enrich.py) to
    the debug side-table, for a future ``--explain`` trace — never
    consulted by retrieve.py/answer.py, purely observability."""
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO chunk_enrichment(chunk_id, collection_id, enrichment_text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (chunk_id, collection_id, enrichment_text, now()),
            )
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DbError(f"failed to record chunk enrichment for chunk {chunk_id}: {exc}") from exc


def get_chunk_enrichment(conn: sqlite3.Connection, chunk_id: int) -> Optional[str]:
    row = conn.execute(
        "SELECT enrichment_text FROM chunk_enrichment WHERE chunk_id = ? ORDER BY id DESC LIMIT 1",
        (chunk_id,),
    ).fetchone()
    return row["enrichment_text"] if row is not None else None


def pending_ingestion_jobs(conn: sqlite3.Connection, *, collection_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return jobs not in a terminal state — used at hermes startup to
    reattach to interrupted long-running ingests."""
    terminal = ("done", "failed", "cancelled")
    placeholders = ",".join("?" for _ in terminal)
    if collection_id is not None:
        rows = conn.execute(
            f"SELECT * FROM ingestion_jobs WHERE status NOT IN ({placeholders}) AND collection_id = ? "
            f"ORDER BY started_at",
            (*terminal, collection_id),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM ingestion_jobs WHERE status NOT IN ({placeholders}) ORDER BY started_at",
            terminal,
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# MULTIUSER: collection_shares (guest ACL)
# ---------------------------------------------------------------------------


def create_share(conn: sqlite3.Connection, *, collection_id: int, user_id: str,
                  permission: str = "read", granted_by: Optional[str] = None) -> int:
    """Grant (or update) *user_id*'s permission on *collection_id*.

    Upsert on the ``UNIQUE(collection_id, user_id)`` constraint — calling
    this again for the same pair changes the permission rather than erroring
    or duplicating a row (``/memobase share <collection> @user write`` after an
    earlier ``read`` grant is a permission CHANGE, not a second grant).
    """
    if permission not in ("read", "write"):
        raise DbError(f"invalid permission {permission!r}; must be 'read' or 'write'")
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO collection_shares(collection_id, user_id, permission, granted_by, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(collection_id, user_id) DO UPDATE SET
                    permission = excluded.permission,
                    granted_by = excluded.granted_by,
                    created_at = excluded.created_at
                """,
                (collection_id, user_id, permission, granted_by, now()),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                "SELECT id FROM collection_shares WHERE collection_id = ? AND user_id = ?",
                (collection_id, user_id),
            ).fetchone()
            return int(row["id"])
    except sqlite3.Error as exc:
        raise DbError(f"failed to create share (collection={collection_id}, user={user_id!r}): {exc}") from exc


def get_share(conn: sqlite3.Connection, collection_id: int, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM collection_shares WHERE collection_id = ? AND user_id = ?",
        (collection_id, user_id),
    ).fetchone()
    return dict(row) if row is not None else None


def revoke_share(conn: sqlite3.Connection, collection_id: int, user_id: str) -> bool:
    """Delete a share row. Returns True iff a row was actually removed —
    the collection itself is untouched either way (§1.4: "коллекция
    остаётся, доступ гаснет")."""
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM collection_shares WHERE collection_id = ? AND user_id = ?",
                (collection_id, user_id),
            )
            return cur.rowcount > 0
    except sqlite3.Error as exc:
        raise DbError(f"failed to revoke share (collection={collection_id}, user={user_id!r}): {exc}") from exc


def list_shares_for_collection(conn: sqlite3.Connection, collection_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM collection_shares WHERE collection_id = ? ORDER BY user_id",
        (collection_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_shares_for_user(conn: sqlite3.Connection, user_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM collection_shares WHERE user_id = ? ORDER BY collection_id",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_permission(conn: sqlite3.Connection, collection_row: Dict[str, Any], user_id: str) -> Optional[str]:
    """Return the *effective* permission a (non-privileged) *user_id* has on
    *collection_row*: ``"owner"`` (they created it via ``memobase_create_for``),
    ``"write"``/``"read"`` (an explicit share), or ``None`` (no access at
    all — default-deny). Callers must handle the *privileged-operator* case
    (unbound session / real owner identity) themselves BEFORE calling this —
    this function only ever answers "what does this specific non-privileged
    user_id have", never "is this the operator"."""
    if collection_row.get("owner_user_id") and str(collection_row["owner_user_id"]) == str(user_id):
        return "owner"
    share = get_share(conn, collection_row["id"], user_id)
    if share is None:
        return None
    return share["permission"]


# ---------------------------------------------------------------------------
# MULTIUSER: guest quotas + daily usage
# ---------------------------------------------------------------------------


def set_guest_quota(conn: sqlite3.Connection, user_id: str, **fields: Any) -> int:
    """Upsert a per-guest quota override row. Allowed fields: ``max_mb``,
    ``max_chunks``, ``daily_upload_mb``, ``daily_budget_usd``, ``daily_calls``
    — any omitted field falls back to ``memobase.guest_defaults.*`` at read time
    (see ``security.effective_guest_quota``), NOT to a stale previous value,
    so this is a full-row upsert (COALESCE against the existing row), not a
    partial PATCH."""
    allowed = {"max_mb", "max_chunks", "daily_upload_mb", "daily_budget_usd", "daily_calls"}
    unknown = set(fields) - allowed
    if unknown:
        raise DbError(f"set_guest_quota: unknown field(s) {sorted(unknown)}")
    cols = ["max_mb", "max_chunks", "daily_upload_mb", "daily_budget_usd", "daily_calls"]
    values = [fields.get(c) for c in cols]
    try:
        with conn:
            cur = conn.execute(
                f"""
                INSERT INTO guest_quotas(user_id, {', '.join(cols)}, created_at, updated_at)
                VALUES (?, {', '.join('?' for _ in cols)}, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    {', '.join(f'{c} = excluded.{c}' for c in cols)},
                    updated_at = excluded.updated_at
                """,
                (user_id, *values, now(), now()),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute("SELECT id FROM guest_quotas WHERE user_id = ?", (user_id,)).fetchone()
            return int(row["id"])
    except sqlite3.Error as exc:
        raise DbError(f"failed to set guest quota for {user_id!r}: {exc}") from exc


def get_guest_quota(conn: sqlite3.Connection, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM guest_quotas WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row is not None else None


def today_str() -> str:
    """UTC calendar date (``YYYY-MM-DD``) — single spot so every guest-usage
    read/write agrees on what "today" means (UTC, not local time; matters on
    a VPS with a non-UTC-adjacent owner timezone)."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def get_guest_usage_today(conn: sqlite3.Connection, user_id: str) -> Dict[str, Any]:
    """Return today's usage counters for *user_id*, zeroed if no row exists
    yet (never None — callers compare directly against quota numbers)."""
    row = conn.execute(
        "SELECT * FROM guest_usage_daily WHERE user_id = ? AND usage_date = ?",
        (user_id, today_str()),
    ).fetchone()
    if row is not None:
        return dict(row)
    return {
        "user_id": user_id, "usage_date": today_str(), "bytes_uploaded": 0,
        "calls": 0, "usd_spent": 0.0, "stt_seconds": 0.0,
    }


def record_guest_usage(conn: sqlite3.Connection, user_id: str, *, bytes_uploaded: int = 0,
                        calls: int = 0, usd_spent: float = 0.0, stt_seconds: float = 0.0) -> None:
    """Add today's increment to *user_id*'s running daily usage counters
    (upsert-and-ADD, not overwrite — multiple calls in the same day
    accumulate)."""
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO guest_usage_daily(user_id, usage_date, bytes_uploaded, calls, usd_spent, stt_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, usage_date) DO UPDATE SET
                    bytes_uploaded = bytes_uploaded + excluded.bytes_uploaded,
                    calls = calls + excluded.calls,
                    usd_spent = usd_spent + excluded.usd_spent,
                    stt_seconds = stt_seconds + excluded.stt_seconds
                """,
                (user_id, today_str(), bytes_uploaded, calls, usd_spent, stt_seconds),
            )
    except sqlite3.Error as exc:
        raise DbError(f"failed to record guest usage for {user_id!r}: {exc}") from exc


def collection_size_stats(conn: sqlite3.Connection, collection_id: int) -> Dict[str, Any]:
    """Return ``{"chunks": int, "approx_mb": float}`` for *collection_id*'s
    LIVE (non-tombstoned) chunks — the storage-quota gate compares against
    this before a guest ingest, so tombstoned rows (already logically gone)
    must not count against their quota."""
    row = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(text)), 0) AS bytes_sum "
        "FROM chunks WHERE collection_id = ? AND tombstoned_at IS NULL",
        (collection_id,),
    ).fetchone()
    return {"chunks": int(row["n"]), "approx_mb": float(row["bytes_sum"]) / (1024 * 1024)}


# ---------------------------------------------------------------------------
# MULTIUSER: guest-upload quarantine (owner review queue)
# ---------------------------------------------------------------------------


def quarantine_insert(conn: sqlite3.Connection, *, collection_id: int, uploader_user_id: Optional[str],
                       source_uri: Optional[str], chunk_index: Optional[int], text: str,
                       findings: Any) -> int:
    import json as _json

    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO quarantine(collection_id, uploader_user_id, source_uri, chunk_index,
                                        text, findings_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (collection_id, uploader_user_id, source_uri, chunk_index, text, _json.dumps(findings), now()),
            )
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DbError(f"failed to quarantine chunk for collection {collection_id}: {exc}") from exc


def quarantine_list(conn: sqlite3.Connection, *, collection_id: Optional[int] = None,
                     status: Optional[str] = "pending") -> List[Dict[str, Any]]:
    clauses, params = [], []
    if collection_id is not None:
        clauses.append("collection_id = ?")
        params.append(collection_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM quarantine {where} ORDER BY created_at", params).fetchall()
    return [dict(r) for r in rows]


def quarantine_review(conn: sqlite3.Connection, quarantine_id: int, *, status: str,
                       reviewed_by: Optional[str] = None) -> bool:
    if status not in ("approved", "rejected"):
        raise DbError(f"invalid quarantine review status {status!r}; must be 'approved' or 'rejected'")
    try:
        with conn:
            cur = conn.execute(
                "UPDATE quarantine SET status = ?, reviewed_at = ?, reviewed_by = ? WHERE id = ? AND status = 'pending'",
                (status, now(), reviewed_by, quarantine_id),
            )
            return cur.rowcount > 0
    except sqlite3.Error as exc:
        raise DbError(f"failed to review quarantine item {quarantine_id}: {exc}") from exc
