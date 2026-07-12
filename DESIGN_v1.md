# MemoBase v1 (core) — Build Spec

Grounded local knowledge base for hermes-agent. v1 = "NotebookLM core": ingest local files → hybrid search → answer with verified citations OR honest refusal. YouTube/STT/guests/Telegram-wizard/Obsidian are v1.x (OUT of v1).

Verify every hermes import/signature against your local hermes-agent checkout (this build targeted v0.18.0). LOCAL CODE WINS over any assumption made below.

Env keys already in `~/.hermes/.env` (all live-verified): `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `COHERE_API_KEY` (+ APIFY_TOKEN, SCRAPECREATORS_API_KEY, GROQ_API_KEY, GEMINI_API_KEY — for v1.x, ignore in v1).

## Non-negotiables
- User-facing strings RU, simple language. Code/comments EN.
- All state under `get_hermes_home()/memobase/` via `from hermes_constants import get_hermes_home`.
- All external HTTP sends a browser-like `User-Agent` (Groq/Cloudflare sit behind Cloudflare WAF — bare UA → 403/1010; VERIFIED).
- Hooks (if any) plain `def`, fast, try/except, never raise. Heavy/host imports inside functions.
- Never modify core files. Never run `hermes` CLI (user does that).
- Every external call: timeout, retry w/ exp backoff on 429/5xx, graceful degradation, and log to the KB spend-ledger.

## Files → plugin root (installs to `<HERMES_HOME>/plugins/memobase/`)
```
plugin.yaml            # name: memobase, kind: standalone, provides_tools/commands, author: Maxim Vasko
install.ps1 / install.sh  # pip install deps INTO hermes venv (general plugins have NO lazy-install)
__init__.py            # register(ctx): tools + slash commands + cli; thin glue only
config.py              # load/save via hermes_cli.config; memobase.* defaults; per-collection chunk/embedder profiles
db.py                  # sqlite schema, connections (WAL, busy_timeout=5000, synchronous=NORMAL), migrations
security.py            # SSRF guard, secret scanner (blocking), collection-name allowlist, injection scanner/fence
extract.py             # pdf(pdfplumber→pypdf), docx(mammoth), html(trafilatura), md/txt/csv(stdlib) → Doc{text,structure,meta}
normalize.py           # 11-step pipeline (ftfy→NFC→ctrl→entities→codetags→pdf-boilerplate→hyphen→ws→quotes→dedup→lang)
chunk.py               # structural chunker (qmd-style scoring, code-fence-safe, per-collection target/overlap)
embed.py               # Cloudflare BGE-M3 + pluggable OpenAI-compat; signature; dim/NaN validation; shadow-table migration
retrieve.py            # FTS5(+RU stemming) BM25 + vec KNN → RRF(k=60) → positional blend with rerank
rerank.py              # Cohere rerank-v3.5 + RRF-only fallback path (separate calibrated threshold)
answer.py              # sufficiency gate(2 thresholds) → subclaim coverage → forced citations → quote verify → gaps → refusal
ingest.py              # orchestrates extract→normalize→chunk→embed→index; re-ingest purge (hash-set diff); resumable jobs
ledger.py              # KB external-spend ledger (Cloudflare/Cohere calls, $ estimate); monthly ceiling check
tools.py               # memobase_ingest/memobase_query/memobase_ask/memobase_list/memobase_delete/memobase_status handlers; collection binding via session_id
commands.py            # /memobase slash (ask/status/list) — works in CLI + gateway
selfcheck.py           # memobase_selfcheck: control questions, coverage report
cli.py                 # hermes memobase ingest|list|reindex|status
README.md / README.en.md / GUIDE.md / LICENSE
tests/                 # pytest; live venv; monkeypatch HERMES_HOME to tmp
```

## DB schema (single `~/.hermes/memobase/memobase.db`, WAL) — one file, collection as column (safer than per-dir: no path traversal on the DB path itself)
```sql
collections(id INTEGER PK, name TEXT UNIQUE, owner_user_id TEXT, visibility TEXT DEFAULT 'private',
  embedder_provider TEXT, embedder_model TEXT, embedder_dims INTEGER,
  chunk_target_tokens INTEGER DEFAULT 900, chunk_overlap_pct REAL DEFAULT 0.15,
  rrf_threshold REAL, rerank_threshold REAL, migration_state TEXT DEFAULT 'idle', created_at REAL)
documents(id INTEGER PK, collection_id INTEGER, source_uri TEXT, source_type TEXT, content_sha256 TEXT,
  title TEXT, page_count INTEGER, ingested_at REAL, superseded_at REAL, UNIQUE(collection_id, source_uri))
chunks(id INTEGER PK, collection_id INTEGER, document_id INTEGER, seq INTEGER,
  text TEXT, content_sha256 TEXT, page_or_timecode TEXT, section TEXT, lang TEXT,
  embed_signature TEXT, tombstoned_at REAL, created_at REAL)
-- FTS5 external-content over chunks.text, porter+unicode61; RU stemming applied to a shadow 'text_stem' column
chunks_fts USING fts5(text, text_stem, chunk_id UNINDEXED, collection_id UNINDEXED, tokenize='unicode61')
-- one vec0 table PER collection dims: vec_c{collection_id} (created lazily with that collection's dims)
ingestion_jobs(id INTEGER PK, collection_id INTEGER, kind TEXT, external_run_id TEXT, stage TEXT,
  items_total INTEGER, items_done INTEGER, status TEXT, started_at REAL, updated_at REAL)
spend(id INTEGER PK, ts REAL, provider TEXT, op TEXT, units REAL, est_usd REAL, collection_id INTEGER)
_meta(key TEXT PK, value TEXT)
```

## Module interfaces (parallel writers MUST match these)
```python
# extract.py
def extract(path_or_url: str, source_type: str) -> dict  # {text, blocks:[{text,page,section,is_code}], meta:{title,pages,...}, skipped:[{reason}]}
# normalize.py
def normalize(doc: dict, profile: str="default") -> dict  # same shape, text cleaned; report counters in doc['norm_report']
# chunk.py
def chunk(doc: dict, target_tokens: int, overlap_pct: float) -> list[dict]  # [{text,seq,page_or_timecode,section,is_code}]
# embed.py
def embed_texts(texts: list[str], collection_cfg: dict) -> list[list[float]]  # validates dims/finite; raises EmbedError
def embedding_signature(cfg: dict) -> str  # "provider|model|dims|chunkT|overlap"
# rerank.py
def rerank(query: str, candidates: list[dict], cfg: dict) -> tuple[list[dict], str]  # (ranked, mode) mode in {'cohere','rrf-only'}
# retrieve.py
def hybrid_search(collection_id: int, query: str, k: int, cfg: dict) -> list[dict]  # fused candidates w/ scores + source
# answer.py
def answer(collection_id: int, query: str, cfg: dict) -> dict  # {answer, citations:[{chunk_id,page_or_timecode,quote}], gaps:[], mode, refused:bool}
# security.py
def check_url(url: str) -> None  # raises SsrfError on private/loopback/link-local/metadata/non-http(s)
def scan_secrets(text: str) -> list[dict]  # findings; caller quarantines
def valid_collection_name(name: str) -> bool  # [a-zA-Z0-9_-]{1,64}
def fence_untrusted(text: str) -> str  # wrap chunk for parent-facing memobase_query
```

## Blockers from §1.9 — MUST be in v1 (map)
1. SSRF: `security.check_url` called in extract/ingest before any URL fetch AND in the size-estimate pre-fetch.
2. memobase_query fencing: tools.py wraps every returned chunk via `fence_untrusted` + runs `scan_secrets`/injection patterns at retrieval.
3. RRF-only threshold: answer.py uses collection.rrf_threshold when rerank mode=='rrf-only'; surfaces "degraded mode" in result.
4. Shadow-table migration: embed.py re-embed writes to vec_c{id}_v2, atomic rename; collections.migration_state guards; memobase_ask serves FTS-only or blocks with status during migration.
5. Entailment/subclaim: answer.py decomposes query into subclaims (heuristic split on 'и'/'?'/','), each must map to a citation or a gaps entry; non-quoted numeric/negation clauses matched against cited chunk or downgraded.
6. Re-ingest purge: ingest.py diffs chunk-hash set per (collection,source_uri); tombstones hashes absent in new pass; retrieve excludes tombstoned.
Also v1: secret scan BLOCKING pre-embed (ingest.py quarantines); collection-name allowlist (tools/create); RU stemming in retrieve/index (snowball via nltk? — prefer a light pure-py Russian stemmer: `Stemmer`/PyStemmer(BSD) or a vendored snowball-ru ~200 lines; DECIDE at build, prefer PyStemmer if wheel installs on win/linux, else vendored); KB spend-ledger + monthly ceiling; SQLite busy_timeout/synchronous; dim/NaN validation; ingestion_jobs resume.

## Retrieval detail
- FTS leg: query and index both RU-stemmed into text_stem; qmd query-hardening for hyphen/dotted tokens; BM25 rank.
- Vector leg: embed query (browser UA), vec0 KNN over vec_c{id}, over-fetch k*3.
- Fuse: RRF k=60, candidateK=limit*3, top-rank bonus (+0.05 #1 / +0.02 #2-3).
- Blend with rerank: positional 75/25 (rank1-3), 60/40 (4-10), 40/60 (11+). If rerank mode=='rrf-only', skip blend, use RRF order + rrf_threshold gate.

## answer() flow
1. hybrid_search → top candidates. 2. rerank → (ranked, mode). 3. gate: best score < threshold(mode) → refused=True, return honest "в базе этого нет" (+ near-miss soft mode if within band). 4. subclaim decompose. 5. `ctx.llm.complete()` (TOOL-LESS — assert no tools; this is the isolated answerer) with fenced chunks + "answer only from context, cite [chunk:N] with verbatim quote per claim". 6. parse structured citations (JSON field + regex fallback). 7. quote-verify: fuzzy-match each quote vs cited chunk raw text; subclaim coverage check; unverifiable clauses → gaps. 8. render RU answer + citations (inline quote + source+page/section) + gaps list.

## Config defaults (config.yaml memobase.*)
```yaml
kb:
  embedder: {provider: cloudflare, model: "@cf/baai/bge-m3", dims: 1024}
  rerank: {provider: cohere, model: rerank-v3.5, enabled: true}
  answer_model: ""          # empty = host active model; user may set a cheap one
  confirm_over_chunks: 500
  monthly_ceiling_usd: {cloudflare: 5, cohere: 5}
  default_collection: default
```

## install script
- Detect hermes venv python (`where hermes` → ..\venv), pip install: `sqlite-vec pdfplumber pypdf mammoth trafilatura>=1.8 ftfy py3langid requests` (+ RU stemmer chosen). All MIT/BSD/Apache. Print success + "restart hermes, run /memobase status".
- No torch, no pymupdf (AGPL), no docling.

## Tests
- Isolation pattern from `tests/plugins/test_disk_cleanup_plugin.py`: monkeypatch HERMES_HOME→tmp; load via hermes_plugins namespace.
- Unit: extract each format (tiny fixtures incl. RU pdf/docx/csv/tg-json), normalize steps, chunker code-fence safety, RU stemming match ("договора"→finds "договор"), RRF fusion, quote-verify catches a fabricated quote, subclaim coverage, SSRF guard rejects 169.254.169.254/localhost/file://, secret scan blocks, collection-name allowlist rejects "../x", re-ingest purge tombstones removed chunks.
- Integration (marked, needs keys from .env — read but don't hardcode): one real Cloudflare embed + one Cohere rerank + full memobase_ingest(tmp pdf)→memobase_ask happy path + refusal path. Guard so CI without keys skips these.
- Run: `<venv python> -m pytest ... -v` with pytest.ini import-mode=importlib.

## Out of v1 (v1.x): YouTube ladder, STT, Obsidian auto-detect, guest collections, Telegram onboarding wizard, contextual enrichment, backup cron, local embedding tier, mind-map, near-miss image snapshots. Design them to slot in (interfaces reserved) but DO NOT build in v1.
