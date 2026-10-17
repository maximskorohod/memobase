"""Configuration access for memobase.

Thin wrapper around ``hermes_cli.config`` — there is no ``ctx.config`` in the
plugin API; every plugin that needs config imports ``hermes_cli.config``
directly. This module owns:

  * the ``memobase.*`` default tree (DESIGN_v1.md "Config defaults" section),
  * merging user overrides from ``config.yaml`` on top of those defaults,
  * writing back either a whole-section patch or a single dotted key, and
  * deriving the *effective* per-collection config (global memobase.* defaults
    overridden by that collection's own DB columns — embedder/chunk/rerank
    profile can differ per collection, see DESIGN_v1.md DB schema).

Heavy/host imports (``hermes_cli.config``) are kept inside functions per the
project convention so importing this module
never has a load-time cost or a hard dependency on hermes internals — useful
for unit-testing config.py in isolation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# memobase.* defaults (verbatim from DESIGN_v1.md "Config defaults (config.yaml memobase.*)")
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "embedder": {
        "provider": "cloudflare",
        "model": "@cf/baai/bge-m3",
        "dims": 1024,
    },
    "rerank": {
        "provider": "cohere",
        "model": "rerank-v3.5",
        "enabled": True,
    },
    # Empty string = host's active model (ctx.llm.complete() with no model
    # override); user may point this at a cheap dedicated model instead.
    "answer_model": "",
    "confirm_over_chunks": 500,
    "monthly_ceiling_usd": {
        "cloudflare": 5,
        "cohere": 5,
    },
    "default_collection": "default",
    # Not in the spec's literal config.yaml sample but required by the
    # module interfaces that consume it (chunk.py's target_tokens/overlap_pct
    # params, embedding_signature()) — mirrors the collections table's own
    # chunk_target_tokens/chunk_overlap_pct column defaults (900 / 0.15) so
    # a brand-new collection with no per-collection override still has a
    # global fallback to read before the first DB row exists.
    "chunk": {
        "target_tokens": 900,
        "overlap_pct": 0.15,
    },
    # Reserved for the v1.x Obsidian auto-detect flow (DESIGN's "1.6c");
    # declared now so config.yaml round-trips predictably even if a user
    # sets these before the feature ships. auto_detect on, auto_connect off
    # matches "never silently ingest, always ask" from the design doc.
    "obsidian": {
        "auto_detect": True,
        "auto_connect": False,
    },
    # YouTube ingestion ladder: auto-failover order for transcript providers,
    # and the separate confirm gate for a whole-channel ingest (distinct
    # from confirm_over_chunks — a channel's cost driver is video COUNT
    # before any chunk exists yet).
    "youtube": {
        "transcript_providers": ["scrapecreators", "apify"],
        "confirm_over_videos": 20,
    },
    # STT preset (stt.py): "groq" (default, whisper-large-v3-turbo) or
    # "gemini" (gemini-2.5-flash-lite, used automatically as a fallback on
    # Groq failure regardless of this setting — this only picks which one
    # is tried FIRST).
    "stt": {
        "preset": "groq",
    },
    # JIT contextual enrichment (enrich.py) — off by default: costs one
    # cheap LLM call per new chunk at ingest time. `model` empty string =
    # host's active model via ctx.llm.complete(); point it at a dedicated
    # cheap auxiliary model instead once enabled.
    "enrich": {
        "enabled": False,
        "model": "",
    },
    # Gap #17 ("дивергенция дом-ПК / VPS"): the
    # canonical host is where kb writes are allowed; other hermes instances
    # sharing the same collections are read/helper-only. Defaults to True
    # (single-instance is the common case) — a multi-host setup must
    # explicitly flip this to False on the non-canonical instance.
    "canonical_host": True,
    # --- MULTIUSER (§1.4 "Гостевые библиотекари" / §1.9 gaps #8, #13) -------
    # The bot owner's platform identity (``event.source.user_id`` from
    # gateway — Telegram numeric id, etc). Empty string = "not claimed yet":
    # every session with a resolved identity is then treated as a guest by
    # `security.is_privileged`, EXCEPT a session with NO resolved identity
    # at all (plain CLI, or a gateway wired without identity hooks), which
    # stays privileged — this is the same "zero-config, nothing to isolate
    # from yet" posture v1's session-binding already uses (tools.py module
    # docstring). `/memobase setup`'s wizard (wizard.py) claims this automatically
    # for whoever first runs it, unless it is already set.
    "owner_user_id": "",
    # Fallback quota for a guest with NO row in `guest_quotas` (per-guest
    # DB overrides via `memobase_share`'s owner-facing quota commands win over
    # these). All four are enforced BEFORE the costly ingest call, not just
    # before the chunk write (§1.9 gap #8).
    "guest_defaults": {
        "max_mb": 200,
        "max_chunks": 4000,
        "daily_upload_mb": 50,
        "daily_budget_usd": 0.50,
        "daily_calls": 200,
    },
    # memobase_query/memobase_ask rate-limit for a GUEST identity only (never applied to
    # the privileged operator) — burns rerank/generation money per call, so
    # gap #8 calls this out separately from the storage/upload quotas above.
    "guest_rate_limit": {
        "calls_per_minute": 6,
    },
    # Extra directories a GUEST may ingest local files from, on top of the
    # gateway's own attachment caches (<HERMES_HOME>/cache/{documents,audio,
    # videos}). Paths are resolved (symlinks included) before the check.
    "allowed_ingest_roots": [],
    # Nightly doctor (backup.py, §1.9 gap #9/#19): VACUUM INTO snapshot +
    # rotation + disk-usage alert threshold. `off_vps_command` is an OPTIONAL
    # shell command template (e.g. an rclone invocation) run after each local
    # snapshot; `{snapshot_path}` is substituted. Empty = local-only backups.
    "backup": {
        "keep": 7,
        "disk_alert_pct": 80,
        "off_vps_command": "",
    },
}


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict: ``base`` with ``override`` merged in, recursively.

    Only dict values are merged recursively; any other type in ``override``
    (including lists) replaces the base value outright. Never mutates either
    input.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _memobase_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.config import cfg_get

    user_kb = cfg_get(cfg, "memobase", default={})
    if not isinstance(user_kb, dict):
        # A user could theoretically clobber `memobase:` with a scalar by hand-
        # editing config.yaml; don't let that crash every memobase.* read site.
        return {}
    return user_kb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_memobase_config() -> Dict[str, Any]:
    """Return the effective ``memobase.*`` config: DEFAULTS deep-merged with the
    user's ``config.yaml``. Safe to mutate — this is a fresh dict every call.
    """
    from hermes_cli.config import load_config

    cfg = load_config()
    return _deep_merge(DEFAULTS, _memobase_section(cfg))


def get_memobase_config_readonly() -> Dict[str, Any]:
    """Fast-path variant of :func:`get_memobase_config` for hot, read-only call
    sites (e.g. per-query threshold lookups in retrieve.py/answer.py).

    Uses ``load_config_readonly()`` to skip hermes' own defensive deepcopy of
    the whole config (see its docstring) — but ``_deep_merge`` here still
    deepcopies DEFAULTS/override into a fresh result, so the dict returned to
    the caller is always safe to read AND safe to mutate; only the *cached*
    config object hermes holds internally is skipped from copying.
    """
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly()
    return _deep_merge(DEFAULTS, _memobase_section(cfg))


def save_memobase_config(patch: Dict[str, Any]) -> None:
    """Deep-merge ``patch`` into the persisted ``memobase.*`` config section and
    save via ``hermes_cli.config.save_config``.

    Use this for multi-key writes (e.g. a setup wizard writing several
    memobase.embedder.* keys at once). For a single dotted key, prefer
    :func:`set_memobase_value` — it routes through ``set_config_value`` which
    handles managed-scope / API-key-to-.env routing correctly.
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    existing = cfg.get("memobase", {})
    if not isinstance(existing, dict):
        existing = {}
    cfg["memobase"] = _deep_merge(existing, patch)
    save_config(cfg)


def set_memobase_value(dotted_key: str, value: Any) -> None:
    """Set a single ``memobase.<dotted_key>`` config value.

    Thin pass-through to ``hermes_cli.config.set_config_value`` (dotted-path
    writer; routes API-key-shaped values to ``.env`` automatically, honors
    managed-scope locks). Example: ``set_memobase_value("confirm_over_chunks", 200)``
    writes ``memobase.confirm_over_chunks: 200``.
    """
    from hermes_cli.config import set_config_value

    key = dotted_key if dotted_key.startswith("memobase.") else f"memobase.{dotted_key}"
    # hermes_cli.config.set_config_value parses a STRING (it calls .strip());
    # passing an int/bool crashed `hermes memobase setup` in local mode.
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif not isinstance(value, str):
        value = str(value)
    set_config_value(key, value)


def get_collection_cfg(collection_row: Dict[str, Any], *, memobase_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the effective per-collection config consumed by embed.py,
    chunk.py, retrieve.py, rerank.py and answer.py.

    ``collection_row`` is a DB row (mapping) from the ``collections`` table
    (see db.py) — its explicit, non-NULL columns override the global
    ``memobase.*`` defaults, matching DESIGN_v1.md's per-collection embedder/chunk
    profile design ("Профиль нарезки — на коллекцию/тип контента").

    Pass ``memobase_cfg`` to reuse an already-loaded config dict (e.g. from a
    request-scoped cache) and skip a redundant ``load_config_readonly()``
    call; otherwise it is loaded internally.

    Returns a plain dict — never a reference into DEFAULTS or the row.
    """
    base = memobase_cfg if memobase_cfg is not None else get_memobase_config_readonly()

    embedder = dict(base.get("embedder", {}))
    if collection_row.get("embedder_provider"):
        embedder["provider"] = collection_row["embedder_provider"]
    if collection_row.get("embedder_model"):
        embedder["model"] = collection_row["embedder_model"]
    if collection_row.get("embedder_dims"):
        embedder["dims"] = collection_row["embedder_dims"]

    chunk = dict(base.get("chunk", {}))
    if collection_row.get("chunk_target_tokens"):
        chunk["target_tokens"] = collection_row["chunk_target_tokens"]
    if collection_row.get("chunk_overlap_pct") is not None:
        chunk["overlap_pct"] = collection_row["chunk_overlap_pct"]

    rerank = dict(base.get("rerank", {}))

    return {
        "collection_id": collection_row.get("id"),
        "collection_name": collection_row.get("name"),
        "visibility": collection_row.get("visibility", "private"),
        "embedder": embedder,
        "chunk": chunk,
        "rerank": rerank,
        # Per-collection calibrated thresholds (may be None if never
        # calibrated yet via memobase_selfcheck — callers must handle
        # that).
        "rrf_threshold": collection_row.get("rrf_threshold"),
        "rerank_threshold": collection_row.get("rerank_threshold"),
        "migration_state": collection_row.get("migration_state", "idle"),
        "answer_model": base.get("answer_model", ""),
        "confirm_over_chunks": base.get("confirm_over_chunks", 500),
        "monthly_ceiling_usd": dict(base.get("monthly_ceiling_usd", {})),
    }
