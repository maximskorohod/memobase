"""Obsidian vault integration for memobase.

Read-only, always: this module NEVER writes into a vault — it only reads
``.md`` notes and Obsidian's own global vault registry
(``%APPDATA%/obsidian/obsidian.json`` on Windows; platform-equivalent paths
elsewhere) to auto-detect vaults worth offering to connect.

Module interface:

    detect_vaults(path=None) -> list[dict]
        # [{"id", "path", "name", "open", "ts", "note_count"}], never raises
    extract_note(path) -> dict
        # extract.py-shaped {text, blocks, meta, skipped} for ONE note
    ingest_vault(conn, collection_row, vault_path, *, memobase_cfg=None, confirm=False) -> dict
        # walks the vault, ingest_source()-per-note (dedup/refresh is free —
        # see module docstring below)

Design notes:

* **Auto-detection, never auto-connect** (unless the owner explicitly opts
  in via ``memobase.obsidian.auto_connect: true``, config.py's default is
  ``False``) — "never silently ingest": finding
  a vault only ever produces a list callers may *offer* to the owner
  (``/memobase connect obsidian`` flow in commands.py), never an automatic ingest.
* **Ignore list**: ``.obsidian/`` (Obsidian's own config folder),
  ``.trash/`` (Obsidian's in-vault trash), ``templates/`` (template notes,
  not real content) — matched case-insensitively against any path segment.
* **Wikilinks `[[...]]`**: parsed into a note-name -> [linked note names]
  graph, stored as chunk/document metadata for a FUTURE graph-rerank pass.
  v1 only builds and returns the graph — nothing in
  retrieve.py/answer.py consumes it yet; that is out of this task's scope.
* **YAML frontmatter** -> chunk metadata (tags/dates/type) so a future
  filter/boost pass can use it. Frontmatter parsing degrades to "no
  metadata" (never raises) if PyYAML is not installed or the block doesn't
  parse — a note's body is never dropped over a broken frontmatter block.
* **Refresh by hash is free**: ``ingest_vault`` re-walks the whole vault
  every call and re-calls ``ingest_source`` per note; ingest.py's own
  content-hash dedup (module docstring: "skip re-processing an unchanged
  source") already makes an unchanged note come back ``"unchanged"`` cheaply
  — no separate "vault watermark" bookkeeping is needed for v1's "nightly
  refresh picks up edited notes" requirement.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("memobase.obsidian")

IGNORE_DIR_NAMES = {".obsidian", ".trash", "templates"}

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n?", re.DOTALL)


class ObsidianError(RuntimeError):
    """Raised only for programmer-error-shaped misuse. Ordinary failures
    (Obsidian not installed, vault path missing, unreadable note) never
    raise — they degrade to an empty result / a ``skipped`` entry, matching
    extract.py's own "never raises" contract for the same shape of module."""


# ---------------------------------------------------------------------------
# obsidian.json auto-detection
# ---------------------------------------------------------------------------


def default_obsidian_json_path() -> Path:
    """Best-effort platform path to Obsidian's global vault registry.
    Never raises. This is a plain, undocumented-but-stable JSON file
    Obsidian itself maintains (not something memobase writes)."""
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "obsidian" / "obsidian.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"
    # Linux and anything else XDG-shaped.
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "obsidian" / "obsidian.json"


def _count_notes(vault_path: Path) -> int:
    try:
        return sum(1 for _ in iter_markdown_files(vault_path))
    except OSError:
        return 0


def parse_obsidian_json(path: Path) -> List[Dict[str, Any]]:
    """Parse Obsidian's ``obsidian.json`` vault registry.

    Schema (observed, not officially documented by Obsidian):
    ``{"vaults": {"<16-hex-id>": {"path": "...", "ts": <epoch-ms>,
    "open": bool}, ...}}``. Returns ``[]`` (never raises) if the file is
    missing, unreadable, or not JSON-shaped as expected — "Obsidian not
    installed" and "Obsidian installed but registry is empty/corrupt" both
    degrade the same way: silence, per module docstring.
    """
    import json

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning("obsidian.json at %s is not valid JSON; ignoring", path)
        return []

    vaults = data.get("vaults") if isinstance(data, dict) else None
    if not isinstance(vaults, dict):
        return []

    out: List[Dict[str, Any]] = []
    for vault_id, info in vaults.items():
        if not isinstance(info, dict):
            continue
        vpath_str = info.get("path")
        if not vpath_str:
            continue
        vpath = Path(vpath_str)
        out.append(
            {
                "id": vault_id,
                "path": str(vpath),
                "name": vpath.name or str(vpath),
                "open": bool(info.get("open", False)),
                "ts": info.get("ts"),
                "exists": vpath.is_dir(),
                "note_count": _count_notes(vpath) if vpath.is_dir() else 0,
            }
        )
    return out


def detect_vaults(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return every vault Obsidian knows about, per :func:`parse_obsidian_json`.

    Never raises. ``path`` defaults to :func:`default_obsidian_json_path`;
    pass it explicitly in tests. An empty return means either "Obsidian not
    installed" or "no vaults registered" — callers should not distinguish
    the two (both mean "nothing to offer the owner").
    """
    try:
        target = path if path is not None else default_obsidian_json_path()
        return parse_obsidian_json(target)
    except Exception:  # noqa: BLE001 - detection must never crash a caller (startup/nightly doctor)
        logger.debug("detect_vaults failed unexpectedly", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Vault walking
# ---------------------------------------------------------------------------


def _is_ignored(rel_parts: Tuple[str, ...]) -> bool:
    return any(part.lower() in IGNORE_DIR_NAMES for part in rel_parts)


def iter_markdown_files(vault_path: Path):
    """Yield every ``.md`` file under *vault_path*, skipping
    ``.obsidian/``/``.trash/``/``templates/`` at any depth (case-insensitive)."""
    vault_path = Path(vault_path)
    for root, dirnames, filenames in os.walk(vault_path):
        root_path = Path(root)
        rel = root_path.relative_to(vault_path).parts
        if _is_ignored(rel):
            dirnames[:] = []  # do not descend further
            continue
        # Prune ignored subdirectories before os.walk recurses into them.
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_DIR_NAMES]
        for fname in filenames:
            if fname.lower().endswith(".md"):
                yield root_path / fname


# ---------------------------------------------------------------------------
# Frontmatter + wikilinks
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split a leading YAML frontmatter block (between ``---`` markers) off
    *text*. Returns ``(metadata_dict, body_without_frontmatter)``.

    Degrades to ``({}, text)`` (never raises) if there is no frontmatter
    block, PyYAML is not installed, or the block fails to parse as a dict —
    a broken frontmatter block must never lose the note's body.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_yaml = m.group(1)
    body = text[m.end():]
    try:
        import yaml  # optional dependency (already used by hermes-core's own config.yaml)

        meta = yaml.safe_load(raw_yaml)
    except Exception:  # noqa: BLE001 - malformed/missing-PyYAML frontmatter is not fatal
        logger.debug("frontmatter parse failed; treating as no metadata", exc_info=True)
        return {}, body
    if not isinstance(meta, dict):
        return {}, body
    return meta, body


def extract_wikilinks(text: str) -> List[str]:
    """Return the list of note names targeted by ``[[wikilink]]`` /
    ``[[wikilink#heading]]`` / ``[[wikilink|alias]]`` references in *text*,
    in first-seen order, de-duplicated. The ``.md`` extension (if a link
    spells it out) is stripped so link targets match note names as returned
    by :func:`iter_markdown_files`'s stems."""
    seen: List[str] = []
    seen_set = set()
    for m in _WIKILINK_RE.finditer(text or ""):
        target = m.group(1).strip()
        if target.lower().endswith(".md"):
            target = target[:-3]
        if target and target not in seen_set:
            seen_set.add(target)
            seen.append(target)
    return seen


def build_link_graph(vault_path: Path) -> Dict[str, List[str]]:
    """Return ``{note_name: [linked_note_name, ...]}`` for every note in
    *vault_path*. Best-effort: a note that fails to read is simply omitted
    (never raises) — reserved for a future graph-rerank pass, see module
    docstring."""
    graph: Dict[str, List[str]] = {}
    for path in iter_markdown_files(vault_path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        graph[path.stem] = extract_wikilinks(text)
    return graph


# ---------------------------------------------------------------------------
# Single-note extraction (extract.py-shaped)
# ---------------------------------------------------------------------------


def extract_note(path: str) -> Dict[str, Any]:
    """Extract ONE Obsidian note into the same ``{text, blocks, meta,
    skipped}`` shape as extract.py's ``extract()`` — this is what
    ``ingest.py`` calls for ``source_type == "obsidian"`` with a single
    ``.md`` file path (see ``ingest_vault`` below for the whole-vault path,
    which calls this indirectly via ``ingest.ingest_source`` per note).

    Never raises: read/parse failures become a ``skipped`` entry with
    empty text, matching extract.py's contract exactly.
    """
    from . import extract as extract_mod  # local import: avoid a hard load-order dependency

    note_path = Path(path)
    try:
        raw = note_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"text": "", "blocks": [], "meta": {"title": None, "pages": None}, "skipped": [{"reason": f"failed to read {path}: {exc}"}]}

    frontmatter, body = parse_frontmatter(raw)
    wikilinks = extract_wikilinks(body)

    # Reuse extract.py's own heading/code-fence block splitter — internal
    # (underscore-prefixed) but same package, and this is exactly the
    # markdown-structure splitting MD/TXT/DOCX/HTML all already share.
    blocks = extract_mod._text_to_blocks(body, page=None)  # noqa: SLF001

    title = frontmatter.get("title") if isinstance(frontmatter.get("title"), str) else None
    meta: Dict[str, Any] = {
        "title": title or note_path.stem,
        "pages": None,
        "frontmatter": frontmatter,
        "wikilinks": wikilinks,
        "note_name": note_path.stem,
    }
    return {"text": body, "blocks": blocks, "meta": meta, "skipped": []}


# ---------------------------------------------------------------------------
# Whole-vault ingestion (multi-document orchestrator)
# ---------------------------------------------------------------------------


def ingest_vault(
    conn,
    collection_row: Dict[str, Any],
    vault_path: str,
    *,
    memobase_cfg: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Ingest every note under *vault_path* into *collection_row*, one
    ``documents`` row per note (source_uri = note's absolute path,
    source_type="obsidian"). Re-running this (nightly refresh) is cheap for
    unchanged notes — see module docstring.

    Returns ``{"status": "done", "notes_total", "notes_ingested",
    "notes_unchanged", "notes_failed", "per_note": [...]}``. Never raises
    for a missing/empty vault (reports ``notes_total=0``); only
    :class:`ObsidianError` for programmer misuse (missing conn/collection).
    """
    from . import config as kb_config
    from . import ingest as ingest_mod

    if conn is None or not isinstance(collection_row, dict) or "id" not in collection_row:
        raise ObsidianError("ingest_vault requires a real conn and a valid collection_row")

    memobase_cfg = memobase_cfg if memobase_cfg is not None else kb_config.get_memobase_config_readonly()
    vault = Path(vault_path)
    if not vault.is_dir():
        return {
            "status": "failed",
            "error": f"vault path not found or not a directory: {vault_path}",
            "notes_total": 0,
        }

    note_paths = sorted(iter_markdown_files(vault))
    per_note: List[Dict[str, Any]] = []
    counts = {"ingested": 0, "unchanged": 0, "failed": 0}

    for note_path in note_paths:
        result = ingest_mod.ingest_source(
            conn, collection_row, str(note_path), "obsidian", memobase_cfg=memobase_cfg, confirm=confirm
        )
        status = result.get("status")
        if status == "done":
            counts["ingested"] += 1
        elif status == "unchanged":
            counts["unchanged"] += 1
        else:
            counts["failed"] += 1
        per_note.append({"path": str(note_path), "status": status})

    return {
        "status": "done",
        "notes_total": len(note_paths),
        "notes_ingested": counts["ingested"],
        "notes_unchanged": counts["unchanged"],
        "notes_failed": counts["failed"],
        "per_note": per_note,
    }
