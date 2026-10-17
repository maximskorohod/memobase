"""Nightly "doctor" backup for memobase.

Public surface, all pure-enough-to-unit-test (file I/O only, no network):

    run_doctor(memobase_cfg=None, *, db_path=None, backup_dir=None) -> dict
        The nightly job entry point (runnable via `hermes cron`, see
        cli.py's `hermes memobase backup-run`): VACUUM INTO snapshot + rotation +
        disk-usage check, all best-effort/never-raise at the top level (a
        backup failure must not crash the cron job that also does other
        doctor work) — but returns a result dict with per-step
        success/error so the caller can alert on it.

    vacuum_into_snapshot(db_path, dest_path) -> None
        The actual consistent-snapshot primitive: SQLite's ``VACUUM INTO``
        (NOT a raw file copy — copying a live WAL-mode db file can capture
        a torn/inconsistent state; VACUUM INTO opens its own read
        transaction and writes out a fully consistent single file).

    rotate_backups(backup_dir, keep) -> list[Path]
        Deletes old snapshots beyond `keep`, oldest first, by filename
        timestamp (not mtime, so a restore/copy that changes mtime doesn't
        confuse rotation).

    check_disk_usage(path, alert_pct) -> dict
        Disk-usage percentage for the filesystem containing *path* --
        flags when it is over `alert_pct` (default 80, per gap #19).

Off-VPS copy (``memobase.backup.off_vps_command``) is an OPTIONAL, user-supplied
shell command template run via ``subprocess`` after a successful local
snapshot — e.g. an rclone invocation. Empty (default) = local-only backups.
Never blocks/fails the local snapshot itself if it errors.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("memobase.backup")

_SNAPSHOT_NAME_RE_PREFIX = "memobase-"
_SNAPSHOT_SUFFIX = ".db"


class BackupError(RuntimeError):
    """Raised only by the low-level primitives (`vacuum_into_snapshot`) for
    a genuine failure — `run_doctor` catches this and reports it in the
    result dict rather than raising, per this plugin's "doctor never
    crashes the process it lives in" convention (§1.9 gap #16)."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def default_backup_dir() -> Path:
    from . import db as db_mod

    return db_mod.get_kb_dir() / "backups"


# ---------------------------------------------------------------------------
# VACUUM INTO snapshot
# ---------------------------------------------------------------------------


def vacuum_into_snapshot(db_path: Path, dest_path: Path) -> None:
    """Write a consistent point-in-time copy of the sqlite db at *db_path*
    to *dest_path* via ``VACUUM INTO``. Raises :class:`BackupError` on any
    failure (caller decides how to surface it) — this function is the one
    place in this module allowed to raise, since a caller unit-testing it
    directly needs to see the real failure, not a swallowed one."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        raise BackupError(f"snapshot destination already exists: {dest_path}")
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
    except sqlite3.Error as exc:
        raise BackupError(f"failed to open source db {db_path}: {exc}") from exc
    try:
        conn.execute("VACUUM INTO ?", (str(dest_path),))
    except sqlite3.Error as exc:
        raise BackupError(f"VACUUM INTO failed ({db_path} -> {dest_path}): {exc}") from exc
    finally:
        conn.close()


def snapshot_filename(prefix: str = "memobase") -> str:
    """``memobase-YYYYmmdd-HHMMSS.db`` — lexicographically sortable by time, so
    :func:`rotate_backups` can order purely by filename, not filesystem
    mtime (which a copy/restore/rclone sync could alter)."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}{_SNAPSHOT_SUFFIX}"


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def rotate_backups(backup_dir: Path, keep: int) -> List[Path]:
    """Delete snapshots in *backup_dir* beyond the newest *keep* (by
    filename sort, oldest first). Returns the list of paths actually
    deleted. Never raises — a delete failure for one file is logged and
    skipped, not fatal to rotation as a whole."""
    if keep < 0:
        keep = 0
    try:
        # Only OUR snapshot names (snapshot_filename): a bare "*.db" would also
        # match memobase.db itself (or anyone else's .db) if backup_dir ever
        # pointed at a shared directory.
        candidates = sorted(
            p for p in backup_dir.glob(f"memobase-*{_SNAPSHOT_SUFFIX}") if p.is_file()
        )
    except OSError:
        logger.warning("backup: failed to list %s for rotation", backup_dir, exc_info=True)
        return []
    to_delete = candidates[:-keep] if keep > 0 else candidates
    deleted: List[Path] = []
    for p in to_delete:
        try:
            p.unlink()
            deleted.append(p)
        except OSError:
            logger.warning("backup: failed to delete old snapshot %s", p, exc_info=True)
    return deleted


# ---------------------------------------------------------------------------
# Disk usage (§1.9 gap #19)
# ---------------------------------------------------------------------------


def check_disk_usage(path: Path, alert_pct: float = 80.0) -> Dict[str, Any]:
    """Return ``{"used_pct": float, "alert": bool, "total_gb": float,
    "free_gb": float}`` for the filesystem containing *path*. Never
    raises — an unavailable disk-usage read (unusual, but ``shutil.
    disk_usage`` can fail on an exotic mount) reports ``alert=False`` with
    ``used_pct=None`` rather than crashing the doctor run."""
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        logger.warning("backup: disk_usage failed for %s", path, exc_info=True)
        return {"used_pct": None, "alert": False, "total_gb": None, "free_gb": None}
    used_pct = (usage.used / usage.total) * 100.0 if usage.total else 0.0
    return {
        "used_pct": round(used_pct, 1),
        "alert": used_pct >= alert_pct,
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "free_gb": round(usage.free / (1024 ** 3), 2),
    }


# ---------------------------------------------------------------------------
# Optional off-VPS copy
# ---------------------------------------------------------------------------


def _run_off_vps_copy(command_template: str, snapshot_path: Path) -> Dict[str, Any]:
    import subprocess

    command = command_template.replace("{snapshot_path}", str(snapshot_path))
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=600)
        return {"ran": True, "returncode": proc.returncode, "stderr": (proc.stderr or "")[:2000]}
    except Exception as exc:  # noqa: BLE001 - off-VPS copy is optional, never fatal to the local snapshot
        logger.warning("backup: off_vps_command failed: %s", exc, exc_info=True)
        return {"ran": True, "error": str(exc)}


# ---------------------------------------------------------------------------
# Doctor entry point
# ---------------------------------------------------------------------------


def run_doctor(memobase_cfg: Optional[Dict[str, Any]] = None, *, db_path: Optional[Path] = None,
                backup_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Nightly doctor step: VACUUM INTO snapshot + rotation + disk check +
    optional off-VPS copy. Never raises — every step is wrapped, and the
    result dict's ``ok`` field is the single "did everything succeed" flag
    a cron wrapper should alert on.
    """
    from . import config as kb_config
    from . import db as db_mod

    memobase_cfg = memobase_cfg if memobase_cfg is not None else kb_config.get_memobase_config_readonly()
    backup_cfg = memobase_cfg.get("backup") or {}
    keep = int(backup_cfg.get("keep", 7))
    alert_pct = float(backup_cfg.get("disk_alert_pct", 80))
    off_vps_command = (backup_cfg.get("off_vps_command") or "").strip()

    source_db = db_path or db_mod.get_db_path()
    dest_dir = backup_dir or default_backup_dir()

    result: Dict[str, Any] = {"ok": True, "steps": {}}

    if not source_db.exists():
        result["ok"] = False
        result["steps"]["snapshot"] = {"ok": False, "error": f"memobase.db not found at {source_db}"}
        return result
    try:
        same_dir = Path(dest_dir).resolve() == Path(source_db).resolve().parent
    except OSError:
        same_dir = False
    if same_dir:
        result["ok"] = False
        result["steps"]["snapshot"] = {
            "ok": False,
            "error": f"backup_dir совпадает с каталогом базы ({dest_dir}) — укажите отдельный каталог",
        }
        return result

    snapshot_path = dest_dir / snapshot_filename()
    try:
        vacuum_into_snapshot(source_db, snapshot_path)
        result["steps"]["snapshot"] = {"ok": True, "path": str(snapshot_path)}
    except BackupError as exc:
        result["ok"] = False
        result["steps"]["snapshot"] = {"ok": False, "error": str(exc)}
        return result  # no point rotating/off-VPS-copying if there's nothing new

    deleted = rotate_backups(dest_dir, keep)
    result["steps"]["rotation"] = {"ok": True, "deleted": [str(p) for p in deleted], "kept": keep}

    disk = check_disk_usage(dest_dir, alert_pct)
    result["steps"]["disk"] = disk
    if disk.get("alert"):
        result["ok"] = False
        logger.warning(
            "memobase backup: disk usage at %.1f%% (alert threshold %.1f%%) near %s",
            disk.get("used_pct"), alert_pct, dest_dir,
        )

    if off_vps_command:
        result["steps"]["off_vps"] = _run_off_vps_copy(off_vps_command, snapshot_path)

    return result


def format_report(result: Dict[str, Any]) -> str:
    steps = result.get("steps", {})
    lines = [f"Бэкап базы знаний: {'успешно' if result.get('ok') else 'ЕСТЬ ПРОБЛЕМЫ'}."]
    snap = steps.get("snapshot", {})
    if snap.get("ok"):
        lines.append(f"  Снимок: {snap.get('path')}")
    else:
        lines.append(f"  Снимок НЕ создан: {snap.get('error')}")
    rot = steps.get("rotation", {})
    if rot:
        lines.append(f"  Ротация: оставлено {rot.get('kept')}, удалено старых {len(rot.get('deleted', []))}")
    disk = steps.get("disk", {})
    if disk.get("used_pct") is not None:
        marker = " (ВНИМАНИЕ)" if disk.get("alert") else ""
        lines.append(f"  Диск: занято {disk['used_pct']}% из {disk.get('total_gb')} ГБ{marker}")
    if "off_vps" in steps:
        ov = steps["off_vps"]
        lines.append(f"  Off-VPS копия: {'выполнена' if ov.get('returncode') == 0 else 'см. лог'}")
    return "\n".join(lines)


def register(ctx: Any) -> None:
    """No tools/hooks of its own — `run_doctor` is invoked from
    ``hermes memobase backup-run`` (cli.py) and is intended to be wired into
    ``hermes cron`` by the operator. Kept as a symmetrical
    ``register(ctx)`` no-op so ``__init__.py`` can call it uniformly with
    every other submodule without a special case.
    """
    return None
