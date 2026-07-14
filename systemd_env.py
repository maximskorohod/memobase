"""systemd --user drop-in for the ``hermes-gateway.service`` unit (bug B12:
"systemd .env gap").

The bug: hermes's systemd **user** unit builds its process environment once,
at unit start, and never reads ``.env`` itself (the shipped unit has no
``EnvironmentFile=``). So when a key is saved via
:func:`setup_core.write_env_secret` (which upserts a plain ``KEY=value``
line into ``<HERMES_HOME>/.env`` -- a format already 100% compatible with
systemd's ``EnvironmentFile=``) and the user runs ``hermes gateway
restart``, that restart just re-execs the same unit with the SAME stale
environment -- the new key never reaches the gateway process. Nothing about
the WRITE side needed to change; the gap was purely on the systemd side.

This module is the fix: it points the gateway unit at ``.env`` via the
sanctioned systemd override mechanism -- a drop-in file under
``<config_home>/systemd/user/hermes-gateway.service.d/`` -- rather than
editing the hermes-shipped unit file itself (which an ``hermes`` upgrade
could overwrite/revert at any time). Run automatically at the end of
``install.sh``; also callable standalone: ``python systemd_env.py
[HERMES_HOME]``.

Platform-safe to import/call anywhere: on any machine where the gateway is
NOT a systemd --user unit (Windows, macOS, a plain foreground/CLI-only
hermes, some other supervisor) every function here degrades to an honest
no-op instead of raising -- see :func:`detect_systemd_gateway`'s guard,
which every other function in this module checks first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

# The gateway unit this plugin wires an EnvironmentFile into. Never created,
# renamed, enabled, or edited by this module -- only ever read (`cat`, to
# detect it exists under systemd --user) and pointed-at via our OWN drop-in.
GATEWAY_UNIT = "hermes-gateway.service"

# Named (not "override.conf") so it's obviously ours in an `ls` of the
# .d/ directory if some other tool ever drops its own override alongside it.
DROP_IN_NAME = "10-memobase-env.conf"


def _default_config_home() -> Path:
    """``$XDG_CONFIG_HOME`` if set -- the same env var ``systemctl --user``
    itself consults -- else ``~/.config``. Callers that don't pass an
    explicit ``config_home`` get the directory systemd will actually load
    drop-ins from on THIS machine, not a hardcoded guess."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg) if xdg else Path.home() / ".config"


def detect_systemd_gateway(*, runner=subprocess.run) -> bool:
    """True if the hermes gateway is registered as a systemd --user unit
    (``systemctl --user cat hermes-gateway.service`` succeeds).

    Safely ``False`` -- never raises -- when: ``systemctl`` isn't on PATH
    at all (fast path, skips the subprocess call entirely on Windows/macOS
    dev boxes), the unit doesn't exist under systemd --user, or the
    ``runner`` call itself blows up for any reason (no systemd user
    session/manager running, permissions, whatever). This is the guard
    every other function in this module checks first -- a hermes install
    that is NOT systemd-supervised must see a silent no-op here, never an
    exception.
    """
    if shutil.which("systemctl") is None:
        return False
    try:
        result = runner(
            ["systemctl", "--user", "cat", GATEWAY_UNIT],
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return getattr(result, "returncode", 1) == 0


def wire_gateway_env(hermes_home, *, config_home=None, runner=subprocess.run) -> Tuple[bool, str]:
    """Idempotently wire ``hermes-gateway.service`` to read
    ``<hermes_home>/.env`` on every start/restart, via a drop-in override
    (never touches the hermes-shipped unit file itself).

    Writes ``{config_home or $XDG_CONFIG_HOME or ~/.config}/systemd/user/
    hermes-gateway.service.d/10-memobase-env.conf``::

        [Service]
        EnvironmentFile=-<hermes_home>/.env

    The leading ``-`` marks the file optional for systemd -- a fresh
    install with no ``.env`` yet (no keys saved) must still start cleanly,
    not fail unit activation. Re-running always rewrites the same
    deterministic content and re-runs ``daemon-reload``, so this is safe to
    call on every ``install.sh`` run, not just the first.

    Returns ``(wired, message)`` and never raises -- every failure mode
    (not under systemd, can't create/write the drop-in, daemon-reload
    fails) degrades to ``(False, <RU explanation>)`` or, once the file
    itself is safely written, ``(True, <RU note about daemon-reload>)`` --
    so callers (``install.sh`` via its own ``|| true``, the setup wizards'
    :func:`setup_core.restart_hint`) can surface the outcome without ever
    crashing the onboarding/install flow.
    """
    if not detect_systemd_gateway(runner=runner):
        return False, "hermes gateway не под systemd, пропускаю"

    config_dir = Path(config_home) if config_home else _default_config_home()
    drop_in_dir = config_dir / "systemd" / "user" / f"{GATEWAY_UNIT}.d"
    drop_in_path = drop_in_dir / DROP_IN_NAME
    env_path = Path(hermes_home) / ".env"
    # .as_posix(): the target is always a POSIX system (systemd itself is
    # Linux-only, per detect_systemd_gateway()'s own guard) -- a unit file
    # must get forward slashes regardless of what separator style the host
    # Path object would otherwise render (relevant for this module's own
    # test suite, which exercises this code path on Windows too).
    content = f"[Service]\nEnvironmentFile=-{env_path.as_posix()}\n"

    try:
        drop_in_dir.mkdir(parents=True, exist_ok=True)
        drop_in_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return False, f"не удалось записать systemd drop-in ({exc}) — настройте EnvironmentFile вручную"

    try:
        runner(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True)
    except Exception as exc:
        return True, (
            f"systemd drop-in записан ({drop_in_path}), но 'systemctl --user daemon-reload' "
            f"не выполнился ({exc}) — выполните его вручную, иначе изменение подхватится не сразу"
        )

    return True, f"systemd drop-in настроен ({drop_in_path}) — ключи из .env подхватятся при 'hermes gateway restart'"


if __name__ == "__main__":
    import sys

    _, msg = wire_gateway_env(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".hermes"))
    print(msg)
