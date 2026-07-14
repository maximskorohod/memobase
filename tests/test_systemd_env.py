"""Tests for systemd_env.py -- the systemd --user drop-in helper for bug B12
("systemd .env gap"): ``hermes-gateway.service`` builds its environment once
at unit start and never re-reads ``.env`` itself, so a plain ``hermes
gateway restart`` never picks up a freshly-saved key without this module's
``EnvironmentFile`` drop-in wired in (by ``install.sh``).

Everything here uses a FAKE runner (never invokes the real ``systemctl``)
and a tmp ``config_home`` (never touches the real
``~/.config/systemd/user/``) -- consistent with conftest.py's `kb` fixture
isolation pattern used across this suite, plus explicit dependency
injection (``runner=``) here since `kb`'s HERMES_HOME isolation alone says
nothing about the (real, ambient) systemd user session this module would
otherwise shell out to.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture()
def systemd_env(kb):
    """A handle to THIS test run's freshly-imported copy of systemd_env.py,
    imported under the same synthetic package ``kb`` was loaded under (so
    it is exactly the module setup_core.py's own ``from . import
    systemd_env`` resolves to). setup_core.py imports it lazily
    (function-local, inside ``restart_hint()``), so it is not already an
    attribute of ``kb`` merely from loading the package -- import it
    directly here instead.
    """
    return importlib.import_module(f"{kb.__name__}.systemd_env")


class FakeCompletedProcess:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


class FakeRunner:
    """Records every invocation as a plain list-of-args; never touches the
    real ``systemctl``. Routes by presence of ``"cat"``/``"daemon-reload"``
    in the argv list (list containment, not substring -- exact and simple)."""

    def __init__(self, cat_returncode: int = 0, reload_returncode: int = 0):
        self.cat_returncode = cat_returncode
        self.reload_returncode = reload_returncode
        self.calls: list = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if "cat" in cmd:
            return FakeCompletedProcess(self.cat_returncode)
        if "daemon-reload" in cmd:
            return FakeCompletedProcess(self.reload_returncode)
        return FakeCompletedProcess(0)


def _fake_systemctl_present(systemd_env, monkeypatch, present: bool = True):
    monkeypatch.setattr(systemd_env.shutil, "which", lambda name: ("/usr/bin/systemctl" if present else None))


# ===========================================================================
# detect_systemd_gateway
# ===========================================================================


class TestDetectSystemdGateway:
    def test_true_when_systemctl_cat_succeeds(self, systemd_env, monkeypatch):
        _fake_systemctl_present(systemd_env, monkeypatch)
        runner = FakeRunner(cat_returncode=0)
        assert systemd_env.detect_systemd_gateway(runner=runner) is True
        assert runner.calls == [["systemctl", "--user", "cat", systemd_env.GATEWAY_UNIT]]

    def test_false_when_systemctl_not_on_path(self, systemd_env, monkeypatch):
        _fake_systemctl_present(systemd_env, monkeypatch, present=False)
        runner = FakeRunner(cat_returncode=0)
        assert systemd_env.detect_systemd_gateway(runner=runner) is False
        assert runner.calls == [], "must short-circuit before ever invoking the runner"

    def test_false_when_unit_does_not_exist(self, systemd_env, monkeypatch):
        _fake_systemctl_present(systemd_env, monkeypatch)
        runner = FakeRunner(cat_returncode=1)
        assert systemd_env.detect_systemd_gateway(runner=runner) is False

    def test_false_when_runner_raises(self, systemd_env, monkeypatch):
        _fake_systemctl_present(systemd_env, monkeypatch)

        def boom(cmd, **kwargs):
            raise OSError("no systemd user session (e.g. no DBus / non-interactive shell)")

        assert systemd_env.detect_systemd_gateway(runner=boom) is False

    def test_false_with_default_runner_when_systemctl_absent(self, systemd_env, monkeypatch):
        # Exercises the true default `runner=subprocess.run` (never passed
        # explicitly) without ever actually invoking it -- deterministic
        # regardless of this host's real PATH, since shutil.which is forced
        # absent explicitly rather than relied on ambiently.
        monkeypatch.setattr(systemd_env.shutil, "which", lambda name: None)
        assert systemd_env.detect_systemd_gateway() is False


# ===========================================================================
# wire_gateway_env
# ===========================================================================


class TestWireGatewayEnv:
    def test_guard_returns_false_off_systemd_and_writes_nothing(self, systemd_env, monkeypatch, tmp_path):
        _fake_systemctl_present(systemd_env, monkeypatch, present=False)
        config_home = tmp_path / "config"
        wired, msg = systemd_env.wire_gateway_env(
            str(tmp_path / "hermes-home"), config_home=config_home, runner=FakeRunner()
        )
        assert wired is False
        assert "systemd" in msg.lower()
        assert not config_home.exists(), "must not create anything when the guard fails"

    def test_off_systemd_noop_with_default_runner(self, systemd_env, monkeypatch, tmp_path):
        # Same guard, but through the real default `runner=subprocess.run`.
        monkeypatch.setattr(systemd_env.shutil, "which", lambda name: None)
        config_home = tmp_path / "config"
        wired, msg = systemd_env.wire_gateway_env(str(tmp_path / "hermes-home"), config_home=config_home)
        assert wired is False
        assert not config_home.exists()

    def test_writes_expected_dropin_content(self, systemd_env, monkeypatch, tmp_path):
        _fake_systemctl_present(systemd_env, monkeypatch)
        runner = FakeRunner(cat_returncode=0, reload_returncode=0)
        hermes_home = tmp_path / "hermes-home"
        config_home = tmp_path / "config"

        wired, msg = systemd_env.wire_gateway_env(str(hermes_home), config_home=config_home, runner=runner)

        assert wired is True
        drop_in = config_home / "systemd" / "user" / f"{systemd_env.GATEWAY_UNIT}.d" / systemd_env.DROP_IN_NAME
        assert drop_in.exists()
        content = drop_in.read_text(encoding="utf-8")
        expected_env_path = (hermes_home / ".env").as_posix()
        assert content == f"[Service]\nEnvironmentFile=-{expected_env_path}\n"
        assert str(drop_in) in msg

    def test_leading_dash_makes_missing_env_optional_for_systemd(self, systemd_env, monkeypatch, tmp_path):
        # The leading '-' in `EnvironmentFile=-...` is what makes a MISSING
        # .env a non-error for systemd -- a fresh install with no keys saved
        # yet must still start cleanly. Pinned explicitly: easy to drop by
        # accident in a refactor, and silently breaks fresh installs.
        _fake_systemctl_present(systemd_env, monkeypatch)
        hermes_home = tmp_path / "hermes-home"  # deliberately never created -- no .env exists
        config_home = tmp_path / "config"
        systemd_env.wire_gateway_env(str(hermes_home), config_home=config_home, runner=FakeRunner())
        drop_in = config_home / "systemd" / "user" / f"{systemd_env.GATEWAY_UNIT}.d" / systemd_env.DROP_IN_NAME
        assert "EnvironmentFile=-" in drop_in.read_text(encoding="utf-8")

    def test_calls_daemon_reload_after_writing(self, systemd_env, monkeypatch, tmp_path):
        _fake_systemctl_present(systemd_env, monkeypatch)
        runner = FakeRunner(cat_returncode=0, reload_returncode=0)
        systemd_env.wire_gateway_env(str(tmp_path / "hermes-home"), config_home=tmp_path / "config", runner=runner)
        assert ["systemctl", "--user", "daemon-reload"] in runner.calls

    def test_idempotent_rerun_produces_identical_file_and_reloads_again(self, systemd_env, monkeypatch, tmp_path):
        _fake_systemctl_present(systemd_env, monkeypatch)
        hermes_home = tmp_path / "hermes-home"
        config_home = tmp_path / "config"
        runner1 = FakeRunner(cat_returncode=0, reload_returncode=0)
        runner2 = FakeRunner(cat_returncode=0, reload_returncode=0)

        wired1, _ = systemd_env.wire_gateway_env(str(hermes_home), config_home=config_home, runner=runner1)
        drop_in = config_home / "systemd" / "user" / f"{systemd_env.GATEWAY_UNIT}.d" / systemd_env.DROP_IN_NAME
        content_after_first = drop_in.read_text(encoding="utf-8")

        wired2, _ = systemd_env.wire_gateway_env(str(hermes_home), config_home=config_home, runner=runner2)
        content_after_second = drop_in.read_text(encoding="utf-8")

        assert wired1 is True and wired2 is True
        assert content_after_first == content_after_second
        assert ["systemctl", "--user", "daemon-reload"] in runner2.calls

    def test_never_touches_the_hermes_shipped_unit_file(self, systemd_env, monkeypatch, tmp_path):
        # Only the .d/ drop-in directory may be written -- the unit file
        # itself (hermes-gateway.service, without .d) must never be created
        # or modified by this module; it only ever reads it (via `cat`).
        _fake_systemctl_present(systemd_env, monkeypatch)
        config_home = tmp_path / "config"
        systemd_env.wire_gateway_env(str(tmp_path / "hermes-home"), config_home=config_home, runner=FakeRunner())
        unit_file = config_home / "systemd" / "user" / systemd_env.GATEWAY_UNIT
        assert not unit_file.exists()

    def test_daemon_reload_failure_still_reports_wired_true_with_note(self, systemd_env, monkeypatch, tmp_path):
        _fake_systemctl_present(systemd_env, monkeypatch)

        class ReloadFailsRunner(FakeRunner):
            def __call__(self, cmd, **kwargs):
                self.calls.append(list(cmd))
                if "cat" in cmd:
                    return FakeCompletedProcess(0)
                raise OSError("daemon-reload: no systemd user session")

        wired, msg = systemd_env.wire_gateway_env(
            str(tmp_path / "hermes-home"), config_home=tmp_path / "config", runner=ReloadFailsRunner()
        )
        # The drop-in FILE is the primary deliverable -- a daemon-reload
        # hiccup is surfaced in the message, not treated as a hard failure
        # of the whole operation (the file is still on disk, effective on
        # the next systemd restart/reload regardless).
        assert wired is True
        assert "daemon-reload" in msg

    def test_write_failure_returns_false(self, systemd_env, monkeypatch, tmp_path):
        _fake_systemctl_present(systemd_env, monkeypatch)
        # Make config_home's parent a FILE, not a directory, so
        # mkdir(parents=True) underneath it fails with a real OSError --
        # exercises the actual failure path without monkeypatching Path
        # globally.
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("blocking file", encoding="utf-8")

        wired, msg = systemd_env.wire_gateway_env(
            str(tmp_path / "hermes-home"), config_home=blocked / "config", runner=FakeRunner()
        )
        assert wired is False
        assert "drop-in" in msg.lower() or "не удалось" in msg.lower()

    def test_returns_false_and_message_when_never_called_under_systemd_guard_twice(self, systemd_env, monkeypatch, tmp_path):
        # Calling wire_gateway_env() twice while consistently off-systemd
        # must keep returning the same guarded (False, ...) result --
        # no partial state accumulates across repeated calls.
        _fake_systemctl_present(systemd_env, monkeypatch, present=False)
        config_home = tmp_path / "config"
        first = systemd_env.wire_gateway_env(str(tmp_path / "hermes-home"), config_home=config_home, runner=FakeRunner())
        second = systemd_env.wire_gateway_env(str(tmp_path / "hermes-home"), config_home=config_home, runner=FakeRunner())
        assert first == second


# ===========================================================================
# _default_config_home
# ===========================================================================


class TestDefaultConfigHome:
    def test_respects_xdg_config_home_when_set(self, systemd_env, monkeypatch, tmp_path):
        xdg = tmp_path / "xdg-config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        assert systemd_env._default_config_home() == xdg

    def test_falls_back_to_dot_config_when_xdg_unset(self, systemd_env, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        assert systemd_env._default_config_home() == Path.home() / ".config"

    def test_explicit_config_home_overrides_the_default_in_wire_gateway_env(self, systemd_env, monkeypatch, tmp_path):
        # wire_gateway_env's own `config_home=` kwarg must win even when
        # XDG_CONFIG_HOME is set to something else entirely.
        _fake_systemctl_present(systemd_env, monkeypatch)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "should-not-be-used"))
        explicit = tmp_path / "explicit-config"
        systemd_env.wire_gateway_env(str(tmp_path / "hermes-home"), config_home=explicit, runner=FakeRunner())
        drop_in = explicit / "systemd" / "user" / f"{systemd_env.GATEWAY_UNIT}.d" / systemd_env.DROP_IN_NAME
        assert drop_in.exists()
        assert not (tmp_path / "should-not-be-used").exists()
