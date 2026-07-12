"""Shared pytest fixtures for memobase's test suite.

Isolation pattern (DESIGN_v1.md "Tests" section, matching
``tests/plugins/test_disk_cleanup_plugin.py`` in the hermes-agent checkout):
monkeypatch ``HERMES_HOME`` to a per-test tmp dir, and load the plugin's
modules via a synthetic ``hermes_plugins.<slug>``-shaped namespace package so
relative imports (``from . import db``, etc.) resolve exactly as they would
under the real ``PluginManager`` loader.

Generalization for a MULTI-module plugin (disk-cleanup's test only ever
loaded one library file): :func:`load_plugin_package` imports a fresh copy of
the WHOLE memobase package (all ~17 modules) under a brand-new synthetic
top-level package name every call, so every test gets fully isolated
module-level state (``tools._session_collection``, ``stem._stemmer``'s
lazy-loaded singleton, etc.) with zero cross-test leakage risk, at the cost of
re-executing the (cheap, no heavy imports at module scope) module bodies each
time.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

# `--import-mode=importlib` (pytest.ini) deliberately does NOT add the tests
# directory to sys.path -- add it explicitly so `from _helpers import ...`
# in test_plugin.py resolves without needing tests/__init__.py (which would
# turn this into a package and complicate the synthetic-namespace trick
# below for no benefit).
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Captured at conftest IMPORT time -- i.e. before any test's monkeypatch has
# run -- so this always reflects the REAL ambient hermes installation, not a
# test's isolated tmp HERMES_HOME. Used only for (a) the real-HERMES_HOME
# untouched guard below and (b) reading real API keys for @pytest.mark.integration.
REAL_HERMES_HOME = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
REAL_ENV_PATH = REAL_HERMES_HOME / ".env"


def load_plugin_package() -> Any:
    """Import a FRESH copy of the entire memobase plugin package (__init__
    + every submodule reachable from it) under a unique synthetic package
    name, and return the __init__ module object.

    All submodules become available as attributes of the returned object
    (``pkg.db``, ``pkg.security``, ``pkg.tools``, ...) -- this is a plain
    consequence of Python's import system always setting an imported
    submodule as an attribute of its parent package, not something this
    helper does manually.
    """
    suffix = uuid.uuid4().hex[:10]
    pkg_name = f"hermes_plugins_test_{suffix}"
    full_pkg = f"{pkg_name}.memobase"

    ns = types.ModuleType(pkg_name)
    ns.__path__ = []
    sys.modules[pkg_name] = ns

    spec = importlib.util.spec_from_file_location(
        full_pkg,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = full_pkg
    mod.__path__ = [str(PLUGIN_DIR)]
    sys.modules[full_pkg] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def kb(tmp_path, monkeypatch):
    """The single fixture almost every test uses: isolates HERMES_HOME to a
    fresh tmp dir AND returns a freshly-imported copy of the whole plugin
    package. Depend on this (not a bare ``tmp_path``) whenever a test touches
    config/db/any module that reads ``HERMES_HOME``.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # HERMES_HOME_OVERRIDE / profile machinery some hermes-core helpers check;
    # unset so get_hermes_home() takes the plain HERMES_HOME env path.
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    pkg = load_plugin_package()
    pkg._hermes_home_for_test = hermes_home  # convenience for tests that want the Path directly
    return pkg


class FakeLlmResult:
    def __init__(self, text: str):
        self.text = text
        self.provider = "fake"
        self.model = "fake"
        self.usage = {}
        self.audit = None


class FakeLlm:
    """Minimal stand-in for ``ctx.llm`` (agent.plugin_llm.PluginLlm façade).
    Tests set ``.next_text`` (or ``.responses`` for a queue) to control what
    ``complete()`` returns; ``.calls`` records every invocation for
    assertions."""

    def __init__(self, next_text: str = ""):
        self.next_text = next_text
        self.responses: list = []
        self.calls: list = []
        self.raise_on_complete: Exception | None = None

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.raise_on_complete is not None:
            raise self.raise_on_complete
        if self.responses:
            return FakeLlmResult(self.responses.pop(0))
        return FakeLlmResult(self.next_text)


@pytest.fixture()
def fake_llm():
    return FakeLlm()


class FakeCtx:
    """Minimal stand-in for ``hermes_cli.plugins.PluginContext`` -- just
    enough surface for ``__init__.py``'s ``register(ctx)`` and its
    sub-registrars (tools.py/commands.py/cli.py) to run against, with every
    call recorded for assertions (mirrors the real registrars' documented
    signatures exactly)."""

    def __init__(self, llm: Any = None):
        self.llm = llm
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.hooks: Dict[str, list] = {}
        self.commands: Dict[str, Dict[str, Any]] = {}
        self.cli_commands: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name, toolset, schema, handler, check_fn=None,
                       requires_env=None, is_async=False, description="", emoji="", override=False):
        self.tools[name] = {
            "toolset": toolset, "schema": schema, "handler": handler,
            "requires_env": requires_env, "is_async": is_async, "emoji": emoji,
        }

    def register_hook(self, event: str, callback):
        self.hooks.setdefault(event, []).append(callback)

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {"handler": handler, "description": description, "args_hint": args_hint}

    def register_cli_command(self, name, help, setup_fn, handler_fn=None, description=""):
        self.cli_commands[name] = {
            "help": help, "setup_fn": setup_fn, "handler_fn": handler_fn, "description": description,
        }


@pytest.fixture()
def fake_ctx(fake_llm):
    return FakeCtx(llm=fake_llm)


def real_env_keys() -> Dict[str, str]:
    """Read CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN/COHERE_API_KEY from
    the REAL ``~/.hermes/.env`` (never hardcoded, never committed). Returns
    an empty dict (never raises) if python-dotenv or the file is unavailable
    -- @pytest.mark.integration tests must treat that as "skip", not "fail".
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    if not REAL_ENV_PATH.exists():
        return {}
    try:
        values = dotenv_values(REAL_ENV_PATH)
    except Exception:
        return {}
    wanted = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "COHERE_API_KEY")
    return {k: v for k, v in values.items() if k in wanted and v}


@pytest.fixture()
def real_api_env(monkeypatch):
    """For @pytest.mark.integration tests only: force CLOUDFLARE_ACCOUNT_ID/
    CLOUDFLARE_API_TOKEN/COHERE_API_KEY in os.environ to the REAL values from
    ~/.hermes/.env, overriding whatever the ambient shell/process environment
    already has set.

    This matters because the process this test suite runs in may have
    STALE copies of these vars inherited from its own parent shell/session
    (observed in practice: a shell's ambient CLOUDFLARE_API_TOKEN did not
    match the current ~/.hermes/.env value at all -- hashes differed
    entirely -- causing a live 401 against a token that .env itself proves
    is valid). Embedding/reranking code reads straight from os.environ, not
    from .env, so without this fixture an integration test could fail on a
    stale credential and be misdiagnosed as a client bug.
    """
    keys = real_env_keys()
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    return keys


def pytest_collection_modifyitems(config, items):
    keys = real_env_keys()
    missing = [k for k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "COHERE_API_KEY") if k not in keys]
    if not missing:
        return
    skip_marker = pytest.mark.skip(
        reason=f"integration test needs real key(s) in {REAL_ENV_PATH}: missing {', '.join(missing)}"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Session-wide guard: the REAL HERMES_HOME's kb/ state and config.yaml must
# never be touched by this suite -- every test isolates HERMES_HOME via the
# `kb` fixture above. This fixture is deliberately narrow (kb/ dir + top-level
# config.yaml mtime) rather than a full-tree rglob snapshot of HERMES_HOME,
# because the real HERMES_HOME here happens to CONTAIN the hermes-agent repo
# checkout itself (importing hermes_constants/hermes_cli during collection
# regenerates __pycache__ *.pyc files there as a completely normal, harmless
# side effect of importing Python modules) -- a full-tree check would produce
# constant false positives unrelated to what this guard actually cares about.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _guard_real_hermes_home():
    kb_dir = REAL_HERMES_HOME / "memobase"
    config_path = REAL_HERMES_HOME / "config.yaml"

    kb_existed_before = kb_dir.exists()
    kb_listing_before = sorted(str(p) for p in kb_dir.rglob("*")) if kb_existed_before else None
    config_mtime_before = config_path.stat().st_mtime if config_path.exists() else None

    yield

    kb_existed_after = kb_dir.exists()
    assert kb_existed_after == kb_existed_before, (
        f"REAL HERMES_HOME/memobase ({kb_dir}) existence changed during the test run -- "
        "some code path escaped the per-test HERMES_HOME isolation (the `kb` fixture) "
        "and wrote to the real memobase.db."
    )
    if kb_existed_before:
        kb_listing_after = sorted(str(p) for p in kb_dir.rglob("*"))
        assert kb_listing_after == kb_listing_before, (
            f"REAL HERMES_HOME/memobase ({kb_dir}) contents changed during the test run."
        )

    config_mtime_after = config_path.stat().st_mtime if config_path.exists() else None
    assert config_mtime_after == config_mtime_before, (
        f"REAL HERMES_HOME/config.yaml ({config_path}) was modified during the test run -- "
        "a test must have called save_config()/set_config_value() against the real config "
        "instead of the isolated tmp HERMES_HOME."
    )
