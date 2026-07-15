"""Tests for the read-only key-visibility surface added to setup_core.py:
``env_file_path`` / ``key_status`` / ``relevant_keys`` / ``format_keys_block``.

Goal of the feature (why these exist): after `/memobase setup` a human should
see "keys are here, this is the .env path, this is what's configured" WITHOUT
hand-opening the file — and the status must catch the systemd bug-B12 case
(key written to .env but the running process started before it, so it isn't in
``os.environ`` yet).

Same isolation as the rest of the suite: the ``kb`` fixture points HERMES_HOME
at a fresh tmp dir, so ``env_file_path()`` is ``<tmp>/.hermes/.env`` and writing
there never touches the real ``~/.hermes/.env``. Every real key name asserted on
is explicitly cleared from ``os.environ`` first, because the process running the
suite may itself carry ambient CLOUDFLARE_*/COHERE_* vars (see conftest.py).
No network anywhere.
"""

from __future__ import annotations

_MEMOBASE_KEYS = (
    "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "COHERE_API_KEY",
    "OPENAI_API_KEY", "SCRAPECREATORS_API_KEY", "APIFY_TOKEN",
    "GROQ_API_KEY", "GEMINI_API_KEY",
)


def _clear_all(monkeypatch):
    for k in _MEMOBASE_KEYS:
        monkeypatch.delenv(k, raising=False)


# ===========================================================================
# env_file_path
# ===========================================================================


class TestEnvFilePath:
    def test_points_at_hermes_home_dotenv(self, kb):
        assert kb.setup_core.env_file_path() == kb._hermes_home_for_test / ".env"


# ===========================================================================
# key_status — in_file / in_process / mask
# ===========================================================================


class TestKeyStatus:
    def test_key_only_in_file_is_the_b12_case(self, kb, monkeypatch):
        # Written to .env but NOT in the process env -> in_file True,
        # in_process False (the gateway started before the key was added).
        _clear_all(monkeypatch)
        kb.setup_core.env_file_path().write_text(
            "CLOUDFLARE_API_TOKEN=cfat_FILEONLY_SECRETTAIL\n", encoding="utf-8"
        )
        st = kb.setup_core.key_status(["CLOUDFLARE_API_TOKEN"])["CLOUDFLARE_API_TOKEN"]
        assert st["in_file"] is True
        assert st["in_process"] is False
        assert st["mask"] == "cfat…"

    def test_key_in_process_is_seen(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_LIVE_SECRETTAIL")
        st = kb.setup_core.key_status(["GROQ_API_KEY"])["GROQ_API_KEY"]
        assert st["in_process"] is True
        assert st["mask"] == "gsk_…"

    def test_key_nowhere_is_empty(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        st = kb.setup_core.key_status(["APIFY_TOKEN"])["APIFY_TOKEN"]
        assert st["in_file"] is False
        assert st["in_process"] is False
        assert st["mask"] == ""

    def test_commented_and_empty_lines_do_not_count(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        kb.setup_core.env_file_path().write_text(
            "# COHERE_API_KEY=commentedOut\nGEMINI_API_KEY=\n", encoding="utf-8"
        )
        got = kb.setup_core.key_status(["COHERE_API_KEY", "GEMINI_API_KEY"])
        assert got["COHERE_API_KEY"]["in_file"] is False
        assert got["GEMINI_API_KEY"]["in_file"] is False

    def test_mask_never_leaks_full_secret(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        monkeypatch.setenv("APIFY_TOKEN", "apify_SUPERSECRETTAIL_9999")
        st = kb.setup_core.key_status(["APIFY_TOKEN"])["APIFY_TOKEN"]
        assert "SUPERSECRETTAIL" not in st["mask"]
        assert st["mask"] == "apif…"

    def test_missing_env_file_yields_all_absent(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        # No .env written at all — must not raise, everything absent.
        assert not kb.setup_core.env_file_path().exists()
        got = kb.setup_core.key_status(["COHERE_API_KEY"])
        assert got["COHERE_API_KEY"] == {"in_file": False, "in_process": False, "mask": ""}


# ===========================================================================
# relevant_keys — catalog derived from effective config
# ===========================================================================


class TestRelevantKeys:
    def _vars(self, items):
        return [i["env_var"] for i in items]

    def test_cloudflare_embedder_needs_account_and_token(self, kb):
        cfg = {"embedder": {"provider": "cloudflare"}, "rerank": {"enabled": True, "provider": "cohere"}}
        items = kb.setup_core.relevant_keys(cfg, include_youtube=False)
        vars_ = self._vars(items)
        assert "CLOUDFLARE_ACCOUNT_ID" in vars_ and "CLOUDFLARE_API_TOKEN" in vars_
        # both embedder keys are required
        for i in items:
            if i["env_var"].startswith("CLOUDFLARE_"):
                assert i["required"] is True

    def test_cohere_embedder_uses_cohere_key_required(self, kb):
        cfg = {"embedder": {"provider": "cohere"}, "rerank": {"enabled": False}}
        items = kb.setup_core.relevant_keys(cfg, include_youtube=False)
        cohere = [i for i in items if i["env_var"] == "COHERE_API_KEY"]
        assert len(cohere) == 1 and cohere[0]["required"] is True

    def test_local_embedder_needs_no_embedder_key(self, kb):
        cfg = {"embedder": {"provider": "local"}, "rerank": {"enabled": True, "provider": "cohere"}}
        vars_ = self._vars(kb.setup_core.relevant_keys(cfg, include_youtube=False))
        assert "CLOUDFLARE_ACCOUNT_ID" not in vars_ and "OPENAI_API_KEY" not in vars_
        # rerank still contributes an optional cohere key
        assert "COHERE_API_KEY" in vars_

    def test_rerank_adds_optional_cohere(self, kb):
        cfg = {"embedder": {"provider": "cloudflare"}, "rerank": {"enabled": True, "provider": "cohere"}}
        items = kb.setup_core.relevant_keys(cfg, include_youtube=False)
        cohere = [i for i in items if i["env_var"] == "COHERE_API_KEY"]
        assert len(cohere) == 1 and cohere[0]["required"] is False

    def test_youtube_keys_toggle(self, kb):
        cfg = {"embedder": {"provider": "cloudflare"}}
        with_yt = self._vars(kb.setup_core.relevant_keys(cfg, include_youtube=True))
        without_yt = self._vars(kb.setup_core.relevant_keys(cfg, include_youtube=False))
        assert "SCRAPECREATORS_API_KEY" in with_yt and "APIFY_TOKEN" in with_yt
        assert "SCRAPECREATORS_API_KEY" not in without_yt

    def test_empty_config_defaults_to_cloudflare(self, kb):
        # No embedder set -> default provider is cloudflare (config.DEFAULTS).
        vars_ = self._vars(kb.setup_core.relevant_keys({}, include_youtube=False))
        assert "CLOUDFLARE_ACCOUNT_ID" in vars_


# ===========================================================================
# format_keys_block — the rendered "Ключи" section
# ===========================================================================


class TestFormatKeysBlock:
    def _cfg(self):
        return {"embedder": {"provider": "cloudflare"}, "rerank": {"enabled": True, "provider": "cohere"}}

    def test_shows_path_and_check_and_cross_marks(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cfat_LIVE_TAIL")  # configured (process)
        block = kb.setup_core.format_keys_block(self._cfg(), include_youtube=False)
        assert str(kb.setup_core.env_file_path()) in block
        assert "CLOUDFLARE_API_TOKEN — ✓ настроен (cfat…)" in block
        assert "CLOUDFLARE_ACCOUNT_ID — ✗ нет" in block  # not set anywhere

    def test_stale_key_in_file_only_warns_with_restart_hint(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        # Force the "not under systemd" branch so restart_hint text is deterministic.
        import importlib
        systemd_env = importlib.import_module(f"{kb.__name__}.systemd_env")
        monkeypatch.setattr(systemd_env, "detect_systemd_gateway", lambda **kw: False)
        kb.setup_core.env_file_path().write_text(
            "CLOUDFLARE_API_TOKEN=cfat_ONLYINFILE\n", encoding="utf-8"
        )
        block = kb.setup_core.format_keys_block(self._cfg(), include_youtube=False)
        assert "⚠ есть в .env" in block
        assert "Перезапустите hermes, чтобы подхватить ключи из .env." in block

    def test_never_prints_full_secret(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cfat_DO_NOT_LEAK_THIS_TAIL")
        block = kb.setup_core.format_keys_block(self._cfg(), include_youtube=True)
        assert "DO_NOT_LEAK_THIS_TAIL" not in block

    def test_optional_keys_labelled(self, kb, monkeypatch):
        _clear_all(monkeypatch)
        block = kb.setup_core.format_keys_block(self._cfg(), include_youtube=True)
        # rerank/youtube keys carry the ", опц." marker
        assert "опц." in block
        assert "SCRAPECREATORS_API_KEY" in block


# ===========================================================================
# done_message now names the .env path (feature: "show where keys went")
# ===========================================================================


class TestDoneMessageShowsPath:
    def test_done_message_includes_env_path(self, kb):
        done = kb.setup_core.done_message()
        assert str(kb.setup_core.env_file_path()) in done
        assert "Ключи записаны в" in done
