"""Tests for setup_core.py — the UI-agnostic onboarding core shared by the
Telegram ``/memobase setup`` wizard (wizard.py) and the terminal ``hermes memobase
setup`` command (cli.py).

Same isolation pattern as the rest of this suite (see conftest.py): the
``kb`` fixture isolates ``HERMES_HOME`` to a fresh tmp dir and returns a
freshly-imported copy of the whole plugin package, so
``kb.setup_core`` is exactly the module wizard.py/cli.py import.

No real network calls anywhere in this file:
  * the dependency check (ffmpeg/pip) is local filesystem/import
    introspection only, and is additionally monkeypatched in the
    present/absent tests for full determinism regardless of what happens to
    be installed on the machine running the suite;
  * the live-probe tests deliberately clear CLOUDFLARE_ACCOUNT_ID/
    CLOUDFLARE_API_TOKEN so ``embed.py``'s own missing-credentials guard
    raises before any HTTP request is ever attempted for the "cloudflare"
    provider; the "openai" and "cohere" cases raise even earlier (missing
    ``base_url`` / unknown provider) regardless of any env var, by
    construction of ``validate_provider_key``'s probe config -- see
    embed.py's ``embed_texts`` dispatch.
"""

from __future__ import annotations

import os

import pytest


# ===========================================================================
# Key format validation + wrong-key-type detection (checklist item (b))
# ===========================================================================


class TestKeyFormatValidation:
    @pytest.mark.parametrize(
        "provider,value",
        [
            ("cloudflare", "cffaketesttoken0123456789abcdef01"),
            ("cohere", "cohfaketesttoken0123456789abcdef01"),
            ("openai", "sk-fakeTestToken1234567890"),
        ],
    )
    def test_accepts_well_shaped_key(self, kb, provider, value):
        ok, hint = kb.setup_core.validate_key_format(provider, value)
        assert ok is True
        assert hint == ""

    def test_rejects_too_short_cloudflare_token(self, kb):
        ok, hint = kb.setup_core.validate_key_format("cloudflare", "short")
        assert ok is False
        assert "формат" in hint.lower() or "Cloudflare" in hint

    def test_rejects_empty_value(self, kb):
        ok, hint = kb.setup_core.validate_key_format("cloudflare", "   ")
        assert ok is False
        assert "пуст" in hint.lower()

    def test_openai_requires_sk_prefix(self, kb):
        ok, hint = kb.setup_core.validate_key_format("openai", "not-the-right-shape-1234567890")
        assert ok is False

    def test_unknown_provider_is_accepted_as_is(self, kb):
        ok, hint = kb.setup_core.validate_key_format("mystery-provider", "anything at all")
        assert ok is True
        assert hint == ""

    # --- wrong-key-type re-ask (the actual "detects wrong-key-type" gap) ---

    def test_openai_shaped_key_rejected_when_asked_for_cloudflare(self, kb):
        ok, hint = kb.setup_core.validate_key_format("cloudflare", "sk-thisIsAnOpenAiShapedKey123456")
        assert ok is False
        assert "OpenAI" in hint

    def test_groq_shaped_key_rejected_when_asked_for_cohere(self, kb):
        ok, hint = kb.setup_core.validate_key_format("cohere", "gsk_thisIsAGroqShapedToken1234567890")
        assert ok is False
        assert "Groq" in hint

    def test_gemini_shaped_key_rejected_when_asked_for_openai(self, kb):
        ok, hint = kb.setup_core.validate_key_format("openai", "AIzaSyFAKEGEMINIKEY0123456789abcdefghi")
        assert ok is False
        assert "Gemini" in hint or "Google" in hint

    def test_apify_shaped_token_rejected_when_asked_for_cloudflare(self, kb):
        ok, hint = kb.setup_core.validate_key_format("cloudflare", "apify_api_FAKE1234567890abcdef")
        assert ok is False
        assert "Apify" in hint

    def test_own_shape_is_not_flagged_as_mismatch(self, kb):
        # openai's OWN expected shape starts with "sk-" -- must not flag itself.
        mismatch = kb.setup_core.classify_key_mismatch("OPENAI_API_KEY", "sk-thisIsFine1234567890")
        assert mismatch is None

    def test_classify_key_mismatch_ignores_empty_value(self, kb):
        assert kb.setup_core.classify_key_mismatch("CLOUDFLARE_API_TOKEN", "") is None


# ===========================================================================
# Masking + .env upsert (checklist item (d))
# ===========================================================================


class TestMaskSecret:
    def test_mask_is_first_four_chars_plus_ellipsis(self, kb):
        assert kb.setup_core.mask_secret("abcdefgh12345678") == "abcd…"

    def test_mask_of_empty_string(self, kb):
        assert kb.setup_core.mask_secret("") == "…"

    def test_mask_of_short_string(self, kb):
        assert kb.setup_core.mask_secret("ab") == "ab…"

    def test_mask_never_contains_the_tail_of_a_longer_secret(self, kb):
        secret = "FAKESECRETVALUE_not_a_real_key_999"
        masked = kb.setup_core.mask_secret(secret)
        assert secret[4:] not in masked
        assert masked == "FAKE…"


class TestEnvUpsert:
    def _env_path(self, kb):
        return kb._hermes_home_for_test / ".env"

    def test_write_creates_file_with_var(self, kb):
        kb.setup_core.write_env_secret("FAKE_TEST_TOKEN", "FAKEVALUE1234567890")
        content = self._env_path(kb).read_text(encoding="utf-8")
        assert "FAKE_TEST_TOKEN=FAKEVALUE1234567890" in content
        assert os.environ["FAKE_TEST_TOKEN"] == "FAKEVALUE1234567890"

    def test_write_upserts_existing_var_in_place(self, kb, monkeypatch):
        monkeypatch.delenv("FAKE_TEST_TOKEN", raising=False)
        kb.setup_core.write_env_secret("FAKE_TEST_TOKEN", "firstFakeValue")
        kb.setup_core.write_env_secret("FAKE_TEST_TOKEN", "secondFakeValue")
        content = self._env_path(kb).read_text(encoding="utf-8")
        lines = [ln for ln in content.splitlines() if ln.startswith("FAKE_TEST_TOKEN=")]
        assert len(lines) == 1, "upsert must replace, never duplicate, the key's line"
        assert lines[0] == "FAKE_TEST_TOKEN=secondFakeValue"
        assert os.environ["FAKE_TEST_TOKEN"] == "secondFakeValue"

    def test_write_preserves_other_existing_vars(self, kb, monkeypatch):
        monkeypatch.delenv("OTHER_FAKE_VAR", raising=False)
        monkeypatch.delenv("FAKE_TEST_TOKEN", raising=False)
        kb.setup_core.write_env_secret("OTHER_FAKE_VAR", "keepMe")
        kb.setup_core.write_env_secret("FAKE_TEST_TOKEN", "newValue")
        content = self._env_path(kb).read_text(encoding="utf-8")
        assert "OTHER_FAKE_VAR=keepMe" in content
        assert "FAKE_TEST_TOKEN=newValue" in content


# ===========================================================================
# Dependency check -- ffmpeg + pip packages (checklist item (f))
# ===========================================================================


class TestDependencyCheck:
    def test_ffmpeg_present_reported_ok(self, kb, monkeypatch):
        monkeypatch.setattr(kb.setup_core, "detect_ffmpeg", lambda: (True, "C:/fake/ffmpeg.exe"))
        monkeypatch.setattr(kb.setup_core, "_pip_package_present", lambda name: True)
        report = kb.setup_core.check_dependencies()
        assert report["ffmpeg"]["ok"] is True
        assert all(dep["ok"] for dep in report["pip"])
        text = kb.setup_core.format_dependency_report(report)
        assert "ffmpeg" in text and "найден" in text
        assert "на месте" in text

    def test_ffmpeg_absent_reported_with_install_hint(self, kb, monkeypatch):
        monkeypatch.setattr(kb.setup_core, "detect_ffmpeg", lambda: (False, None))
        monkeypatch.setattr(kb.setup_core, "_pip_package_present", lambda name: True)
        monkeypatch.setattr(kb.setup_core.platform, "system", lambda: "Windows")
        report = kb.setup_core.check_dependencies()
        assert report["ffmpeg"]["ok"] is False
        text = kb.setup_core.format_dependency_report(report)
        assert "НЕ найден" in text
        assert "winget" in text.lower()

    def test_ffmpeg_absent_install_hint_is_platform_specific(self, kb, monkeypatch):
        monkeypatch.setattr(kb.setup_core, "detect_ffmpeg", lambda: (False, None))
        monkeypatch.setattr(kb.setup_core.platform, "system", lambda: "Darwin")
        assert "brew" in kb.setup_core._ffmpeg_install_hint().lower()
        monkeypatch.setattr(kb.setup_core.platform, "system", lambda: "Linux")
        assert "apt" in kb.setup_core._ffmpeg_install_hint().lower()

    def test_missing_pip_package_surfaced_with_pip_install_hint(self, kb, monkeypatch):
        monkeypatch.setattr(kb.setup_core, "detect_ffmpeg", lambda: (True, "/usr/bin/ffmpeg"))

        def fake_present(name):
            return name != "mammoth"

        monkeypatch.setattr(kb.setup_core, "_pip_package_present", fake_present)
        report = kb.setup_core.check_dependencies()
        missing = [d["import_name"] for d in report["pip"] if not d["ok"]]
        assert missing == ["mammoth"]
        text = kb.setup_core.format_dependency_report(report)
        assert "mammoth" in text
        assert "pip install mammoth" in text

    def test_detect_ffmpeg_never_raises_and_returns_well_formed_tuple(self, kb):
        # Whatever the real machine running this suite has installed, this
        # must never raise and must always return (bool, str|None).
        found, path = kb.setup_core.detect_ffmpeg()
        assert isinstance(found, bool)
        assert path is None or isinstance(path, str)

    def test_check_dependencies_never_raises_when_stt_import_is_broken(self, kb, monkeypatch):
        # detect_ffmpeg() prefers stt.py's own discovery and falls back to a
        # bare shutil.which() if importing stt.py fails for any reason --
        # force that fallback path and confirm it degrades cleanly, no raise.
        import builtins

        real_import = builtins.__import__

        def broken_import(name, globals=None, locals=None, fromlist=(), level=0):
            if fromlist and "stt" in fromlist:
                raise ImportError("simulated stt import failure")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        found, path = kb.setup_core.detect_ffmpeg()
        assert isinstance(found, bool)
        assert path is None or isinstance(path, str)


# ===========================================================================
# Step sequencing + shared question/content text
# ===========================================================================


class TestStepSequencing:
    def test_steps_order(self, kb):
        assert kb.setup_core.STEPS == (
            "embedder", "embedder_confirm", "cloud_provider", "cloud_key",
            "first_ingest", "control_question", "done",
        )

    def test_cloud_providers_mapping(self, kb):
        assert kb.setup_core.CLOUD_PROVIDERS == {"1": "cloudflare", "2": "cohere", "3": "openai"}

    def test_cloud_key_env_derived_from_catalog(self, kb):
        assert kb.setup_core.CLOUD_KEY_ENV == {
            "cloudflare": "CLOUDFLARE_API_TOKEN",
            "cohere": "COHERE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }

    def test_local_embedder_model_is_e5_large(self, kb):
        # Single supported local model (fastembed, 1024-dim, drop-in for bge-m3).
        assert kb.setup_core.local_embedder_model() == "intfloat/multilingual-e5-large"

    def test_embedder_question_two_options_local_and_cloud(self, kb):
        text = kb.setup_core.embedder_question(16.0)
        # Exactly the two options, no redundant duplicate local line.
        assert "1 —" in text and "2 —" in text and "3 —" not in text
        assert "локально" in text.lower() and "облак" in text.lower()

    def test_embedder_question_mentions_ram_and_recommends_full_variant(self, kb):
        text = kb.setup_core.embedder_question(16.0)
        assert "16" in text
        assert "вариант 1" in text

    def test_embedder_question_recommends_cloud_for_low_ram(self, kb):
        # Local e5-large needs ~2 GB RAM; a low-RAM box is steered to cloud (option 2).
        text = kb.setup_core.embedder_question(4.0)
        assert "вариант 2" in text

    def test_embedder_question_handles_unknown_ram(self, kb):
        text = kb.setup_core.embedder_question(None)
        assert "эмбеддинги" in text.lower() or "где считать" in text.lower()

    def test_local_confirm_question_warns_about_download(self, kb):
        text = kb.setup_core.local_confirm_question()
        assert "2.2" in text and "fastembed" in text.lower()
        assert "да" in text.lower() and "назад" in text.lower()

    def test_affirmative_and_back_parsers(self, kb):
        assert kb.setup_core.is_affirmative("да") and kb.setup_core.is_affirmative("Y")
        assert kb.setup_core.is_back("назад") and kb.setup_core.is_back("back")
        assert not kb.setup_core.is_affirmative("назад")
        assert not kb.setup_core.is_back("да")

    def test_install_local_embedder_success_mocked(self, kb, monkeypatch):
        import subprocess
        import sys
        import types

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
        fake = types.ModuleType("fastembed")
        fake.TextEmbedding = lambda *a, **k: object()
        monkeypatch.setitem(sys.modules, "fastembed", fake)
        ok, msg = kb.setup_core.install_local_embedder()
        assert ok is True and "готов" in msg.lower()

    def test_install_local_embedder_pip_failure_mocked(self, kb, monkeypatch):
        import subprocess

        def _boom(*a, **k):
            raise RuntimeError("pip down")

        monkeypatch.setattr(subprocess, "run", _boom)
        ok, msg = kb.setup_core.install_local_embedder()
        assert ok is False and "install.sh --local" in msg

    def test_cloud_provider_question_lists_all_three(self, kb):
        text = kb.setup_core.cloud_provider_question()
        assert "Cloudflare" in text and "Cohere" in text and "OpenAI" in text

    def test_cloud_key_question_explains_with_metaphor_and_mentions_key(self, kb):
        text = kb.setup_core.cloud_key_question("cloudflare")
        assert "ключ" in text.lower()
        assert "CLOUDFLARE_API_TOKEN" in text
        # advises the user to delete the message themselves (checklist item (e))
        assert "удали" in text.lower()

    def test_cloud_key_question_falls_back_for_unknown_provider(self, kb):
        text = kb.setup_core.cloud_key_question("mystery-provider")
        assert "mystery-provider" in text

    def test_first_ingest_control_and_done_text(self, kb):
        assert "загруз" in kb.setup_core.first_ingest_question().lower()
        cq = kb.setup_core.control_question_question().lower()
        assert "вопрос" in cq or "цитат" in cq
        done = kb.setup_core.done_message().lower()
        assert "заверш" in done or "шпаргалка" in done or "статус" in done


# ===========================================================================
# Live provider probe -- never a real network call in this suite
# ===========================================================================


class TestValidateProviderKeyNoNetwork:
    def test_cloudflare_probe_fails_honestly_without_credentials(self, kb, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        ok, msg = kb.setup_core.validate_provider_key("cloudflare")
        assert ok is False
        assert isinstance(msg, str) and msg

    def test_openai_probe_fails_honestly_missing_base_url_no_network(self, kb):
        # embed_texts requires embedder.base_url for the openai-compat path,
        # which validate_provider_key's probe config never sets -- raises
        # before any HTTP request is attempted, regardless of OPENAI_API_KEY.
        ok, msg = kb.setup_core.validate_provider_key("openai")
        assert ok is False

    def test_cohere_probe_fails_honestly_unknown_embed_provider_no_network(self, kb):
        # Audit finding: embed.py has no "cohere" embedder provider at all
        # (Cohere is only wired up for reranking) -- this call always fails,
        # even with a perfectly valid Cohere key. Documented, not fixed here
        # (see setup_core.validate_provider_key's docstring) -- this test
        # pins that pre-existing, out-of-scope behavior so it isn't silently
        # "fixed" by an unrelated future change without a deliberate decision.
        ok, msg = kb.setup_core.validate_provider_key("cohere")
        assert ok is False

    def test_validate_provider_key_never_raises_for_unknown_provider(self, kb):
        ok, msg = kb.setup_core.validate_provider_key("totally-unknown")
        assert ok is False
        assert isinstance(msg, str) and msg
