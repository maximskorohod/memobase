"""Regression tests for the second audit round (fix/audit-2).

Same isolation pattern as the rest of the suite: the ``kb`` fixture gives a
fresh plugin import with an isolated HERMES_HOME.
"""

from __future__ import annotations

import os
import sys

import pytest


def _fake_embed():
    def _embed(texts, cfg):
        dims = (cfg.get("embedder") or {}).get("dims") or 3
        return [[0.1] * dims for _ in texts]
    return _embed


def _guest_collection(kb, name="guest_coll", owner="guest-x"):
    conn = kb.db.get_connection()
    try:
        cid = kb.db.create_collection(conn, name, owner_user_id=owner, embedder_dims=3, visibility="private")
    finally:
        conn.close()
    return cid


# ===========================================================================
# 1. Guest local-path ingest allowlist / obsidian hardening
# ===========================================================================


class TestGuestIngestPaths:
    def test_guest_cannot_ingest_arbitrary_local_path(self, kb, tmp_path, monkeypatch):
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())
        _guest_collection(kb)
        secret = tmp_path / "outside.txt"
        secret.write_text("host secret file content", encoding="utf-8")
        kb.tools.bind_session_user("sess-x", "guest-x")
        out = kb.tools.memobase_ingest(
            {"source": str(secret), "source_type": "txt", "collection": "guest_coll"}, session_id="sess-x",
        )
        assert "запрещены" in out
        conn = kb.db.get_connection(read_only=True)
        try:
            assert conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 0
        finally:
            conn.close()

    def test_guest_can_ingest_gateway_attachment(self, kb, monkeypatch):
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())
        _guest_collection(kb)
        cache = kb._hermes_home_for_test / "cache" / "documents"
        cache.mkdir(parents=True)
        f = cache / "doc_abc_note.txt"
        f.write_text("Гостевой документ про сад и огород. " * 5, encoding="utf-8")
        kb.tools.bind_session_user("sess-x", "guest-x")
        out = kb.tools.memobase_ingest(
            {"source": str(f), "source_type": "txt", "collection": "guest_coll"}, session_id="sess-x",
        )
        assert "Загружено" in out, out

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_guest_symlink_out_of_attachment_dir_refused(self, kb, tmp_path, monkeypatch):
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())
        _guest_collection(kb)
        target = tmp_path / "secret.txt"
        target.write_text("SECRET", encoding="utf-8")
        cache = kb._hermes_home_for_test / "cache" / "documents"
        cache.mkdir(parents=True)
        link = cache / "innocent.txt"
        os.symlink(target, link)
        kb.tools.bind_session_user("sess-x", "guest-x")
        out = kb.tools.memobase_ingest(
            {"source": str(link), "source_type": "txt", "collection": "guest_coll"}, session_id="sess-x",
        )
        assert "запрещены" in out

    def test_guest_cannot_ingest_obsidian_vault(self, kb, tmp_path):
        _guest_collection(kb)
        vault = kb._hermes_home_for_test / "cache" / "documents" / "vault"
        vault.mkdir(parents=True)
        (vault / "a.md").write_text("# note", encoding="utf-8")
        kb.tools.bind_session_user("sess-x", "guest-x")
        out = kb.tools.memobase_ingest(
            {"source": str(vault), "source_type": "obsidian", "collection": "guest_coll"}, session_id="sess-x",
        )
        assert "Загружено" not in out and "Vault загружен" not in out

    def test_extract_note_rejects_non_md(self, kb, tmp_path):
        f = tmp_path / "x.env"
        f.write_text("API_KEY=1", encoding="utf-8")
        doc = kb.obsidian.extract_note(str(f))
        assert doc["text"] == ""

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_vault_walk_skips_symlink_outside(self, kb, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "ok.md").write_text("ok", encoding="utf-8")
        outside = tmp_path / "outside.md"
        outside.write_text("outside", encoding="utf-8")
        os.symlink(outside, vault / "evil.md")
        names = sorted(p.name for p in kb.obsidian.iter_markdown_files(vault))
        assert names == ["ok.md"]
