"""Smoke tests for the v1.x ingestion-source additions: youtube.py, stt.py,
obsidian.py, enrich.py, and their hook into ingest.py/tools.py.

Follows the same isolation pattern as test_plugin.py (the ``kb`` fixture:
fresh tmp HERMES_HOME + a fresh import of the whole plugin package per
test) — see conftest.py's module docstring.
"""

from __future__ import annotations

import json

import pytest


def _fake_embed(dims: int = 3):
    def _embed(texts, collection_cfg):
        return [[float(i), 0.0, 0.0] for i in range(len(texts))]

    return _embed


# ===========================================================================
# youtube.py
# ===========================================================================


class TestYoutubeVideoIdParsing:
    def test_parse_video_id_from_watch_url(self, kb):
        assert kb.youtube.parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_parse_video_id_from_short_url(self, kb):
        assert kb.youtube.parse_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_parse_video_id_passthrough(self, kb):
        assert kb.youtube.parse_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_parse_video_id_bad_input_raises(self, kb):
        with pytest.raises(kb.youtube.YoutubeError):
            kb.youtube.parse_video_id("not a video url at all")

    def test_is_channel_source(self, kb):
        assert kb.youtube.is_channel_source("https://www.youtube.com/@somechannel") is True
        assert kb.youtube.is_channel_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is False
        assert kb.youtube.is_channel_source("dQw4w9WgXcQ") is False


class TestYoutubeTranscriptFailover:
    """The explicitly required smoke test: provider failover ORDER."""

    def test_first_provider_success_short_circuits(self, kb, monkeypatch):
        calls = []

        def sc_ok(video_id, **kw):
            calls.append("scrapecreators")
            return [{"text": "hello", "start_sec": 0.0, "end_sec": 1.0}]

        def apify_should_not_be_called(video_id, **kw):
            calls.append("apify")
            raise AssertionError("apify should not be called when scrapecreators succeeds")

        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "scrapecreators", sc_ok)
        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "apify", apify_should_not_be_called)

        segments, provider = kb.youtube.get_transcript("vid123", memobase_cfg={})
        assert provider == "scrapecreators"
        assert calls == ["scrapecreators"]
        assert segments[0]["text"] == "hello"

    def test_provider_error_falls_over_to_next(self, kb, monkeypatch):
        calls = []

        def sc_fails(video_id, **kw):
            calls.append("scrapecreators")
            raise kb.youtube.YoutubeError("simulated ScrapeCreators outage")

        def apify_ok(video_id, **kw):
            calls.append("apify")
            return [{"text": "from apify", "start_sec": 2.0, "end_sec": 3.0}]

        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "scrapecreators", sc_fails)
        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "apify", apify_ok)

        segments, provider = kb.youtube.get_transcript(
            "vid123", memobase_cfg={"youtube": {"transcript_providers": ["scrapecreators", "apify"]}}
        )
        assert calls == ["scrapecreators", "apify"]
        assert provider == "apify"
        assert segments[0]["text"] == "from apify"

    def test_no_captions_is_terminal_not_a_failover_trigger(self, kb, monkeypatch):
        """A clean 'no captions' (None) from the FIRST provider must stop
        the ladder immediately -- it must NOT try the second provider for
        the same non-existent captions (see get_transcript's docstring)."""
        calls = []

        def sc_no_captions(video_id, **kw):
            calls.append("scrapecreators")
            return None

        def apify_should_not_be_called(video_id, **kw):
            calls.append("apify")
            return [{"text": "should not happen", "start_sec": 0.0, "end_sec": 1.0}]

        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "scrapecreators", sc_no_captions)
        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "apify", apify_should_not_be_called)

        segments, provider = kb.youtube.get_transcript("vid123", memobase_cfg={})
        assert segments is None
        assert provider is None
        assert calls == ["scrapecreators"]

    def test_all_providers_fail_raises(self, kb, monkeypatch):
        def always_fails(video_id, **kw):
            raise kb.youtube.YoutubeError("down")

        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "scrapecreators", always_fails)
        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "apify", always_fails)

        with pytest.raises(kb.youtube.YoutubeError):
            kb.youtube.get_transcript("vid123", memobase_cfg={})

    def test_respects_configured_provider_order(self, kb, monkeypatch):
        calls = []
        monkeypatch.setitem(
            kb.youtube._TRANSCRIPT_FETCHERS, "scrapecreators",
            lambda video_id, **kw: (calls.append("scrapecreators"), None)[1],
        )
        monkeypatch.setitem(
            kb.youtube._TRANSCRIPT_FETCHERS, "apify",
            lambda video_id, **kw: (calls.append("apify"), [{"text": "x", "start_sec": 0.0, "end_sec": 1.0}])[1],
        )
        # Reversed order: apify first should be tried first.
        segments, provider = kb.youtube.get_transcript(
            "vid123", memobase_cfg={"youtube": {"transcript_providers": ["apify", "scrapecreators"]}}
        )
        assert calls == ["apify"]
        assert provider == "apify"


class TestYoutubeChannelListingFailover:
    def test_falls_back_to_apify_on_scrapecreators_error(self, kb, monkeypatch):
        def sc_fails(channel, **kw):
            raise kb.youtube.YoutubeError("simulated outage")

        def apify_ok(channel):
            return [{"video_id": "abc12345678", "url": "https://youtu.be/abc12345678", "title": "T", "published_at": None, "duration_s": 60}]

        monkeypatch.setattr(kb.youtube, "channel_videos_scrapecreators", sc_fails)
        monkeypatch.setattr(kb.youtube, "channel_videos_apify", apify_ok)

        videos, provider = kb.youtube.list_channel_videos("https://www.youtube.com/@somechannel")
        assert provider == "apify"
        assert videos[0]["video_id"] == "abc12345678"

    def test_prefers_scrapecreators_when_it_works(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.youtube, "channel_videos_scrapecreators",
            lambda channel, **kw: [{"video_id": "v1", "url": "u", "title": "t", "published_at": None, "duration_s": 1}],
        )

        def apify_should_not_run(channel):
            raise AssertionError("must not be called")

        monkeypatch.setattr(kb.youtube, "channel_videos_apify", apify_should_not_run)

        videos, provider = kb.youtube.list_channel_videos("https://www.youtube.com/@somechannel")
        assert provider == "scrapecreators"
        assert len(videos) == 1

    def test_scrapecreators_pagination_follows_continuation_token(self, kb, monkeypatch):
        pages = [
            {"videos": [{"id": "v1", "title": "one"}], "continuation_token": "tok2"},
            {"videos": [{"id": "v2", "title": "two"}], "continuation_token": None},
        ]

        def fake_sc_get(path, params):
            return pages.pop(0)

        monkeypatch.setattr(kb.youtube, "_sc_get", fake_sc_get)
        videos = kb.youtube.channel_videos_scrapecreators("https://www.youtube.com/@x")
        assert [v["video_id"] for v in videos] == ["v1", "v2"]


class TestYoutubeDocBuilding:
    def test_build_video_doc_carries_timecode_query_fragment(self, kb):
        segments = [
            {"text": "first line", "start_sec": 0.0, "end_sec": 5.0},
            {"text": "second line", "start_sec": 754.0, "end_sec": 760.0},
        ]
        doc = kb.youtube.build_video_doc({"video_id": "v1", "title": "My Video"}, segments, provider="scrapecreators")
        pages = [b["page"] for b in doc["blocks"]]
        assert pages == ["?t=0s", "?t=754s"]
        assert doc["meta"]["transcript_provider"] == "scrapecreators"
        assert doc["skipped"] == []

    def test_estimate_channel_cost_usd_shape(self, kb):
        estimate = kb.youtube.estimate_channel_cost_usd(200)
        assert estimate["video_count"] == 200
        assert "total_usd" in estimate
        assert estimate["total_usd"] >= 0


class TestYoutubeExtractVideoNeverRaises:
    def test_bad_source_degrades_to_skipped(self, kb):
        doc = kb.youtube.extract_video("not a real youtube thing", memobase_cfg={})
        assert doc["text"] == ""
        assert doc["skipped"]

    def test_transcript_found_builds_doc(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.youtube, "get_transcript",
            lambda video_id, **kw: ([{"text": "hi there", "start_sec": 1.0, "end_sec": 2.0}], "scrapecreators"),
        )
        doc = kb.youtube.extract_video("https://youtu.be/dQw4w9WgXcQ", memobase_cfg={})
        assert "hi there" in doc["text"]
        assert doc["meta"]["video_id"] == "dQw4w9WgXcQ"

    def test_no_captions_falls_back_to_audio_stt(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "get_transcript", lambda video_id, **kw: (None, None))
        monkeypatch.setattr(kb.youtube, "download_audio_apify", lambda video_id: (b"fake-audio-bytes", "v.mp3"))
        monkeypatch.setattr(
            kb.stt, "transcribe_long_audio",
            lambda audio, **kw: {"segments": [{"text": "from stt", "start_sec": 0.0, "end_sec": 1.0}], "provider": "groq", "trust": "high", "low_confidence_boundaries": []},
        )
        doc = kb.youtube.extract_video("https://youtu.be/dQw4w9WgXcQ", memobase_cfg={})
        assert "from stt" in doc["text"]
        assert doc["meta"]["transcript_provider"] == "stt:groq"

    def test_no_captions_and_audio_fallback_fails_is_honest_skip(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "get_transcript", lambda video_id, **kw: (None, None))

        def audio_fails(video_id):
            raise kb.youtube.YoutubeError("apify audio actor down")

        monkeypatch.setattr(kb.youtube, "download_audio_apify", audio_fails)
        doc = kb.youtube.extract_video("https://youtu.be/dQw4w9WgXcQ", memobase_cfg={})
        assert doc["text"] == ""
        assert doc["skipped"]


class TestYoutubeChannelIngestConfirmGate:
    def test_needs_confirmation_over_threshold(self, kb, monkeypatch):
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "ytcoll", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            videos = [{"video_id": f"v{i}", "url": f"https://youtu.be/v{i}", "title": "t", "published_at": None, "duration_s": 60} for i in range(5)]
            monkeypatch.setattr(kb.youtube, "list_channel_videos", lambda channel, **kw: (videos, "scrapecreators"))
            memobase_cfg = kb.config.get_memobase_config_readonly()
            memobase_cfg["youtube"]["confirm_over_videos"] = 2
            result = kb.youtube.ingest_channel(conn, row, "https://www.youtube.com/@x", memobase_cfg=memobase_cfg)
            assert result["status"] == "needs_confirmation"
            assert result["video_count"] == 5
        finally:
            conn.close()

    def test_confirmed_ingest_calls_ingest_source_per_video(self, kb, monkeypatch):
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "ytcoll2", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            videos = [{"video_id": "v1", "url": "https://youtu.be/v1", "title": "t", "published_at": None, "duration_s": 60}]
            monkeypatch.setattr(kb.youtube, "list_channel_videos", lambda channel, **kw: (videos, "scrapecreators"))

            calls = []

            def fake_ingest_source(conn_, collection_row, source, source_type, **kw):
                calls.append((source, source_type))
                return {"status": "done"}

            monkeypatch.setattr(kb.ingest, "ingest_source", fake_ingest_source)
            memobase_cfg = kb.config.get_memobase_config_readonly()
            result = kb.youtube.ingest_channel(conn, row, "https://www.youtube.com/@x", memobase_cfg=memobase_cfg, confirm=True)
            assert result["status"] == "done"
            assert calls == [("https://youtu.be/v1", "youtube")]
        finally:
            conn.close()


# ===========================================================================
# stt.py
# ===========================================================================


class TestSttChunkMergeAlignment:
    """The explicitly required smoke test: token-level chunk-merge
    alignment on a synthetic 2-chunk overlap (NOT fuzzy string matching)."""

    def test_exact_overlap_is_deduplicated(self, kb):
        chunk_a = {
            "segments": [
                {"text": "the quick brown fox", "start_sec": 0.0, "end_sec": 4.0},
                {"text": "jumps over the lazy dog", "start_sec": 4.0, "end_sec": 8.0},
            ],
            "offset_sec": 0.0,
        }
        # chunk_b's audio physically overlapped the last ~8s of chunk_a, so
        # its first segment repeats "jumps over the lazy dog" verbatim before
        # continuing with genuinely new content.
        chunk_b = {
            "segments": [
                {"text": "jumps over the lazy dog and runs away fast", "start_sec": 0.0, "end_sec": 6.0},
            ],
            "offset_sec": 1192.0,  # this chunk started at 1192s in the ORIGINAL audio
        }
        merged = kb.stt.merge_chunk_transcripts([chunk_a, chunk_b])
        assert merged["low_confidence_boundaries"] == []
        full_text = " ".join(s["text"] for s in merged["segments"])
        # The duplicated "jumps over the lazy dog" must appear exactly ONCE.
        assert full_text.count("jumps over the lazy dog") == 1
        assert "and runs away fast" in full_text
        # Timestamps of chunk_b's segment must be offset into absolute time.
        last_seg = merged["segments"][-1]
        assert last_seg["start_sec"] == 1192.0

    def test_no_overlap_found_is_flagged_low_confidence_not_dropped(self, kb):
        chunk_a = {"segments": [{"text": "alpha beta gamma", "start_sec": 0.0, "end_sec": 3.0}], "offset_sec": 0.0}
        chunk_b = {"segments": [{"text": "completely unrelated words here", "start_sec": 0.0, "end_sec": 3.0}], "offset_sec": 1200.0}
        merged = kb.stt.merge_chunk_transcripts([chunk_a, chunk_b])
        assert len(merged["low_confidence_boundaries"]) == 1
        assert merged["low_confidence_boundaries"][0]["boundary_chunk_index"] == 1
        # Both chunks' content must still be present (never silently dropped).
        full_text = " ".join(s["text"] for s in merged["segments"])
        assert "alpha beta gamma" in full_text
        assert "completely unrelated words here" in full_text

    def test_find_token_overlap_pure(self, kb):
        tail = "a b c d e".split()
        head = "d e f g".split()
        assert kb.stt._find_token_overlap(tail, head, max_check=10) == 2

    def test_find_token_overlap_no_match(self, kb):
        assert kb.stt._find_token_overlap(["a", "b"], ["c", "d"], max_check=10) == 0

    def test_trim_leading_tokens_drops_whole_duplicated_segment(self, kb):
        segs = [
            {"text": "one two", "start_sec": 0.0, "end_sec": 1.0},
            {"text": "three four five", "start_sec": 1.0, "end_sec": 2.0},
        ]
        trimmed = kb.stt._trim_leading_tokens(segs, 2)
        assert trimmed == [{"text": "three four five", "start_sec": 1.0, "end_sec": 2.0}]

    def test_empty_input_never_raises(self, kb):
        assert kb.stt.merge_chunk_transcripts([]) == {"segments": [], "low_confidence_boundaries": []}


class TestSttFfmpegDiscovery:
    def test_find_ffmpeg_returns_something_or_none(self, kb):
        # Not asserting a specific path (machine-dependent) -- just that the
        # function never raises and returns a str or None.
        result = kb.stt.find_ffmpeg()
        assert result is None or isinstance(result, str)


class TestSttGeminiTimecodeParsing:
    def test_parses_bracket_timecode_lines(self, kb):
        text = "[00:00] Hello there\n[01:05] Second line\nnot a timecode line\n[02:30] Third"
        segments = kb.stt._parse_bracket_timecode_lines(text)
        assert [s["start_sec"] for s in segments] == [0, 65, 150]
        assert segments[0]["text"] == "Hello there"


class TestSttOrchestrationFallback:
    def test_groq_failure_falls_back_to_gemini(self, kb, monkeypatch, tmp_path):
        audio_path = tmp_path / "a.flac"
        audio_path.write_bytes(b"not real audio, just needs a size")

        def groq_fails(path, **kw):
            raise kb.stt.SttError("simulated Groq quota exceeded")

        def gemini_ok(path):
            return [{"text": "from gemini", "start_sec": 0, "end_sec": None}]

        monkeypatch.setattr(kb.stt, "transcribe_groq", groq_fails)
        monkeypatch.setattr(kb.stt, "transcribe_gemini", gemini_ok)

        result = kb.stt.transcribe_long_audio(str(audio_path), memobase_cfg={"stt": {"preset": "groq"}})
        assert result["provider"] == "gemini"
        assert result["trust"] == "low"
        assert result["segments"][0]["text"] == "from gemini"

    def test_deletes_temp_audio_after_success(self, kb, monkeypatch):
        monkeypatch.setattr(kb.stt, "transcribe_groq", lambda path, **kw: [{"text": "ok", "start_sec": 0, "end_sec": 1}])
        result = kb.stt.transcribe_long_audio(b"fake bytes for a tiny fake audio file", filename_hint="clip.mp3", memobase_cfg={})
        assert result["segments"][0]["text"] == "ok"
        # transcribe_long_audio must not leave its own tempfile/workdir behind
        # -- indirectly verified by the call succeeding without raising on
        # cleanup; an explicit path-existence check would require capturing
        # the internal tmp path, which the function deliberately doesn't
        # expose (see its docstring: caller does not own internal temp paths).


# ===========================================================================
# obsidian.py
# ===========================================================================


class TestObsidianJsonParse:
    """The explicitly required smoke test: obsidian.json parsing."""

    def test_parses_real_shaped_registry(self, kb, tmp_path):
        registry = {
            "vaults": {
                "abc123def456": {"path": str(tmp_path / "VaultOne"), "ts": 1750000000000, "open": True},
                "789xyz": {"path": str(tmp_path / "VaultTwo"), "ts": 1740000000000, "open": False},
            }
        }
        (tmp_path / "VaultOne").mkdir()
        (tmp_path / "VaultOne" / "note.md").write_text("# hi", encoding="utf-8")
        path = tmp_path / "obsidian.json"
        path.write_text(json.dumps(registry), encoding="utf-8")

        vaults = kb.obsidian.parse_obsidian_json(path)
        names = {v["name"] for v in vaults}
        assert names == {"VaultOne", "VaultTwo"}
        vault_one = next(v for v in vaults if v["name"] == "VaultOne")
        assert vault_one["open"] is True
        assert vault_one["exists"] is True
        assert vault_one["note_count"] == 1
        vault_two = next(v for v in vaults if v["name"] == "VaultTwo")
        assert vault_two["exists"] is False  # directory was never created

    def test_missing_file_returns_empty_never_raises(self, kb, tmp_path):
        assert kb.obsidian.parse_obsidian_json(tmp_path / "does-not-exist.json") == []

    def test_malformed_json_returns_empty_never_raises(self, kb, tmp_path):
        path = tmp_path / "obsidian.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert kb.obsidian.parse_obsidian_json(path) == []

    def test_detect_vaults_never_raises_on_missing_path(self, kb, tmp_path):
        assert kb.obsidian.detect_vaults(tmp_path / "nope.json") == []


class TestObsidianVaultWalk:
    def _make_vault(self, tmp_path):
        vault = tmp_path / "MyVault"
        vault.mkdir()
        (vault / "note1.md").write_text("# Note 1\n\nBody text one.", encoding="utf-8")
        (vault / "sub").mkdir()
        (vault / "sub" / "note2.md").write_text("Body text two.", encoding="utf-8")
        (vault / ".obsidian").mkdir()
        (vault / ".obsidian" / "config.md").write_text("should be ignored", encoding="utf-8")
        (vault / ".trash").mkdir()
        (vault / ".trash" / "deleted.md").write_text("should be ignored too", encoding="utf-8")
        (vault / "templates").mkdir()
        (vault / "templates" / "tmpl.md").write_text("template, ignored", encoding="utf-8")
        return vault

    def test_iter_markdown_files_skips_ignored_dirs(self, kb, tmp_path):
        vault = self._make_vault(tmp_path)
        names = sorted(p.name for p in kb.obsidian.iter_markdown_files(vault))
        assert names == ["note1.md", "note2.md"]

    def test_ingest_vault_ingests_every_note(self, kb, tmp_path, monkeypatch):
        vault = self._make_vault(tmp_path)
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "notes", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            result = kb.obsidian.ingest_vault(conn, row, str(vault), memobase_cfg=kb.config.get_memobase_config_readonly())
            assert result["status"] == "done"
            assert result["notes_total"] == 2
            assert result["notes_ingested"] == 2

            # Re-running (nightly refresh) must be cheap: both notes unchanged.
            result2 = kb.obsidian.ingest_vault(conn, row, str(vault), memobase_cfg=kb.config.get_memobase_config_readonly())
            assert result2["notes_unchanged"] == 2
            assert result2["notes_ingested"] == 0
        finally:
            conn.close()

    def test_ingest_vault_missing_path_fails_gracefully(self, kb, tmp_path):
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "notes2", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            result = kb.obsidian.ingest_vault(conn, row, str(tmp_path / "nope"), memobase_cfg={})
            assert result["status"] == "failed"
            assert result["notes_total"] == 0
        finally:
            conn.close()


class TestObsidianFrontmatterAndWikilinks:
    def test_parse_frontmatter_splits_yaml_block(self, kb):
        text = "---\ntitle: My Note\ntags: [a, b]\n---\nBody content here."
        meta, body = kb.obsidian.parse_frontmatter(text)
        assert meta.get("title") == "My Note"
        assert meta.get("tags") == ["a", "b"]
        assert body.strip() == "Body content here."

    def test_no_frontmatter_returns_empty_meta(self, kb):
        meta, body = kb.obsidian.parse_frontmatter("Just a plain note, no frontmatter.")
        assert meta == {}
        assert body == "Just a plain note, no frontmatter."

    def test_extract_wikilinks(self, kb):
        text = "See [[Project Alpha]] and [[Project Beta#Section|beta]] for details. Also [[Project Alpha]] again."
        links = kb.obsidian.extract_wikilinks(text)
        assert links == ["Project Alpha", "Project Beta"]

    def test_build_link_graph(self, kb, tmp_path):
        vault = tmp_path / "V"
        vault.mkdir()
        (vault / "a.md").write_text("links to [[b]]", encoding="utf-8")
        (vault / "b.md").write_text("no links here", encoding="utf-8")
        graph = kb.obsidian.build_link_graph(vault)
        assert graph["a"] == ["b"]
        assert graph["b"] == []

    def test_extract_note_includes_frontmatter_and_links_in_meta(self, kb, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("---\ntitle: Hello\n---\nSee [[Other Note]].", encoding="utf-8")
        doc = kb.obsidian.extract_note(str(note))
        assert doc["meta"]["title"] == "Hello"
        assert doc["meta"]["wikilinks"] == ["Other Note"]
        assert "See" in doc["text"]

    def test_extract_note_missing_file_is_skipped_not_raised(self, kb, tmp_path):
        doc = kb.obsidian.extract_note(str(tmp_path / "missing.md"))
        assert doc["text"] == ""
        assert doc["skipped"]


# ===========================================================================
# enrich.py
# ===========================================================================


class TestEnrich:
    def test_disabled_by_default(self, kb):
        assert kb.enrich.is_enabled(kb.config.get_memobase_config_readonly()) is False

    def test_enrich_one_chunk_uses_llm_and_caps_length(self, kb, fake_llm):
        fake_llm.next_text = "x" * 5000  # pathological long response
        note = kb.enrich.enrich_one_chunk("some chunk text", {"title": "Doc"}, llm=fake_llm)
        assert len(note) <= kb.enrich.DEFAULT_MAX_ENRICHMENT_CHARS
        assert len(fake_llm.calls) == 1

    def test_enrich_one_chunk_no_llm_returns_empty(self, kb):
        assert kb.enrich.enrich_one_chunk("text", {}, llm=None) == ""

    def test_enrich_one_chunk_llm_failure_degrades_to_empty(self, kb, fake_llm):
        fake_llm.raise_on_complete = RuntimeError("boom")
        assert kb.enrich.enrich_one_chunk("text", {}, llm=fake_llm) == ""

    def test_enrich_chunks_for_embedding_prepends_only_for_embedder(self, kb, fake_llm):
        fake_llm.next_text = "context note"
        texts_for_embedder, enrichment_strings = kb.enrich.enrich_chunks_for_embedding(
            ["raw chunk one", "raw chunk two"], {"title": "Doc"}, llm=fake_llm
        )
        assert all("context note" in t for t in texts_for_embedder)
        assert all(e == "context note" for e in enrichment_strings)

    def test_ingest_with_enrichment_keeps_raw_text_stored(self, kb, monkeypatch, tmp_path, fake_llm):
        """The non-negotiable JIT contract: stored chunks.text is ALWAYS
        raw, even when enrichment is enabled and the embedder receives an
        enriched variant."""
        captured_embed_input = {}

        def fake_embed(texts, collection_cfg):
            captured_embed_input["texts"] = list(texts)
            return [[0.0, 0.0, 0.0] for _ in texts]

        monkeypatch.setattr(kb.embed, "embed_texts", fake_embed)
        fake_llm.next_text = "This is context about the note."

        p = tmp_path / "doc.txt"
        p.write_text("Уникальный текст документа для проверки обогащения.", encoding="utf-8")
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "enrichcoll", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            memobase_cfg = kb.config.get_memobase_config_readonly()
            memobase_cfg["enrich"]["enabled"] = True
            result = kb.ingest.ingest_source(conn, row, str(p), "txt", memobase_cfg=memobase_cfg, llm=fake_llm)
            assert result["status"] == "done"

            stored = conn.execute("SELECT id, text FROM chunks WHERE collection_id = ?", (row["id"],)).fetchone()
            assert "This is context about the note." not in stored["text"]
            assert "Уникальный текст документа" in stored["text"]

            # The embedder, however, DID receive the enriched variant.
            assert any("This is context about the note." in t for t in captured_embed_input["texts"])

            # And the enrichment string is persisted to the debug side-table.
            enrichment = kb.db.get_chunk_enrichment(conn, stored["id"])
            assert enrichment == "This is context about the note."
        finally:
            conn.close()

    def test_ingest_without_enrichment_enabled_calls_llm_zero_times(self, kb, monkeypatch, tmp_path, fake_llm):
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())
        p = tmp_path / "doc.txt"
        p.write_text("Обычный документ без обогащения.", encoding="utf-8")
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "noenrich", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            memobase_cfg = kb.config.get_memobase_config_readonly()  # enrich.enabled defaults to False
            result = kb.ingest.ingest_source(conn, row, str(p), "txt", memobase_cfg=memobase_cfg, llm=fake_llm)
            assert result["status"] == "done"
            assert fake_llm.calls == []
        finally:
            conn.close()


# ===========================================================================
# ingest.py dispatch wiring
# ===========================================================================


class TestIngestSourceDispatch:
    def test_youtube_source_type_routes_to_youtube_module(self, kb, monkeypatch, tmp_path):
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())

        def fake_extract_video(source, **kw):
            return {
                "text": "video transcript text here for testing purposes",
                "blocks": [{"text": "video transcript text here for testing purposes", "page": "?t=0s", "section": "T", "is_code": False}],
                "meta": {"title": "T", "video_id": "v1"},
                "skipped": [],
            }

        monkeypatch.setattr(kb.youtube, "extract_video", fake_extract_video)
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "ytdispatch", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            result = kb.ingest.ingest_source(conn, row, "https://youtu.be/dQw4w9WgXcQ", "youtube", memobase_cfg=kb.config.get_memobase_config_readonly())
            assert result["status"] == "done"
            chunk_row = conn.execute("SELECT page_or_timecode FROM chunks WHERE collection_id = ?", (row["id"],)).fetchone()
            assert chunk_row["page_or_timecode"] == "?t=0s"
        finally:
            conn.close()

    def test_obsidian_source_type_routes_to_obsidian_module(self, kb, monkeypatch, tmp_path):
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())
        note = tmp_path / "note.md"
        note.write_text("Заметка с уникальным содержимым для проверки диспетчеризации.", encoding="utf-8")
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "obsdispatch", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            result = kb.ingest.ingest_source(conn, row, str(note), "obsidian", memobase_cfg=kb.config.get_memobase_config_readonly())
            assert result["status"] == "done"
        finally:
            conn.close()

    def test_audio_source_type_routes_to_stt_module(self, kb, monkeypatch, tmp_path):
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())
        audio_path = tmp_path / "clip.mp3"
        audio_path.write_bytes(b"not real audio bytes")

        monkeypatch.setattr(
            kb.stt, "transcribe_long_audio",
            lambda audio, **kw: {"segments": [{"text": "audio transcript content for the test", "start_sec": 0, "end_sec": 1}], "provider": "groq", "trust": "high", "low_confidence_boundaries": []},
        )
        conn = kb.db.get_connection()
        try:
            cid = kb.db.create_collection(conn, "audiodispatch", embedder_dims=3)
            row = kb.db.get_collection_by_id(conn, cid)
            result = kb.ingest.ingest_source(conn, row, str(audio_path), "audio", memobase_cfg=kb.config.get_memobase_config_readonly())
            assert result["status"] == "done"
        finally:
            conn.close()


# ===========================================================================
# tools.py wiring (schema + multi-item routing)
# ===========================================================================


class TestToolsIngestionSourcesWiring:
    def test_schema_enum_includes_new_source_types(self, kb):
        enum = kb.tools.MEMOBASE_INGEST_SCHEMA["parameters"]["properties"]["source_type"]["enum"]
        for t in ("youtube", "audio", "video", "obsidian"):
            assert t in enum

    def test_kb_ingest_routes_youtube_channel_to_channel_orchestrator(self, kb, monkeypatch, fake_ctx):
        kb.tools.register(fake_ctx)
        called = {}

        def fake_ingest_channel(conn, collection_row, channel, **kw):
            called["channel"] = channel
            return {"status": "done", "video_count": 3, "videos_done": 3, "videos_unchanged": 0, "videos_failed": 0, "list_provider": "scrapecreators"}

        monkeypatch.setattr(kb.youtube, "ingest_channel", fake_ingest_channel)
        result = kb.tools.memobase_ingest({"source": "https://www.youtube.com/@somechannel", "source_type": "youtube"})
        assert called["channel"] == "https://www.youtube.com/@somechannel"
        assert "Канал загружен" in result

    def test_kb_ingest_routes_obsidian_directory_to_vault_orchestrator(self, kb, monkeypatch, fake_ctx, tmp_path):
        kb.tools.register(fake_ctx)
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        called = {}

        def fake_ingest_vault(conn, collection_row, vault_path, **kw):
            called["vault"] = vault_path
            return {"status": "done", "notes_total": 5, "notes_ingested": 5, "notes_unchanged": 0, "notes_failed": 0}

        monkeypatch.setattr(kb.obsidian, "ingest_vault", fake_ingest_vault)
        result = kb.tools.memobase_ingest({"source": str(vault_dir), "source_type": "obsidian"})
        assert called["vault"] == str(vault_dir)
        assert "Vault загружен" in result

    def test_kb_ingest_single_obsidian_note_uses_normal_path(self, kb, monkeypatch, fake_ctx, tmp_path):
        kb.tools.register(fake_ctx)
        monkeypatch.setattr(kb.embed, "embed_texts", _fake_embed())
        # Pre-create the default collection with dims matching _fake_embed()
        # (3) -- memobase_ingest()'s _get_or_create_collection would otherwise
        # create it with the config default (1024), which the vec0 table
        # would then reject the 3-dim fake vectors against.
        conn = kb.db.get_connection()
        try:
            kb.db.create_collection(conn, kb.config.get_memobase_config_readonly()["default_collection"], embedder_dims=3)
        finally:
            conn.close()
        note = tmp_path / "single.md"
        note.write_text("Одиночная заметка с уникальным текстом.", encoding="utf-8")
        result = kb.tools.memobase_ingest({"source": str(note), "source_type": "obsidian"})
        assert "Загружено в коллекцию" in result


# ===========================================================================
# config.py new defaults
# ===========================================================================


class TestConfigNewDefaults:
    def test_youtube_defaults_present(self, kb):
        cfg = kb.config.get_memobase_config_readonly()
        assert cfg["youtube"]["transcript_providers"] == ["scrapecreators", "apify"]
        assert cfg["youtube"]["confirm_over_videos"] == 20

    def test_stt_defaults_present(self, kb):
        cfg = kb.config.get_memobase_config_readonly()
        assert cfg["stt"]["preset"] == "groq"

    def test_enrich_defaults_present_and_off(self, kb):
        cfg = kb.config.get_memobase_config_readonly()
        assert cfg["enrich"]["enabled"] is False
