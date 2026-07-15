"""Owner-only YouTube tool wrappers (tools.py) + their youtube.py helpers:
list_channel / get_transcript / dl_audio, sort_and_filter_videos, language &
max_pages threading, and the tolerant (live-unverified) Apify transcript
parser. Every external call is monkeypatched at the youtube.py/stt.py boundary
— no network, no API keys."""

from __future__ import annotations

import datetime as dt

import pytest


# --- youtube.py pure helpers ------------------------------------------------


class TestSortAndFilterVideos:
    def _vids(self):
        return [
            {"video_id": "old", "published_at": "2026-01-01T00:00:00Z", "url": "u", "title": "t", "duration_s": 60},
            {"video_id": "new", "published_at": "2026-07-10T00:00:00Z", "url": "u", "title": "t", "duration_s": 60},
            {"video_id": "mid", "published_at": "2026-05-05", "url": "u", "title": "t", "duration_s": 60},
        ]

    def test_sorts_newest_first(self, kb):
        out, _more = kb.youtube.sort_and_filter_videos(self._vids(), limit=10)
        assert [v["video_id"] for v in out] == ["new", "mid", "old"]

    def test_limit_truncates(self, kb):
        out, _more = kb.youtube.sort_and_filter_videos(self._vids(), limit=2)
        assert [v["video_id"] for v in out] == ["new", "mid"]

    def test_since_filters_older(self, kb):
        out, _more = kb.youtube.sort_and_filter_videos(self._vids(), limit=10, since_date=dt.date(2026, 5, 1))
        assert [v["video_id"] for v in out] == ["new", "mid"]

    def test_since_all_in_range_flags_possibly_more(self, kb):
        out, more = kb.youtube.sort_and_filter_videos(self._vids(), limit=10, since_date=dt.date(2020, 1, 1))
        assert len(out) == 3 and more is True

    def test_unparseable_date_sorts_last_and_excluded_by_since(self, kb):
        vids = self._vids() + [{"video_id": "rel", "published_at": "2 days ago", "url": "u", "title": "t", "duration_s": 1}]
        out_all, _ = kb.youtube.sort_and_filter_videos(vids, limit=10)
        assert out_all[-1]["video_id"] == "rel"
        out_since, _ = kb.youtube.sort_and_filter_videos(vids, limit=10, since_date=dt.date(2020, 1, 1))
        assert "rel" not in [v["video_id"] for v in out_since]

    def test_equal_dates_tiebreak_by_video_id(self, kb):
        a = {"video_id": "aaa", "published_at": "2026-07-10T20:11:40.946Z"}
        b = {"video_id": "bbb", "published_at": "2026-07-10T20:11:40.946Z"}
        out1, _ = kb.youtube.sort_and_filter_videos([b, a], limit=10)
        out2, _ = kb.youtube.sort_and_filter_videos([a, b], limit=10)
        assert [v["video_id"] for v in out1] == [v["video_id"] for v in out2] == ["aaa", "bbb"]


class TestParseSinceDate:
    def test_valid(self, kb):
        d, err = kb.youtube.parse_since_date("2026-07-10")
        assert err is None and d == dt.date(2026, 7, 10)

    def test_invalid(self, kb):
        d, err = kb.youtube.parse_since_date("10.07.2026")
        assert d is None and "YYYY-MM-DD" in err


class TestFormatHelpers:
    def test_minutes_seconds(self, kb):
        assert kb.youtube.format_hms(125) == "02:05"

    def test_hours(self, kb):
        assert kb.youtube.format_hms(3725) == "1:02:05"

    def test_duration_unknown(self, kb):
        assert kb.youtube.format_duration(None) == "?"

    def test_duration_number(self, kb):
        assert kb.youtube.format_duration(754) == "12:34"


class TestLanguageThreading:
    def test_get_transcript_passes_language_to_fetcher(self, kb, monkeypatch):
        seen = {}

        def fake_sc(vid, *, language="ru", **_kw):
            seen["language"] = language
            return [{"text": "hi", "start_sec": 0.0, "end_sec": 1.0}]

        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "scrapecreators", fake_sc)
        segs, prov = kb.youtube.get_transcript(
            "vid", memobase_cfg={"youtube": {"transcript_providers": ["scrapecreators"]}}, language="en"
        )
        assert seen["language"] == "en" and prov == "scrapecreators" and segs


class TestListChannelMaxPages:
    def test_max_pages_threaded(self, kb, monkeypatch):
        seen = {}

        def fake_sc(channel, *, max_pages=None):
            seen["max_pages"] = max_pages
            return [{"video_id": "a", "url": "u", "title": "t", "published_at": "2026-07-10", "duration_s": 1}]

        monkeypatch.setattr(kb.youtube, "channel_videos_scrapecreators", fake_sc)
        vids, prov = kb.youtube.list_channel_videos("@x", max_pages=2)
        assert seen["max_pages"] == 2 and prov == "scrapecreators"


class TestTranscriptApifyParser:
    def test_parses_supreme_coder_shape_and_fixes_input_field(self, kb, monkeypatch):
        captured = {}

        def fake_run(actor, run_input, **kw):
            captured["input"] = run_input
            return [{"transcript": [{"text": "hello", "start": 1.5, "duration": 2.0}]}]

        monkeypatch.setattr(kb.youtube, "_apify_run_actor", fake_run)
        segs = kb.youtube.transcript_apify("vid")
        assert "urls" in captured["input"]  # was the buggy "videoUrls"
        assert segs == [{"text": "hello", "start_sec": 1.5, "end_sec": 3.5}]

    def test_parses_end_field_shape(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "_apify_run_actor", lambda *a, **k: [{"segments": [{"text": "yo", "start": 0.0, "end": 4.0}]}])
        assert kb.youtube.transcript_apify("vid") == [{"text": "yo", "start_sec": 0.0, "end_sec": 4.0}]

    def test_no_items_returns_none(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "_apify_run_actor", lambda *a, **k: [])
        assert kb.youtube.transcript_apify("vid") is None


class TestTranscriptProviderVideoMetaTitle:
    """Fix A, single-video case: a transcript provider's own response may
    incidentally carry a title -- captured into an OPTIONAL `video_meta`
    out-param (never a new network call, never required, never overwrites a
    title the caller already knew)."""

    def test_scrapecreators_fills_title_when_present(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.youtube, "_sc_get",
            lambda path, params: {
                "transcript": [{"text": "hi", "startMs": "0", "endMs": "500"}],
                "title": "SC Video Title",
            },
        )
        meta: dict = {}
        kb.youtube.transcript_scrapecreators("vid123", video_meta=meta)
        assert meta["title"] == "SC Video Title"

    def test_scrapecreators_does_not_override_known_title(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.youtube, "_sc_get",
            lambda path, params: {
                "transcript": [{"text": "hi", "startMs": "0", "endMs": "500"}],
                "title": "Provider Title",
            },
        )
        meta = {"title": "Already Known"}
        kb.youtube.transcript_scrapecreators("vid123", video_meta=meta)
        assert meta["title"] == "Already Known"

    def test_scrapecreators_missing_title_in_payload_is_a_noop(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.youtube, "_sc_get",
            lambda path, params: {"transcript": [{"text": "hi", "startMs": "0", "endMs": "500"}]},
        )
        meta: dict = {}
        kb.youtube.transcript_scrapecreators("vid123", video_meta=meta)
        assert "title" not in meta

    def test_apify_fills_title_when_present(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.youtube, "_apify_run_actor",
            lambda actor, run_input, **kw: [
                {"transcript": [{"text": "hello", "start": 0.0, "duration": 1.0}], "title": "Apify Video Title"}
            ],
        )
        meta: dict = {}
        kb.youtube.transcript_apify("vid123", video_meta=meta)
        assert meta["title"] == "Apify Video Title"

    def test_get_transcript_threads_video_meta_to_fetcher(self, kb, monkeypatch):
        def fake_sc(video_id, *, language="ru", video_meta=None, **_kw):
            if video_meta is not None:
                video_meta["title"] = "From Fetcher"
            return [{"text": "hi", "start_sec": 0.0, "end_sec": 1.0}]

        monkeypatch.setitem(kb.youtube._TRANSCRIPT_FETCHERS, "scrapecreators", fake_sc)
        meta: dict = {}
        segs, prov = kb.youtube.get_transcript(
            "vid", memobase_cfg={"youtube": {"transcript_providers": ["scrapecreators"]}}, video_meta=meta
        )
        assert meta["title"] == "From Fetcher"
        assert prov == "scrapecreators"
        assert segs


class TestDownloadAudioApifyKeyMatching:
    """Fix B: the real actor's response uses `audioFileUrl`, not any of the
    previously-guessed keys -- that mismatch made the ENTIRE STT audio
    fallback unconditionally fail with 'has no audio URL'."""

    def test_audioFileUrl_key_is_found(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.youtube, "_apify_run_actor",
            lambda actor, run_input, **kw: [{"audioFileUrl": "https://apify.example/real.mp3", "audioFormat": "mp3"}],
        )
        seen = {}

        def fake_safe_get(url, **kw):
            seen["url"] = url
            return b"fake-mp3-bytes"

        monkeypatch.setattr(kb.youtube, "safe_get", fake_safe_get)
        audio_bytes, filename = kb.youtube.download_audio_apify("abcdefghijk")
        assert seen["url"] == "https://apify.example/real.mp3"
        assert audio_bytes == b"fake-mp3-bytes"
        assert filename == "abcdefghijk.mp3"

    def test_falls_back_to_legacy_guessed_keys(self, kb, monkeypatch):
        """No regression: a response shaped with only one of the old
        (never-confirmed) keys must still resolve."""
        monkeypatch.setattr(
            kb.youtube, "_apify_run_actor",
            lambda actor, run_input, **kw: [{"downloadUrl": "https://apify.example/legacy.mp3"}],
        )
        monkeypatch.setattr(kb.youtube, "safe_get", lambda url, **kw: b"legacy-bytes")
        audio_bytes, _filename = kb.youtube.download_audio_apify("abcdefghijk")
        assert audio_bytes == b"legacy-bytes"

    def test_no_known_key_still_raises_honest_error(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.youtube, "_apify_run_actor",
            lambda actor, run_input, **kw: [{"somethingElse": "nope"}],
        )
        with pytest.raises(kb.youtube.YoutubeError, match="no audio URL"):
            kb.youtube.download_audio_apify("abcdefghijk")


# --- tool handlers ----------------------------------------------------------


class TestListChannelTool:
    def test_owner_lists_sorted(self, kb, monkeypatch):
        def fake_list(channel, *, memobase_cfg=None, max_pages=None):
            return (
                [
                    {"video_id": "new", "url": "https://youtu.be/new", "title": "New", "published_at": "2026-07-10T00:00:00Z", "duration_s": 754},
                    {"video_id": "old", "url": "https://youtu.be/old", "title": "Old", "published_at": "2026-01-01T00:00:00Z", "duration_s": 60},
                ],
                "scrapecreators",
            )

        monkeypatch.setattr(kb.youtube, "list_channel_videos", fake_list)
        out = kb.tools.memobase_youtube_list_channel({"channel": "@x"})
        assert "12:34" in out  # 754s duration formatted
        assert out.index("new") < out.index("old")  # newest first

    def test_bad_since_rejected(self, kb):
        out = kb.tools.memobase_youtube_list_channel({"channel": "@x", "since": "07/10/2026"})
        assert "YYYY-MM-DD" in out

    def test_guest_refused(self, kb):
        kb.tools.bind_session_user("sess-g", "guest-1")
        out = kb.tools.memobase_youtube_list_channel({"channel": "@x"}, session_id="sess-g")
        assert "владельц" in out


class TestGetTranscriptTool:
    def test_returns_timecoded(self, kb, monkeypatch):
        def fake_get(video_id, *, memobase_cfg=None, language="ru"):
            return ([{"text": "Привет", "start_sec": 5.0, "end_sec": 7.0}], "scrapecreators")

        monkeypatch.setattr(kb.youtube, "get_transcript", fake_get)
        out = kb.tools.memobase_youtube_get_transcript({"video": "https://youtu.be/abcdefghijk"})
        assert "[00:05] Привет" in out and "scrapecreators" in out

    def test_no_captions_points_to_dl_audio(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "get_transcript", lambda *a, **k: (None, None))
        out = kb.tools.memobase_youtube_get_transcript({"video": "abcdefghijk"})
        assert "memobase_youtube_dl_audio" in out

    def test_prefer_provider_reorders(self, kb, monkeypatch):
        seen = {}

        def fake_get(video_id, *, memobase_cfg=None, language="ru"):
            seen["order"] = (memobase_cfg or {}).get("youtube", {}).get("transcript_providers")
            return ([{"text": "x", "start_sec": 0.0, "end_sec": 1.0}], "apify")

        monkeypatch.setattr(kb.youtube, "get_transcript", fake_get)
        kb.tools.memobase_youtube_get_transcript({"video": "abcdefghijk", "prefer_provider": "apify"})
        assert seen["order"][0] == "apify"

    def test_bad_prefer_provider_rejected(self, kb):
        out = kb.tools.memobase_youtube_get_transcript({"video": "abcdefghijk", "prefer_provider": "bogus"})
        assert "prefer_provider" in out

    def test_guest_refused(self, kb):
        kb.tools.bind_session_user("sess-g", "guest-1")
        out = kb.tools.memobase_youtube_get_transcript({"video": "abcdefghijk"}, session_id="sess-g")
        assert "владельц" in out

    def test_segments_joined_with_blank_lines(self, kb, monkeypatch):
        def fake_get(video_id, *, memobase_cfg=None, language="ru"):
            return (
                [
                    {"text": "Привет", "start_sec": 5.0, "end_sec": 7.0},
                    {"text": "Мир", "start_sec": 8.0, "end_sec": 9.0},
                ],
                "scrapecreators",
            )

        monkeypatch.setattr(kb.youtube, "get_transcript", fake_get)
        out = kb.tools.memobase_youtube_get_transcript({"video": "abcdefghijk"})
        assert "[00:05] Привет\n\n[00:08] Мир" in out

    def _five_segments(self):
        return [
            {"text": f"сег{i}", "start_sec": float(i * 10), "end_sec": float(i * 10 + 2)}
            for i in range(5)
        ]

    def test_range_filter(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "get_transcript", lambda *a, **k: (self._five_segments(), "scrapecreators"))
        out = kb.tools.memobase_youtube_get_transcript(
            {"video": "abcdefghijk", "from_sec": 10, "to_sec": 30}
        )
        assert "[00:10]" in out and "[00:30]" in out
        assert "[00:00]" not in out and "[00:40]" not in out
        assert "Показано 3 из 5" in out

    def test_max_segments(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "get_transcript", lambda *a, **k: (self._five_segments(), "scrapecreators"))
        out = kb.tools.memobase_youtube_get_transcript({"video": "abcdefghijk", "max_segments": 2})
        assert "Показано 2 из 5" in out

    def test_no_captions_default_does_not_download(self, kb, monkeypatch):
        called = {"dl": False}

        def fake_dl(vid):
            called["dl"] = True
            return (b"x", "v.mp3")

        monkeypatch.setattr(kb.youtube, "get_transcript", lambda *a, **k: (None, None))
        monkeypatch.setattr(kb.youtube, "download_audio_apify", fake_dl)
        out = kb.tools.memobase_youtube_get_transcript({"video": "abcdefghijk"})
        assert "memobase_youtube_dl_audio" in out and called["dl"] is False

    def test_allow_audio_fallback_runs_stt(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "get_transcript", lambda *a, **k: (None, None))
        monkeypatch.setattr(kb.youtube, "download_audio_apify", lambda vid: (b"audio", "v.mp3"))
        monkeypatch.setattr(
            kb.stt, "transcribe_long_audio",
            lambda audio, *, filename_hint="audio", memobase_cfg=None: {
                "segments": [{"text": "распознано", "start_sec": 0.0, "end_sec": 2.0}],
                "provider": "groq", "trust": "high",
            },
        )
        out = kb.tools.memobase_youtube_get_transcript({"video": "abcdefghijk", "allow_audio_fallback": True})
        assert "фолбэк" in out and "[00:00] распознано" in out


class TestDlAudioTool:
    def test_stt_transcript_with_trust_note(self, kb, monkeypatch):
        monkeypatch.setattr(kb.youtube, "download_audio_apify", lambda vid: (b"audio", "vid.mp3"))
        monkeypatch.setattr(
            kb.stt,
            "transcribe_long_audio",
            lambda audio, *, filename_hint="audio", memobase_cfg=None: {
                "segments": [{"text": "распознано", "start_sec": 0.0, "end_sec": 2.0}],
                "provider": "gemini",
                "trust": "low",
            },
        )
        out = kb.tools.memobase_youtube_dl_audio({"video": "abcdefghijk"})
        assert "[00:00] распознано" in out and "gemini" in out and "низк" in out.lower()

    def test_guest_refused(self, kb):
        kb.tools.bind_session_user("sess-g", "guest-1")
        out = kb.tools.memobase_youtube_dl_audio({"video": "abcdefghijk"}, session_id="sess-g")
        assert "владельц" in out

    def test_stt_failure_message_is_compact_and_has_video_id(self, kb, monkeypatch):
        """Fix C: whatever stt.py raises must not dump a raw provider
        JSON/body into the chat -- the message stays short and still names
        the video id; the full detail only goes to the debug log."""
        monkeypatch.setattr(kb.youtube, "download_audio_apify", lambda vid: (b"audio", "vid.mp3"))
        huge_message = "RAW PROVIDER JSON " * 60  # simulates a leaked provider body

        def fake_transcribe(*a, **k):
            raise kb.stt.SttError(huge_message)

        monkeypatch.setattr(kb.stt, "transcribe_long_audio", fake_transcribe)
        out = kb.tools.memobase_youtube_dl_audio({"video": "abcdefghijk"})
        assert "abcdefghijk" in out
        assert huge_message not in out
        assert len(out) < 300
