"""CLI/slash youtube-* подкоманды: парсер argparse и тонкая делегация в
tools.memobase_youtube_* (сами тулы замоканы — сети нет)."""

from __future__ import annotations

import argparse


class TestYoutubeCliParser:
    def _configured_parser(self, kb):
        """Mirror TestCliRegistration (test_plugin.py): a bare
        argparse.ArgumentParser handed straight to cli._setup(). ``_setup``
        only ever calls ``.add_subparsers()``/``.add_parser()`` on whatever
        it's given, so it works identically whether that's a top-level
        parser (here) or a parser nested under an outer dispatcher (the real
        hermes CLI's own ``register_cli_command`` plumbing)."""
        parser = argparse.ArgumentParser()
        kb.cli._setup(parser)
        return parser

    def test_parses_youtube_list(self, kb):
        parser = self._configured_parser(kb)
        ns = parser.parse_args(["youtube-list", "@x", "--limit", "5"])
        assert ns.channel == "@x" and ns.limit == 5

    def test_parses_youtube_list_defaults(self, kb):
        parser = self._configured_parser(kb)
        ns = parser.parse_args(["youtube-list", "@x"])
        assert ns.channel == "@x"
        assert ns.limit is None and ns.since is None and ns.max_pages is None

    def test_parses_youtube_transcript_flags(self, kb):
        parser = self._configured_parser(kb)
        ns = parser.parse_args(
            ["youtube-transcript", "vid123", "--from-sec", "10", "--allow-audio-fallback"]
        )
        assert ns.video == "vid123" and ns.from_sec == 10.0 and ns.allow_audio_fallback is True

    def test_parses_youtube_transcript_defaults(self, kb):
        parser = self._configured_parser(kb)
        ns = parser.parse_args(["youtube-transcript", "vid123"])
        assert ns.video == "vid123"
        assert ns.allow_audio_fallback is False
        assert ns.max_segments is None and ns.from_sec is None and ns.to_sec is None

    def test_parses_youtube_audio(self, kb):
        parser = self._configured_parser(kb)
        ns = parser.parse_args(["youtube-audio", "vid123"])
        assert ns.video == "vid123"


class TestYoutubeSlash:
    def test_youtube_list_dispatches(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.tools, "memobase_youtube_list_channel",
            lambda args, **kw: f"OK-LIST {args.get('channel')} {args.get('limit')}",
        )
        out = kb.commands.handle_kb_command("youtube-list @x 5")
        assert "OK-LIST @x 5" in out

    def test_youtube_list_no_args_usage(self, kb):
        out = kb.commands.handle_kb_command("youtube-list")
        assert "Использование" in out

    def test_youtube_list_bad_limit(self, kb):
        out = kb.commands.handle_kb_command("youtube-list @x notanumber")
        assert "не число" in out

    def test_youtube_transcript_dispatches(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.tools, "memobase_youtube_get_transcript",
            lambda args, **kw: f"OK-TRANSCRIPT {args.get('video')}",
        )
        out = kb.commands.handle_kb_command("youtube-transcript vid123")
        assert "OK-TRANSCRIPT vid123" in out

    def test_youtube_transcript_usage_line(self, kb):
        out = kb.commands.handle_kb_command("youtube-transcript")
        assert "Использование" in out

    def test_youtube_audio_dispatches(self, kb, monkeypatch):
        monkeypatch.setattr(
            kb.tools, "memobase_youtube_dl_audio",
            lambda args, **kw: f"OK-AUDIO {args.get('video')}",
        )
        out = kb.commands.handle_kb_command("youtube-audio vid123")
        assert "OK-AUDIO vid123" in out

    def test_youtube_audio_usage_line(self, kb):
        out = kb.commands.handle_kb_command("youtube-audio")
        assert "Использование" in out
