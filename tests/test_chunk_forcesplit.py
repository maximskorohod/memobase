"""Force-split of oversized prose units in chunk.py (v2): an ASR-transcript
blob with single-\n seams must produce many target-sized chunks, while code
fences stay atomic and micro-target configs are protected by the absolute
FORCE_SPLIT_MIN_TOKENS floor."""

from __future__ import annotations


def _transcript_blob(lines: int = 400) -> str:
    rows = []
    for i in range(lines):
        rows.append(f"[{i // 60:02d}:{i % 60:02d}] сегмент номер {i} " + "слово " * 12)
    return "\n".join(rows)


class TestForceSplitTranscript:
    def _doc(self, text):
        return {"text": text, "blocks": [{"text": text, "page": None, "section": None, "is_code": False}]}

    def test_single_paragraph_transcript_splits_into_many_chunks(self, kb):
        chunks = kb.chunk.chunk(self._doc(_transcript_blob()), 900, 0.15)
        assert len(chunks) >= 5
        assert max(kb.chunk.approx_tokens(c["text"]) for c in chunks) <= 1800

    def test_window_overlap_present_between_consecutive_chunks(self, kb):
        chunks = kb.chunk.chunk(self._doc(_transcript_blob()), 900, 0.15)
        assert len(chunks) >= 2
        tail_lines = chunks[0]["text"].split("\n\n")[-3:]
        assert any(line in chunks[1]["text"] for line in tail_lines)

    def test_monster_single_line_hard_sliced(self, kb):
        chunks = kb.chunk.chunk(self._doc("слово " * 5000), 900, 0.15)
        assert len(chunks) >= 3
        # no piece got silently dropped: total text is preserved-ish (words survive)
        assert sum(c["text"].count("слово") for c in chunks) >= 4500

    def test_micro_target_paragraphs_not_shredded(self, kb):
        paras = ["Абзац номер один про возврат товара в магазине." for _ in range(3)]
        text = "\n\n".join(f"{p} вариант {i}" for i, p in enumerate(paras))
        chunks = kb.chunk.chunk(self._doc(text), 5, 0.15)
        # target=5 => upper=8 < paragraph size, but FORCE_SPLIT_MIN_TOKENS=256
        # floor keeps each paragraph intact as its own atomic-ish chunk.
        assert len(chunks) == 3
        for c in chunks:
            assert c["text"].startswith("Абзац номер один")

    def test_code_fence_stays_atomic_even_when_huge(self, kb):
        code = "def f():\n" + "    x = 1  # довольно длинная строка кода\n" * 150
        doc = {
            "text": code,
            "blocks": [{"text": code, "page": None, "section": None, "is_code": True}],
        }
        chunks = kb.chunk.chunk(doc, 900, 0.15)
        assert any(code in c["text"] for c in chunks)
