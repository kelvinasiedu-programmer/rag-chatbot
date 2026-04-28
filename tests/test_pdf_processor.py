"""Tests for the PDFProcessor class."""

import pytest

from src.pdf_processor import PDFProcessor


@pytest.fixture
def processor():
    return PDFProcessor(chunk_size=100, chunk_overlap=20)


class TestCleanText:
    def test_collapses_whitespace(self):
        assert PDFProcessor.clean_text("  Hello   world\n\nfoo   bar  ") == "Hello world foo bar"

    def test_empty_string(self):
        assert PDFProcessor.clean_text("") == ""

    def test_already_clean(self):
        assert PDFProcessor.clean_text("no change") == "no change"


class TestSplitText:
    def test_produces_multiple_chunks(self, processor):
        text = "A" * 250
        chunks = processor._split_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_short_text_single_chunk(self, processor):
        chunks = processor._split_text("Short text")
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_empty_text(self, processor):
        assert processor._split_text("") == []

    def test_overlap_exists(self):
        proc = PDFProcessor(chunk_size=10, chunk_overlap=3)
        chunks = proc._split_text("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert len(chunks) > 1

    def test_respects_sentence_boundaries(self):
        """Sentences should not be split mid-word when they fit."""
        proc = PDFProcessor(chunk_size=80, chunk_overlap=10)
        text = (
            "Capital expenditures rose 12% year over year. "
            "Net interest income fell on margin compression. "
            "Management reaffirmed full-year guidance."
        )
        chunks = proc._split_text(text)
        # Each chunk should end on sentence-final punctuation (no mid-sentence cuts).
        for c in chunks:
            assert c.rstrip().endswith((".", "!", "?")), f"chunk does not end on sentence boundary: {c!r}"


class TestExtractChunks:
    def test_blank_pdf_returns_empty(self, sample_pdf):
        proc = PDFProcessor(chunk_size=50, chunk_overlap=10)
        results = proc.extract_chunks(sample_pdf)
        assert isinstance(results, list)
        # Blank page has no text → expect empty
        assert len(results) == 0
