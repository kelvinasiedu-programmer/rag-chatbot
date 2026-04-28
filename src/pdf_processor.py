"""PDF text extraction and chunking utilities."""

import logging
import re
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Sentence boundary: end punctuation followed by whitespace.
# Avoids splitting on common abbreviations (Mr., U.S., etc.) for the simple cases.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")


class PDFProcessor:
    """Extracts and chunks text from PDF documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def clean_text(text: str) -> str:
        """Collapse whitespace and strip surrounding blanks."""
        return " ".join(text.split())

    def extract_chunks(
        self,
        pdf_path: str,
        start_page: int = 0,
        end_page: int | None = None,
    ) -> list[dict]:
        """Extract text chunks from a PDF with page-level metadata.

        Returns a list of dicts, each with ``text`` and ``metadata`` keys.
        """
        reader = PdfReader(pdf_path)
        filename = Path(pdf_path).name
        pages = reader.pages[start_page:end_page]

        results: list[dict] = []
        for page_num, page in enumerate(pages, start_page):
            text = self.clean_text(page.extract_text() or "")
            if len(text) < 50:
                continue

            chunks = self._split_text(text)
            for i, chunk in enumerate(chunks):
                results.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                            "chunk_index": i,
                        },
                    }
                )

        logger.info(
            "Extracted %d chunks from %s (%d pages)",
            len(results),
            filename,
            len(pages),
        )
        return results

    def _split_text(self, text: str) -> list[str]:
        """Split *text* into chunks at sentence boundaries when possible.

        Greedily packs sentences into chunks up to ``chunk_size`` chars. Carries
        the tail of the previous chunk forward as overlap to preserve context
        across boundaries. Falls back to character-window splitting for very
        long sentences (e.g. tables flattened into one line).
        """
        if not text:
            return []

        sentences = _SENT_SPLIT_RE.split(text)
        chunks: list[str] = []
        current = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Sentence longer than chunk_size on its own — slice it by chars.
            if len(sent) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                step = max(self.chunk_size - self.chunk_overlap, 1)
                for i in range(0, len(sent), step):
                    piece = sent[i : i + self.chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                continue

            candidate = f"{current} {sent}".strip() if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current)
                # Overlap: carry the tail of the previous chunk forward
                tail = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = f"{tail} {sent}".strip() if tail else sent

        if current:
            chunks.append(current)
        return chunks
