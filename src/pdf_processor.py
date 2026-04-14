"""PDF text extraction and chunking utilities."""

import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


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
        """Split *text* into overlapping chunks."""
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks
