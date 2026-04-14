"""Shared test fixtures."""

import tempfile

import pytest


@pytest.fixture
def sample_texts():
    return [
        "Python is a popular programming language used for web development and data science.",
        "Machine learning models learn patterns from training data to make predictions.",
        "Docker containers package applications with their dependencies for consistent deployment.",
        "REST APIs use HTTP methods like GET, POST, PUT, and DELETE for communication.",
        "Vector databases store embeddings for efficient similarity search operations.",
    ]


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal test PDF using pypdf."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return str(pdf_path)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d
