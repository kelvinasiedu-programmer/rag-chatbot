"""Tests for the FastAPI HTTP layer (mocked engine)."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.main as main_mod


@pytest.fixture
def client(monkeypatch):
    fake_engine = MagicMock()
    fake_engine.vector_store.count = 1
    monkeypatch.setattr(main_mod, "engine", fake_engine)
    return TestClient(main_mod.app), fake_engine


class TestUploadEndpoint:
    def test_rejects_non_pdf(self, client):
        c, _ = client
        resp = c.post(
            "/api/v1/documents/upload",
            files={"file": ("hello.txt", b"hi", "text/plain")},
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    def test_returns_422_when_no_chunks_extracted(self, client):
        c, eng = client
        eng.ingest_pdf.return_value = 0  # scanned/image-only PDF case
        resp = c.post(
            "/api/v1/documents/upload",
            files={"file": ("scan.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "scanned" in detail.lower() or "image-only" in detail.lower()

    def test_success_returns_chunks_added(self, client):
        c, eng = client
        eng.ingest_pdf.return_value = 7
        eng.vector_store.count = 7
        resp = c.post(
            "/api/v1/documents/upload",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_added"] == 7
