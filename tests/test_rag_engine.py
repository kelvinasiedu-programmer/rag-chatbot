"""Tests for the RAG engine (uses mocked models for CI speed)."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.rag_engine import PROMPT_TEMPLATE, RAGEngine


class TestPromptTemplate:
    def test_has_placeholders(self):
        assert "{context}" in PROMPT_TEMPLATE
        assert "{question}" in PROMPT_TEMPLATE

    def test_renders_correctly(self):
        rendered = PROMPT_TEMPLATE.format(context="ctx", question="q")
        assert "ctx" in rendered
        assert "q" in rendered


class TestRAGEngineQuery:
    @pytest.fixture
    def mock_engine(self, tmp_path):
        settings = Settings(
            embedding_model="all-MiniLM-L12-v2",
            llm_model="google/flan-t5-base",
            vector_store_path=str(tmp_path / "vs"),
        )
        with (
            patch("src.rag_engine.hf_pipeline") as mock_pipe,
            patch("src.rag_engine.VectorStore") as mock_vs_cls,
        ):
            mock_pipe.return_value = MagicMock(
                return_value=[{"generated_text": "Test answer"}]
            )
            mock_vs = MagicMock()
            mock_vs.count = 5
            mock_vs.search.return_value = [
                {"text": "Relevant chunk 1", "score": 0.85, "metadata": {"page": 1}},
                {"text": "Relevant chunk 2", "score": 0.72, "metadata": {"page": 2}},
            ]
            mock_vs_cls.return_value = mock_vs

            engine = RAGEngine(settings)
            yield engine

    def test_query_returns_answer_and_sources(self, mock_engine):
        result = mock_engine.query("What is the policy?")
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_empty_store_returns_fallback(self, tmp_path):
        settings = Settings(
            embedding_model="all-MiniLM-L12-v2",
            llm_model="google/flan-t5-base",
            vector_store_path=str(tmp_path / "vs"),
        )
        with (
            patch("src.rag_engine.hf_pipeline"),
            patch("src.rag_engine.VectorStore") as mock_vs_cls,
        ):
            mock_vs = MagicMock()
            mock_vs.count = 0
            mock_vs.search.return_value = []
            mock_vs_cls.return_value = mock_vs

            engine = RAGEngine(settings)
            result = engine.query("anything")
            assert "no documents" in result["answer"].lower() or "upload" in result["answer"].lower()
