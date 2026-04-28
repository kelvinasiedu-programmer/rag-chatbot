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


def _patched_engine(settings, generator_path):
    """Patch the appropriate generator class and the vector store, return engine."""
    with (
        patch(generator_path) as mock_gen_cls,
        patch("src.rag_engine.VectorStore") as mock_vs_cls,
    ):
        gen = MagicMock()
        gen.generate.return_value = "Test answer"
        mock_gen_cls.return_value = gen

        vs = MagicMock()
        vs.count = 5
        vs.search.return_value = [
            {"text": "Relevant chunk 1", "score": 0.85, "metadata": {"page": 1}},
            {"text": "Relevant chunk 2", "score": 0.72, "metadata": {"page": 2}},
        ]
        mock_vs_cls.return_value = vs

        return RAGEngine(settings)


class TestLocalBackend:
    def test_query_returns_answer_and_sources(self, tmp_path):
        settings = Settings(
            llm_backend="local",
            vector_store_path=str(tmp_path / "vs"),
        )
        engine = _patched_engine(settings, "src.rag_engine._LocalGenerator")
        result = engine.query("What is the policy?")
        assert result["answer"] == "Test answer"
        assert len(result["sources"]) == 2


class TestAnthropicBackend:
    def test_requires_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = Settings(
            llm_backend="anthropic",
            anthropic_api_key=None,
            vector_store_path=str(tmp_path / "vs"),
        )
        with patch("src.rag_engine.VectorStore"):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                RAGEngine(settings)

    def test_query_uses_anthropic_generator(self, tmp_path):
        settings = Settings(
            llm_backend="anthropic",
            anthropic_api_key="test-key",
            vector_store_path=str(tmp_path / "vs"),
        )
        engine = _patched_engine(settings, "src.rag_engine._AnthropicGenerator")
        result = engine.query("Anything?")
        assert result["answer"] == "Test answer"


class TestUnknownBackend:
    def test_rejects_invalid_backend(self, tmp_path):
        settings = Settings(
            llm_backend="banana",
            vector_store_path=str(tmp_path / "vs"),
        )
        with patch("src.rag_engine.VectorStore"):
            with pytest.raises(ValueError, match="Unknown llm_backend"):
                RAGEngine(settings)


class TestEmptyStore:
    def test_empty_store_returns_fallback(self, tmp_path):
        settings = Settings(
            llm_backend="local",
            vector_store_path=str(tmp_path / "vs"),
        )
        with (
            patch("src.rag_engine._LocalGenerator"),
            patch("src.rag_engine.VectorStore") as mock_vs_cls,
        ):
            vs = MagicMock()
            vs.count = 0
            vs.search.return_value = []
            mock_vs_cls.return_value = vs

            engine = RAGEngine(settings)
            result = engine.query("anything")
            assert "no documents" in result["answer"].lower() or "upload" in result["answer"].lower()
