"""Tests for the VectorStore class."""

import os

import pytest

from src.vector_store import VectorStore

EMBEDDING_MODEL = "all-MiniLM-L12-v2"


@pytest.fixture
def store(temp_dir):
    return VectorStore(model_name=EMBEDDING_MODEL, persist_dir=temp_dir)


class TestVectorStoreInit:
    def test_dimension_positive(self, store):
        assert store.dimension > 0

    def test_starts_empty(self, store):
        assert store.count == 0
        assert store.documents == []


class TestAddDocuments:
    def test_add_returns_count(self, store, sample_texts):
        assert store.add_documents(sample_texts) == len(sample_texts)

    def test_count_after_add(self, store, sample_texts):
        store.add_documents(sample_texts)
        assert store.count == len(sample_texts)

    def test_add_empty_list(self, store):
        assert store.add_documents([]) == 0
        assert store.count == 0

    def test_metadata_stored(self, store):
        docs = ["hello world"]
        meta = [{"source": "test.pdf", "page": 1}]
        store.add_documents(docs, metadata=meta)
        results = store.search("hello", top_k=1)
        assert results[0]["metadata"] == meta[0]


class TestSearch:
    def test_returns_relevant_results(self, store, sample_texts):
        store.add_documents(sample_texts)
        results = store.search("What is machine learning?", top_k=2)
        assert len(results) == 2
        assert any("machine learning" in r["text"].lower() for r in results)

    def test_empty_store(self, store):
        assert store.search("anything") == []

    def test_respects_top_k(self, store, sample_texts):
        store.add_documents(sample_texts)
        assert len(store.search("programming", top_k=1)) == 1

    def test_result_has_score(self, store, sample_texts):
        store.add_documents(sample_texts)
        results = store.search("Python", top_k=1)
        assert "score" in results[0]
        assert 0 < results[0]["score"] <= 1


class TestPersistence:
    def test_save_and_load(self, store, sample_texts, temp_dir):
        store.add_documents(sample_texts)
        store.save()

        new_store = VectorStore(model_name=EMBEDDING_MODEL, persist_dir=temp_dir)
        assert new_store.load() is True
        assert new_store.count == len(sample_texts)
        assert new_store.documents == store.documents

    def test_load_nonexistent(self, temp_dir):
        store = VectorStore(
            model_name=EMBEDDING_MODEL,
            persist_dir=os.path.join(temp_dir, "nonexistent"),
        )
        assert store.load() is False


class TestClear:
    def test_clear_resets(self, store, sample_texts):
        store.add_documents(sample_texts)
        store.clear()
        assert store.count == 0
        assert store.documents == []
