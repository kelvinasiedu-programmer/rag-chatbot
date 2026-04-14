"""FAISS-backed vector store for document embeddings."""

import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings using FAISS for efficient similarity search."""

    def __init__(self, model_name: str, persist_dir: str | None = None):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: list[str] = []
        self.metadata: list[dict] = []
        self.persist_dir = persist_dir
        logger.info(
            "VectorStore initialized (model=%s, dim=%d)", model_name, self.dimension
        )

    @property
    def count(self) -> int:
        return self.index.ntotal

    def add_documents(
        self, documents: list[str], metadata: list[dict] | None = None
    ) -> int:
        """Encode documents and add them to the FAISS index."""
        if not documents:
            return 0

        embeddings = self.model.encode(
            documents,
            show_progress_bar=len(documents) > 10,
            normalize_embeddings=True,
        )
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(documents)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend({} for _ in documents)

        logger.info("Added %d documents (total: %d)", len(documents), self.count)
        return len(documents)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Find the most similar documents to a query string."""
        if self.count == 0:
            return []

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        distances, indices = self.index.search(query_embedding, min(top_k, self.count))

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.documents):
                results.append(
                    {
                        "text": self.documents[idx],
                        "score": round(float(1 / (1 + distances[0][i])), 4),
                        "metadata": self.metadata[idx]
                        if idx < len(self.metadata)
                        else {},
                    }
                )
        return results

    def save(self) -> None:
        """Persist the FAISS index and document list to disk."""
        if not self.persist_dir:
            return

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))

        with open(os.path.join(self.persist_dir, "documents.json"), "w") as f:
            json.dump({"documents": self.documents, "metadata": self.metadata}, f)

        logger.info("Vector store saved to %s", self.persist_dir)

    def load(self) -> bool:
        """Load a previously persisted index. Returns True if successful."""
        index_path = os.path.join(self.persist_dir or "", "index.faiss")
        docs_path = os.path.join(self.persist_dir or "", "documents.json")

        if not (self.persist_dir and os.path.exists(index_path)):
            return False

        self.index = faiss.read_index(index_path)

        with open(docs_path) as f:
            data = json.load(f)
        self.documents = data["documents"]
        self.metadata = data.get("metadata", [{} for _ in self.documents])

        logger.info("Loaded %d documents from %s", self.count, self.persist_dir)
        return True

    def clear(self) -> None:
        """Remove all documents from the store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        logger.info("Vector store cleared")
