"""Core RAG pipeline: retrieval + generation."""

import logging

from transformers import pipeline as hf_pipeline

from .config import Settings
from .pdf_processor import PDFProcessor
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Answer the question based only on the provided context. \
Be concise and accurate. If the context does not contain enough information, \
say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""


class RAGEngine:
    """Orchestrates the retrieval-augmented generation pipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings

        logger.info("Loading embedding model: %s", settings.embedding_model)
        self.vector_store = VectorStore(
            model_name=settings.embedding_model,
            persist_dir=settings.vector_store_path,
        )

        self.pdf_processor = PDFProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        logger.info("Loading LLM: %s", settings.llm_model)
        self.llm = hf_pipeline("text2text-generation", model=settings.llm_model)

        logger.info("RAG Engine ready")

    def ingest_pdf(self, pdf_path: str) -> int:
        """Parse a PDF, chunk its text, and add to the vector store."""
        results = self.pdf_processor.extract_chunks(pdf_path)
        texts = [r["text"] for r in results]
        metadata = [r["metadata"] for r in results]
        count = self.vector_store.add_documents(texts, metadata)
        self.vector_store.save()
        return count

    def query(self, question: str) -> dict:
        """Retrieve relevant context and generate a grounded answer."""
        results = self.vector_store.search(question, top_k=self.settings.top_k)

        if not results:
            return {
                "answer": "No documents have been loaded. Please upload a PDF first.",
                "sources": [],
            }

        context = "\n\n".join(r["text"] for r in results)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        response = self.llm(
            prompt,
            max_new_tokens=self.settings.max_tokens,
            num_return_sequences=1,
        )
        answer = response[0]["generated_text"].strip()

        sources = [
            {
                "text": r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
                "score": r["score"],
                "metadata": r["metadata"],
            }
            for r in results
        ]

        return {"answer": answer, "sources": sources}
