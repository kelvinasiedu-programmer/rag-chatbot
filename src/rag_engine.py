"""Core RAG pipeline: retrieval + generation."""

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.llm_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        logger.info("RAG Engine ready (device=%s)", self.device)

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

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.settings.max_tokens
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        sources = [
            {
                "text": r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
                "score": r["score"],
                "metadata": r["metadata"],
            }
            for r in results
        ]

        return {"answer": answer, "sources": sources}
