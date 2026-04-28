"""Core RAG pipeline: retrieval + generation.

Generation backend is pluggable via Settings.llm_backend:
- "local"     : flan-t5-base via transformers (free, slower, lower quality)
- "anthropic" : Claude via Anthropic API (faster, higher quality, requires credits)
"""

import logging
import os

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


class _LocalGenerator:
    """flan-t5-base via HuggingFace transformers — free, runs on CPU."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._torch = torch

    def generate(self, prompt: str, max_tokens: int) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        with self._torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class _AnthropicGenerator:
    """Claude via Anthropic API — requires ANTHROPIC_API_KEY + paid credits."""

    def __init__(self, model_name: str, api_key: str):
        from anthropic import Anthropic

        self.model_name = model_name
        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int) -> str:
        msg = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()


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

        backend = settings.llm_backend.lower()
        if backend == "anthropic":
            api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "llm_backend='anthropic' but ANTHROPIC_API_KEY is not set."
                )
            logger.info("Loading Anthropic LLM: %s", settings.anthropic_llm_model)
            self.generator = _AnthropicGenerator(settings.anthropic_llm_model, api_key)
        elif backend == "local":
            logger.info("Loading local LLM: %s", settings.local_llm_model)
            self.generator = _LocalGenerator(settings.local_llm_model)
        else:
            raise ValueError(
                f"Unknown llm_backend '{settings.llm_backend}'. Use 'local' or 'anthropic'."
            )

        logger.info("RAG Engine ready (backend=%s)", backend)

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
        answer = self.generator.generate(prompt, self.settings.max_tokens)

        sources = [
            {
                "text": r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
                "score": r["score"],
                "metadata": r["metadata"],
            }
            for r in results
        ]

        return {"answer": answer, "sources": sources}
