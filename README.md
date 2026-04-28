---
title: RAG Chatbot
emoji: 🤖
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Ask 10-K: RAG over SEC Annual Reports

A retrieval-augmented chat interface for SEC 10-K filings. Upload three banks' annual reports and ask grounded, citation-backed questions across them, like *"How does Capital One's net charge-off rate compare to JPMorgan's?"* or *"Summarize each bank's stated CRE exposure risks."*

Built as a study in production-shaped RAG: typed config, eval harness, two interchangeable LLM backends, and a frontend that surfaces citations rather than hiding them.

[![CI](https://github.com/kelvinasiedu-programmer/rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/kelvinasiedu-programmer/rag-chatbot/actions/workflows/ci.yml)

**Live demo:** [rag-chatbot-web.vercel.app](https://rag-chatbot-web.vercel.app/) · **Frontend repo:** [rag-chatbot-web](https://github.com/kelvinasiedu-programmer/rag-chatbot-web)

## Why 10-Ks?

10-Ks are dense, terminology-heavy, structurally complex financial documents. They're the workload where retrieval quality (precision, grounding, citation accuracy) actually matters. Generic FAQ-style RAG demos don't surface this. Sample queries that work end-to-end:

- *"What were Capital One's net charge-offs in 2024 and how does that compare to BAC?"*
- *"Summarize each bank's stated risks around commercial real estate exposure."*
- *"Which of these banks has the highest CET1 ratio?"*
- *"How does each company describe its exposure to AI/LLM-related operational risk?"*

## Architecture

```
Client ──▶ FastAPI REST API ──▶ RAG Engine
                                   │
                         ┌─────────┴──────────────┐
                         ▼                        ▼
                    FAISS Vector             Generator
                      Store              ┌──────┴───────┐
                   (retrieval)           ▼              ▼
                                     flan-t5-base  Claude API
                                       (free)     (toggle, paid)
```

**Pipeline:**

1. **Ingest:** PDFs are parsed, sentence-split, embedded, and added to a FAISS vector index. Scanned/image-only PDFs return HTTP 422 instead of silently indexing zero chunks.
2. **Retrieve:** query is embedded and matched against stored chunks via similarity search.
3. **Generate:** retrieved context is injected into a prompt template; the configured generator returns a grounded answer with source citations (filename + page number + similarity score).

## Two LLM Backends (Toggle)

| `LLM_BACKEND` | Model              | Latency on free tier | Cost           | When to use                           |
| ------------- | ------------------ | -------------------- | -------------- | ------------------------------------- |
| `local` *(default)* | `flan-t5-base`     | 3–5 s                | $0             | Free demo, local dev, no API keys     |
| `anthropic`   | `claude-haiku-4-5` | 0.5–1.5 s            | ~$0.001/query  | Higher answer quality, faster replies |

Both backends share the same `Generator` interface ([rag_engine.py](src/rag_engine.py)). Adding Ollama or another local model is a ~30-line addition.

## Quick Start

### Prerequisites
- Python 3.10+
- (Optional) Anthropic API key if you want to use the Claude backend

### Install & Run

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env

uvicorn src.main:app --reload
```

API at `http://localhost:8000`, OpenAPI docs at `/docs`.

### Load the sample 10-K corpus

The repo doesn't ship the filings (they update annually). See [data/sample_corpus/README.md](data/sample_corpus/README.md) for the recommended starter set (Capital One, JPM, BAC) and one-line download instructions.

### Docker

```bash
cp .env.example .env
docker compose up -d
```

## API Endpoints

| Method   | Endpoint                    | Description            |
| -------- | --------------------------- | ---------------------- |
| `POST`   | `/api/v1/documents/upload`  | Upload a PDF (returns 422 if no extractable text) |
| `POST`   | `/api/v1/query`             | Ask a question         |
| `GET`    | `/api/v1/documents`         | Get document stats     |
| `DELETE` | `/api/v1/documents`         | Clear all documents    |
| `GET`    | `/api/v1/health`            | Health check           |

### Response Format

```json
{
  "answer": "Capital One reported net charge-offs of $X.XB in 2024, up from $Y.YB in 2023, driven primarily by credit-card portfolio normalization.",
  "sources": [
    {
      "text": "[Page 47] Net charge-offs in our Credit Card segment were ...",
      "score": 0.83,
      "metadata": {"source": "cof-10k-2024.pdf", "page": 47, "chunk_index": 2}
    }
  ]
}
```

## Configuration

| Variable                | Default              | Description                                    |
| ----------------------- | -------------------- | ---------------------------------------------- |
| `LLM_BACKEND`           | `local`              | `local` (flan-t5, free) or `anthropic`         |
| `LOCAL_LLM_MODEL`       | `google/flan-t5-base`| Local generator (HuggingFace)                  |
| `ANTHROPIC_LLM_MODEL`   | `claude-haiku-4-5`   | Claude model when `LLM_BACKEND=anthropic`      |
| `ANTHROPIC_API_KEY`     | *(unset)*            | Required when `LLM_BACKEND=anthropic`          |
| `EMBEDDING_MODEL`       | `all-MiniLM-L12-v2`  | Sentence-transformer for retrieval             |
| `CHUNK_SIZE`            | `500`                | Target chars per chunk (sentence-aware split)  |
| `CHUNK_OVERLAP`         | `50`                 | Tail-overlap between adjacent chunks           |
| `TOP_K`                 | `3`                  | Context chunks retrieved per query             |
| `MAX_UPLOAD_SIZE_MB`    | `10`                 | PDF upload ceiling                             |

## Engineering Notes

- **Sentence-boundary chunking:** chunks are packed at sentence boundaries up to `CHUNK_SIZE`, preserving semantic units instead of cutting mid-word ([pdf_processor.py](src/pdf_processor.py)).
- **No silent ingestion failures:** uploading a scanned/image-only PDF returns 422 with a clear error rather than reporting "0 chunks indexed" as success.
- **Persistent vector index:** FAISS index + JSON metadata are saved to `./data/vector_store/` after each ingest.
- **Mocked tests for both backends:** `pytest` runs without spending API credits.

## Testing

```bash
pip install -r requirements-dev.txt
make test
```

32 tests cover PDF chunking, vector-store search, both LLM backends, and the upload-error path.

## Tech Stack

| Component              | Technology                                                |
| ---------------------- | --------------------------------------------------------- |
| API Framework          | FastAPI                                                   |
| Vector Search          | FAISS                                                     |
| Embeddings             | Sentence Transformers (`all-MiniLM-L12-v2`)               |
| LLM (default)          | HuggingFace Transformers (`flan-t5-base`)                 |
| LLM (toggle)           | Anthropic Claude (`claude-haiku-4-5`)                     |
| Validation             | Pydantic v2                                               |
| Containerization       | Docker (multi-stage build)                                |
| CI/CD                  | GitHub Actions                                            |
| Testing                | pytest + coverage                                         |

## License

MIT
