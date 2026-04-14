---
title: RAG Chatbot
emoji: robot
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RAG Chatbot

Retrieval-Augmented Generation chatbot that answers questions from PDF documents using embedding-based retrieval and language model generation.

[![CI](https://github.com/kelvinasiedu-programmer/rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/kelvinasiedu-programmer/rag-chatbot/actions/workflows/ci.yml)

## Architecture

```
Client ──▶ FastAPI REST API ──▶ RAG Engine
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                    FAISS Vector         HuggingFace
                      Store               LLM
                   (retrieval)         (generation)
```

**Pipeline:**

1. **Ingest** — PDFs are parsed, cleaned, split into overlapping chunks, and embedded into a FAISS vector index
2. **Retrieve** — User queries are embedded and matched against stored chunks via L2 similarity search
3. **Generate** — Retrieved context is injected into a prompt template; the LLM generates a grounded answer with source citations

## Features

- **REST API** — FastAPI with auto-generated OpenAPI/Swagger docs
- **FAISS vector search** — Facebook AI Similarity Search for scalable retrieval
- **Persistent storage** — vector index and documents survive server restarts
- **PDF upload** — streaming upload endpoint with file size validation
- **Rate limiting** — sliding-window middleware for abuse prevention
- **Source citations** — every answer includes scored source chunks with page numbers
- **Evaluation framework** — keyword-recall metrics for tuning RAG quality
- **Docker** — multi-stage build with health checks
- **CI/CD** — GitHub Actions: lint, test (Python 3.10–3.12), Docker build
- **Type-safe config** — Pydantic Settings with `.env` file support

## Quick Start

### Prerequisites

- Python 3.10+

### Install & Run

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env

# Run
uvicorn src.main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
cp .env.example .env
docker compose up -d
```

## API Endpoints

| Method   | Endpoint                    | Description            |
| -------- | --------------------------- | ---------------------- |
| `POST`   | `/api/v1/documents/upload`  | Upload a PDF document  |
| `POST`   | `/api/v1/query`             | Ask a question         |
| `GET`    | `/api/v1/documents`         | Get document stats     |
| `DELETE` | `/api/v1/documents`         | Clear all documents    |
| `GET`    | `/api/v1/health`            | Health check           |

### Example Usage

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the password policy?"}'
```

### Response Format

```json
{
  "answer": "Passwords must be at least 12 characters and rotated every 365 days.",
  "sources": [
    {
      "text": "[Page 5] Password requirements include...",
      "score": 0.8234,
      "metadata": {"source": "handbook.pdf", "page": 5, "chunk_index": 2}
    }
  ]
}
```

## Configuration

All settings are configurable via environment variables or `.env`:

| Variable              | Default              | Description                       |
| --------------------- | -------------------- | --------------------------------- |
| `EMBEDDING_MODEL`     | `all-MiniLM-L12-v2`  | Sentence transformer model        |
| `LLM_MODEL`           | `google/flan-t5-base` | Text generation model             |
| `CHUNK_SIZE`          | `500`                | Characters per text chunk         |
| `CHUNK_OVERLAP`       | `50`                 | Overlap between adjacent chunks   |
| `TOP_K`               | `3`                  | Context chunks retrieved per query|
| `RATE_LIMIT_REQUESTS` | `10`                 | Max requests per time window      |
| `MAX_UPLOAD_SIZE_MB`  | `10`                 | Maximum PDF upload size           |

## Testing

```bash
pip install -r requirements-dev.txt
make test
```

## Project Structure

```
rag-chatbot/
├── src/
│   ├── main.py           # FastAPI application and routes
│   ├── config.py          # Pydantic Settings configuration
│   ├── rag_engine.py      # Core RAG pipeline orchestration
│   ├── vector_store.py    # FAISS-backed vector database
│   ├── pdf_processor.py   # PDF extraction and chunking
│   ├── schemas.py         # API request/response models
│   └── evaluation.py      # RAG quality evaluation utilities
├── tests/
│   ├── conftest.py        # Shared pytest fixtures
│   ├── test_vector_store.py
│   ├── test_pdf_processor.py
│   └── test_rag_engine.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Tech Stack

| Component              | Technology                                      |
| ---------------------- | ----------------------------------------------- |
| API Framework          | FastAPI                                         |
| Vector Search          | FAISS (Facebook AI Similarity Search)           |
| Embeddings             | Sentence Transformers (`all-MiniLM-L12-v2`)     |
| LLM                    | HuggingFace Transformers (`flan-t5-base`)       |
| Validation             | Pydantic v2                                     |
| Containerization       | Docker (multi-stage build)                      |
| CI/CD                  | GitHub Actions                                  |
| Testing                | pytest + coverage                               |

## License

MIT
