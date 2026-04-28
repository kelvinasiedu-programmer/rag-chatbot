"""FastAPI application for the RAG Chatbot API."""

import logging
import os
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import Settings
from .rag_engine import RAGEngine
from .schemas import (
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

settings = Settings()

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate-limiting middleware
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding-window rate limiter keyed by client IP."""

    def __init__(self, app, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Prune timestamps outside the current window
        self._hits[client_ip] = [
            t for t in self._hits[client_ip] if now - t < self.window_seconds
        ]

        if len(self._hits[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        self._hits[client_ip].append(now)
        return await call_next(request)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Starting %s ...", settings.app_name)
    engine = RAGEngine(settings)
    engine.vector_store.load()
    logger.info("Ready — %d documents loaded", engine.vector_store.count)
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description="Production-grade Retrieval-Augmented Generation chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the frontend
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(_static_dir, "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return service health and basic stats."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        documents_loaded=engine.vector_store.count if engine else 0,
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Chat"])
async def query_documents(body: QueryRequest):
    """Ask a question about the loaded documents."""
    if not engine or engine.vector_store.count == 0:
        raise HTTPException(
            status_code=400, detail="No documents loaded. Upload a PDF first."
        )

    result = engine.query(body.question)
    return result


@app.post(
    "/api/v1/documents/upload", response_model=IngestResponse, tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and ingest it into the vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    size = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        while chunk := await file.read(8192):
            size += len(chunk)
            if size > settings.max_upload_size_mb * 1024 * 1024:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
                )
            tmp.write(chunk)
        tmp.close()

        count = engine.ingest_pdf(tmp.name)
        if count == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No extractable text found. The PDF may be scanned/image-only, "
                    "encrypted, or empty. Try a text-based PDF."
                ),
            )
        return IngestResponse(
            message=f"Processed {file.filename}",
            chunks_added=count,
            total_documents=engine.vector_store.count,
        )
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@app.get("/api/v1/documents", response_model=DocumentsResponse, tags=["Documents"])
async def get_document_stats():
    """Return the number of indexed document chunks."""
    return DocumentsResponse(
        total_documents=engine.vector_store.count if engine else 0,
        persist_dir=settings.vector_store_path,
    )


@app.delete("/api/v1/documents", tags=["Documents"])
async def clear_documents():
    """Remove all documents from the vector store."""
    if engine:
        engine.vector_store.clear()
    return {"message": "All documents cleared."}
