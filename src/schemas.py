"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to ask about the loaded documents",
    )


class SourceDocument(BaseModel):
    text: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    total_documents: int


class DocumentsResponse(BaseModel):
    total_documents: int
    persist_dir: str | None


class HealthResponse(BaseModel):
    status: str
    version: str
    documents_loaded: int
