"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    app_name: str = "RAG Chatbot API"
    debug: bool = False

    # Models
    embedding_model: str = "all-MiniLM-L12-v2"
    # Backend toggle: "local" (flan-t5, free) or "anthropic" (Claude, API key + credits required)
    llm_backend: str = "local"
    local_llm_model: str = "google/flan-t5-base"
    anthropic_llm_model: str = "claude-haiku-4-5"
    anthropic_api_key: str | None = None

    # Vector Store
    vector_store_path: str = "./data/vector_store"

    # RAG Parameters
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    max_tokens: int = 512

    # Security
    rate_limit_requests: int = 10
    rate_limit_window: int = 60
    cors_origins: list[str] = ["*"]
    max_upload_size_mb: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
