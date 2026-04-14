"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    app_name: str = "RAG Chatbot API"
    debug: bool = False

    # Models
    embedding_model: str = "all-MiniLM-L12-v2"
    llm_model: str = "google/flan-t5-base"

    # Vector Store
    vector_store_path: str = "./data/vector_store"

    # RAG Parameters
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    max_tokens: int = 200

    # Security
    rate_limit_requests: int = 10
    rate_limit_window: int = 60
    cors_origins: list[str] = ["*"]
    max_upload_size_mb: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
