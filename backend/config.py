"""Centralised settings – reads from .env / environment."""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    LLM_BACKEND: str = "ollama"          # ollama | huggingface | stub
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    HF_MODEL: str = "google/flan-t5-base"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector store
    VECTOR_STORE: str = "faiss"          # faiss | chroma

    # Retrieval
    TOP_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    USE_RERANKER: bool = True
    USE_HYBRID: bool = True
    HYBRID_ALPHA: float = 0.5

    # Reranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Database
    SQLITE_DB_PATH: str = "data/rag_eval.db"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
