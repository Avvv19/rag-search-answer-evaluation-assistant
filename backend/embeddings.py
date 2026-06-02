"""Embedding model wrapper (sentence-transformers)."""
from functools import lru_cache
from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from backend.config import settings

SUPPORTED_MODELS = [
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-small-en",
    "BAAI/bge-base-en",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
]

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or settings.EMBEDDING_MODEL
    if name not in _model_cache:
        logger.info(f"Loading embedding model: {name}")
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def embed_texts(texts: List[str], model_name: str | None = None) -> np.ndarray:
    """Return (N, D) float32 embeddings."""
    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str, model_name: str | None = None) -> np.ndarray:
    """Return (D,) float32 embedding for a single query."""
    return embed_texts([query], model_name)[0]
