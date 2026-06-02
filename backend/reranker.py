"""Cross-encoder reranker."""
from loguru import logger

from backend.config import settings

_reranker_cache: dict = {}


def get_reranker(model_name: str | None = None):
    from sentence_transformers import CrossEncoder
    name = model_name or settings.RERANKER_MODEL
    if name not in _reranker_cache:
        logger.info(f"Loading reranker: {name}")
        _reranker_cache[name] = CrossEncoder(name, max_length=512)
    return _reranker_cache[name]


def rerank(query: str, chunks: list[dict], model_name: str | None = None) -> list[dict]:
    """
    Rerank chunks using a cross-encoder.
    Returns chunks sorted by rerank_score desc, with 'rerank_score' added.
    """
    if not chunks:
        return chunks
    try:
        reranker = get_reranker(model_name)
        pairs = [(query, c["text"]) for c in chunks]
        scores = reranker.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.debug(f"Reranked {len(chunks)} chunks")
        return reranked
    except Exception as e:
        logger.warning(f"Reranker failed ({e}), returning original order")
        return chunks
