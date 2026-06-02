"""Hybrid retrieval: RRF fusion of vector search + BM25."""
from loguru import logger

from backend.config import settings
from backend.vector_store import vector_search
from backend.bm25_search import bm25_search


def _reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Combine multiple ranked lists via Reciprocal Rank Fusion.
    k=60 is the standard constant.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    output = []
    for cid, rrf_score in fused:
        item = {**chunk_map[cid], "rrf_score": rrf_score}
        output.append(item)
    return output


def hybrid_search(
    query: str,
    top_k: int | None = None,
    alpha: float | None = None,
    model_name: str | None = None,
    backend: str | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """
    Hybrid search: vector + BM25 fused with RRF.
    alpha is kept for API compatibility (RRF doesn't use it directly).
    Returns top_k results with combined ranking.
    """
    k = top_k or settings.TOP_K
    fetch_k = k * 3  # over-fetch before fusion

    vec_results = vector_search(query, fetch_k, model_name, backend, filters)
    bm25_results = bm25_search(query, fetch_k, filters)

    fused = _reciprocal_rank_fusion([vec_results, bm25_results])

    # Annotate retrieval method
    vec_ids = {r["chunk_id"] for r in vec_results}
    bm25_ids = {r["chunk_id"] for r in bm25_results}
    for item in fused:
        cid = item["chunk_id"]
        if cid in vec_ids and cid in bm25_ids:
            item["retrieval_method"] = "hybrid"
        elif cid in vec_ids:
            item["retrieval_method"] = "vector"
        else:
            item["retrieval_method"] = "bm25"

    logger.debug(f"Hybrid search: {len(fused)} fused results for '{query[:60]}'")
    return fused[:k]


def retrieve(
    query: str,
    top_k: int | None = None,
    use_hybrid: bool | None = None,
    alpha: float | None = None,
    model_name: str | None = None,
    backend: str | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Unified retrieval entry-point."""
    do_hybrid = use_hybrid if use_hybrid is not None else settings.USE_HYBRID
    if do_hybrid:
        return hybrid_search(query, top_k, alpha, model_name, backend, filters)
    return vector_search(query, top_k or settings.TOP_K, model_name, backend, filters)
