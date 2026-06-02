"""
High-level pipeline: ingest → chunk → embed → index → retrieve → rerank → generate.
This module provides the single entry-point used by both the Streamlit UI and FastAPI.
"""
import time
from datetime import datetime, timezone

from loguru import logger

from backend.config import settings
from backend.ingestion import ingest_document
from backend.chunking import chunk_document
from backend.vector_store import index_chunks
from backend.bm25_search import rebuild_bm25
from backend.hybrid_retriever import retrieve
from backend.reranker import rerank
from backend.generator import generate_answer
from database import db


def run_ingestion_pipeline(
    file_bytes: bytes,
    filename: str,
    category: str = "general",
    strategy: str = "overlap",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
    vector_store_backend: str | None = None,
) -> dict:
    """Full ingestion: parse → chunk → embed → vector-index → BM25-index."""
    doc = ingest_document(file_bytes, filename, category)
    doc_id = doc["doc_id"]
    pages = doc["pages"]

    if not pages:
        db.update_document_status(doc_id, "error")
        raise ValueError(f"No usable text extracted from '{filename}'")

    chunks = chunk_document(
        doc_id=doc_id,
        doc_name=filename,
        pages=pages,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        vector_store=vector_store_backend,
    )

    index_chunks(chunks, model_name=embedding_model, backend=vector_store_backend)
    rebuild_bm25()

    db.update_document_status(doc_id, "indexed", len(chunks))
    logger.info(f"Pipeline complete for '{filename}': {len(chunks)} chunks indexed")
    return {**doc, "num_chunks": len(chunks), "status": "indexed"}


def run_query_pipeline(
    query: str,
    top_k: int | None = None,
    use_hybrid: bool | None = None,
    use_reranker: bool | None = None,
    embedding_model: str | None = None,
    vector_store_backend: str | None = None,
    llm_backend: str | None = None,
    filters: dict | None = None,
) -> dict:
    """Full query pipeline: retrieve → rerank → generate."""
    t0 = time.time()

    chunks = retrieve(
        query=query,
        top_k=top_k or settings.TOP_K,
        use_hybrid=use_hybrid if use_hybrid is not None else settings.USE_HYBRID,
        model_name=embedding_model,
        backend=vector_store_backend,
        filters=filters,
    )

    do_rerank = use_reranker if use_reranker is not None else settings.USE_RERANKER
    if do_rerank and chunks:
        chunks = rerank(query, chunks)

    result = generate_answer(query, chunks, backend=llm_backend)
    result["retrieved_chunks"] = chunks
    result["total_latency_ms"] = round((time.time() - t0) * 1000, 1)

    # Log query
    db.log_query({
        "query": query,
        "answer": result["answer"],
        "retrieved_chunks": [{"chunk_id": c.get("chunk_id"), "doc_name": c.get("doc_name")} for c in chunks],
        "latency_ms": result["total_latency_ms"],
        "config_snapshot": {
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_reranker": do_rerank,
            "embedding_model": embedding_model,
            "llm_backend": llm_backend,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    return result
