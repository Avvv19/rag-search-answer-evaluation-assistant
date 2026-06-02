"""FastAPI backend – optional REST API layer."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from backend.pipeline import run_ingestion_pipeline, run_query_pipeline
from backend.config import settings
from database import db

app = FastAPI(
    title="RAG Evaluation Assistant API",
    version="1.0.0",
    description="Enterprise RAG system with retrieval evaluation and hallucination detection.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup ─────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    db.init_db()
    logger.info("RAG API started")


# ─── Health ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ─── Documents ───────────────────────────────────────────────────────────────────

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    strategy: str = Form("overlap"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    embedding_model: str = Form("all-MiniLM-L6-v2"),
    vector_store: str = Form("faiss"),
):
    file_bytes = await file.read()
    try:
        result = run_ingestion_pipeline(
            file_bytes=file_bytes,
            filename=file.filename,
            category=category,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            vector_store_backend=vector_store,
        )
        return result
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    return db.list_documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    db.delete_document(doc_id)
    return {"deleted": doc_id}


# ─── Query ───────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    use_hybrid: Optional[bool] = None
    use_reranker: Optional[bool] = None
    embedding_model: Optional[str] = None
    vector_store: Optional[str] = None
    llm_backend: Optional[str] = None
    filters: Optional[dict] = None


@app.post("/query")
def query(req: QueryRequest):
    try:
        result = run_query_pipeline(
            query=req.query,
            top_k=req.top_k,
            use_hybrid=req.use_hybrid,
            use_reranker=req.use_reranker,
            embedding_model=req.embedding_model,
            vector_store_backend=req.vector_store,
            llm_backend=req.llm_backend,
            filters=req.filters,
        )
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Evaluation ──────────────────────────────────────────────────────────────────

@app.get("/eval/questions")
def list_eval_questions():
    return db.list_eval_questions()


@app.get("/eval/results")
def list_eval_results(run_id: Optional[str] = None):
    return db.get_eval_results(run_id)


# ─── Feedback ────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int
    comment: str = ""
    helpful: Optional[bool] = None


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    from backend.feedback import save_feedback
    save_feedback(req.query, req.answer, req.rating, req.comment, req.helpful)
    return {"status": "saved"}


# ─── Settings ────────────────────────────────────────────────────────────────────

@app.get("/settings")
def get_settings():
    return {
        "LLM_BACKEND": settings.LLM_BACKEND,
        "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        "VECTOR_STORE": settings.VECTOR_STORE,
        "TOP_K": settings.TOP_K,
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "CHUNK_OVERLAP": settings.CHUNK_OVERLAP,
        "USE_RERANKER": settings.USE_RERANKER,
        "USE_HYBRID": settings.USE_HYBRID,
    }
