"""
Answer generation module.
Supports three backends: ollama | huggingface | stub
"""
import re
import time
from typing import Any

from loguru import logger

from backend.config import settings

# ─── Prompt builder ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer questions using ONLY the provided document excerpts.
If the excerpts do not contain enough information, say:
"The uploaded documents do not contain enough information to answer this confidently."
Never fabricate information or use general knowledge.
Always cite the source document and page number for every claim."""

def _build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}] Document: {c.get('doc_name','?')} | "
            f"Page: {c.get('page_number','?')} | Chunk: {c.get('chunk_id','?')}\n"
            f"{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== Document Excerpts ===\n{context}\n\n"
        f"=== Question ===\n{query}\n\n"
        f"=== Answer ==="
    )


# ─── No-answer detection ─────────────────────────────────────────────────────────

NO_ANSWER_PHRASES = [
    "do not contain enough",
    "cannot answer",
    "not enough information",
    "no information",
    "unable to find",
    "not mentioned",
    "not provided",
    "not available in",
]

def _is_no_answer(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in NO_ANSWER_PHRASES)


def _evidence_strength(chunks: list[dict], min_score: float = 0.3) -> str:
    if not chunks:
        return "none"
    top_score = max(c.get("rerank_score", c.get("score", 0)) for c in chunks)
    if top_score >= 0.7:
        return "strong"
    if top_score >= min_score:
        return "moderate"
    return "weak"


# ─── Backends ────────────────────────────────────────────────────────────────────

def _generate_ollama(prompt: str) -> str:
    import httpx
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    resp = httpx.post(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _generate_hf(prompt: str) -> str:
    from transformers import pipeline
    if not hasattr(_generate_hf, "_pipe"):
        logger.info(f"Loading HF pipeline: {settings.HF_MODEL}")
        _generate_hf._pipe = pipeline(
            "text2text-generation" if "t5" in settings.HF_MODEL.lower() else "text-generation",
            model=settings.HF_MODEL,
            max_new_tokens=256,
        )
    result = _generate_hf._pipe(prompt, truncation=True)
    if isinstance(result, list) and result:
        return (result[0].get("generated_text") or result[0].get("summary_text", "")).strip()
    return ""


def _generate_stub(prompt: str) -> str:
    """Deterministic stub for testing without a real LLM."""
    return (
        "Based on the provided document excerpts, the answer is available in the context. "
        "[STUB RESPONSE – configure LLM_BACKEND in .env for real answers]"
    )


_BACKENDS = {
    "ollama": _generate_ollama,
    "huggingface": _generate_hf,
    "stub": _generate_stub,
}


# ─── Public API ──────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[dict],
    backend: str | None = None,
    min_evidence_score: float = 0.25,
    min_chunks: int = 1,
) -> dict:
    """
    Generate a grounded answer.

    Returns:
    {
      answer, is_no_answer, evidence_strength,
      sources: [{doc_name, page_number, chunk_id, score, text_snippet}],
      warning, latency_ms, backend_used
    }
    """
    t0 = time.time()
    b = backend or settings.LLM_BACKEND
    gen_fn = _BACKENDS.get(b, _generate_stub)

    # Check evidence quality before calling LLM
    strength = _evidence_strength(chunks, min_score=min_evidence_score)
    top_chunks = [c for c in chunks if c.get("rerank_score", c.get("score", 0)) >= min_evidence_score]

    if not top_chunks or strength == "none":
        answer = "The uploaded documents do not contain enough information to answer this confidently."
        is_no_answer = True
    else:
        prompt = _build_prompt(query, top_chunks[:5])
        try:
            answer = gen_fn(prompt)
        except Exception as e:
            logger.error(f"LLM backend '{b}' error: {e}")
            answer = f"[LLM error: {e}] Falling back: " + _generate_stub(prompt)
        is_no_answer = _is_no_answer(answer)

    latency_ms = (time.time() - t0) * 1000
    sources = [
        {
            "doc_name": c.get("doc_name", ""),
            "page_number": c.get("page_number", ""),
            "chunk_id": c.get("chunk_id", ""),
            "score": round(c.get("rerank_score", c.get("score", 0)), 4),
            "retrieval_method": c.get("retrieval_method", ""),
            "text_snippet": c["text"][:300],
        }
        for c in (top_chunks or chunks)[:5]
    ]

    warning = None
    if strength == "weak":
        warning = "⚠️ Evidence is weak. Answer may be unreliable."
    elif strength == "moderate":
        warning = "ℹ️ Moderate evidence. Verify against source documents."

    return {
        "answer": answer,
        "is_no_answer": is_no_answer,
        "evidence_strength": strength,
        "sources": sources,
        "warning": warning,
        "latency_ms": round(latency_ms, 1),
        "backend_used": b,
    }
