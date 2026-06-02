"""
Evaluation metrics for retrieval and answer quality.
All metrics are heuristic / embedding-based (no external APIs needed).
"""
import math
import re
from typing import Any

import numpy as np
from loguru import logger

from backend.embeddings import embed_texts


# ─── Text utilities ──────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-9)
    b_n = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_n, b_n))


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ─── Retrieval metrics ───────────────────────────────────────────────────────────

def precision_at_k(retrieved_docs: list[str], expected_doc: str, k: int) -> float:
    top_k = retrieved_docs[:k]
    hits = sum(1 for d in top_k if expected_doc and expected_doc.lower() in d.lower())
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_docs: list[str], expected_doc: str, k: int) -> float:
    top_k = retrieved_docs[:k]
    hit = any(expected_doc and expected_doc.lower() in d.lower() for d in top_k)
    return 1.0 if hit else 0.0


def mean_reciprocal_rank(retrieved_docs: list[str], expected_doc: str) -> float:
    for i, d in enumerate(retrieved_docs, 1):
        if expected_doc and expected_doc.lower() in d.lower():
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], expected_doc: str, k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(retrieved_docs[:k], 1):
        rel = 1.0 if expected_doc and expected_doc.lower() in d.lower() else 0.0
        dcg += rel / math.log2(i + 1)
    idcg = 1.0 / math.log2(2)  # ideal: hit at rank 1
    return dcg / idcg if idcg > 0 else 0.0


def source_coverage(retrieved_docs: list[str], expected_doc: str) -> float:
    """Fraction of retrieved docs that match expected source."""
    if not retrieved_docs or not expected_doc:
        return 0.0
    hits = sum(1 for d in retrieved_docs if expected_doc.lower() in d.lower())
    return hits / len(retrieved_docs)


# ─── Answer metrics ──────────────────────────────────────────────────────────────

def answer_relevance(question: str, answer: str) -> float:
    """Semantic similarity between question and answer."""
    if not answer.strip():
        return 0.0
    embs = embed_texts([question, answer])
    return max(0.0, _cosine(embs[0], embs[1]))


def faithfulness(answer: str, context_chunks: list[str]) -> float:
    """
    Estimate faithfulness: average max similarity of answer sentences to context.
    """
    if not answer.strip() or not context_chunks:
        return 0.0
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0
    context_text = " ".join(context_chunks)
    embs = embed_texts(sentences + [context_text])
    sent_embs = embs[:-1]
    ctx_emb = embs[-1]
    sims = [_cosine(s, ctx_emb) for s in sent_embs]
    return float(np.mean(sims))


def answer_completeness(answer: str, expected_answer: str) -> float:
    """Token overlap + semantic similarity hybrid."""
    if not expected_answer.strip() or not answer.strip():
        return 0.0
    overlap = _token_overlap(answer, expected_answer)
    embs = embed_texts([answer, expected_answer])
    sem_sim = max(0.0, _cosine(embs[0], embs[1]))
    return 0.4 * overlap + 0.6 * sem_sim


def citation_accuracy(answer: str, retrieved_docs: list[str]) -> float:
    """Check if cited doc names in the answer match retrieved docs."""
    cited = re.findall(r"Document:\s*([\w\.\-]+)", answer, re.IGNORECASE)
    if not cited:
        return 1.0  # no citations to be wrong about
    hits = sum(
        1 for c in cited
        if any(c.lower() in d.lower() for d in retrieved_docs)
    )
    return hits / len(cited)


def hallucination_risk(answer: str, context_chunks: list[str]) -> float:
    """1 - faithfulness: higher means more likely hallucinated."""
    f = faithfulness(answer, context_chunks)
    return round(1.0 - f, 4)


def no_answer_accuracy(generated_answer: str, answerable: bool) -> int:
    """
    1 if no-answer decision was correct, 0 otherwise.
    answerable=False → expect a refusal
    answerable=True  → expect an actual answer
    """
    is_refusal = any(p in generated_answer.lower() for p in [
        "do not contain", "cannot answer", "not enough information",
        "not mentioned", "not available in", "unable to find",
    ])
    if not answerable and is_refusal:
        return 1
    if answerable and not is_refusal:
        return 1
    return 0


# ─── Failure classifier ──────────────────────────────────────────────────────────

FAILURE_CATEGORIES = [
    "retrieval_miss",
    "low_relevance_retrieval",
    "partial_context",
    "bad_chunking",
    "conflicting_sources",
    "unsupported_answer",
    "citation_mismatch",
    "no_answer_failure",
    "ambiguous_query",
    "metadata_filtering_failure",
    "correct",
]


def classify_failure(result: dict) -> str:
    """
    Classify a failed eval result into a failure category.
    result keys: source_hit, answer_relevance, faithfulness, citation_accuracy,
                 no_answer_correct, answerable, generated_answer, completeness
    """
    if result.get("no_answer_correct") == 0 and not result.get("answerable", True):
        return "no_answer_failure"
    if result.get("source_hit", 0) == 0:
        return "retrieval_miss"
    if result.get("answer_relevance", 1) < 0.3:
        return "low_relevance_retrieval"
    if result.get("faithfulness", 1) < 0.3:
        return "unsupported_answer"
    if result.get("citation_accuracy", 1) < 0.5:
        return "citation_mismatch"
    if result.get("completeness", 1) < 0.4:
        return "partial_context"
    return "correct"


# ─── Full evaluation of a single question ────────────────────────────────────────

def evaluate_single(
    question: str,
    expected_answer: str,
    expected_doc: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
    answerable: bool = True,
    k: int = 5,
) -> dict:
    retrieved_docs = [c.get("doc_name", "") for c in retrieved_chunks]
    context_texts = [c.get("text", "") for c in retrieved_chunks]

    prec = precision_at_k(retrieved_docs, expected_doc, k)
    rec = recall_at_k(retrieved_docs, expected_doc, k)
    mrr = mean_reciprocal_rank(retrieved_docs, expected_doc)
    ndcg = ndcg_at_k(retrieved_docs, expected_doc, k)
    src_cov = source_coverage(retrieved_docs, expected_doc)
    src_hit = int(rec > 0)

    ans_rel = answer_relevance(question, generated_answer)
    faith = faithfulness(generated_answer, context_texts)
    compl = answer_completeness(generated_answer, expected_answer)
    cit_acc = citation_accuracy(generated_answer, retrieved_docs)
    hall_risk = hallucination_risk(generated_answer, context_texts)
    na_acc = no_answer_accuracy(generated_answer, answerable)

    metrics = {
        "source_hit": src_hit,
        "precision_at_k": round(prec, 4),
        "recall_at_k": round(rec, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
        "source_coverage": round(src_cov, 4),
        "answer_relevance": round(ans_rel, 4),
        "faithfulness": round(faith, 4),
        "completeness": round(compl, 4),
        "citation_accuracy": round(cit_acc, 4),
        "hallucination_risk": round(hall_risk, 4),
        "no_answer_correct": na_acc,
        "answerable": int(answerable),
    }
    metrics["failure_category"] = classify_failure({**metrics, "answerable": answerable})
    return metrics
