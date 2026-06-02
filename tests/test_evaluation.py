"""Tests for evaluation metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from backend.evaluator import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    source_coverage,
    citation_accuracy,
    no_answer_accuracy,
    classify_failure,
    hallucination_risk,
)


# ─── Retrieval metrics ───────────────────────────────────────────────────────────

def test_precision_at_k_hit():
    retrieved = ["hr.pdf", "compliance.pdf", "sla.pdf"]
    assert precision_at_k(retrieved, "hr.pdf", k=3) == pytest.approx(1/3, abs=0.01)


def test_precision_at_k_no_hit():
    retrieved = ["compliance.pdf", "sla.pdf"]
    assert precision_at_k(retrieved, "hr.pdf", k=2) == 0.0


def test_recall_at_k_hit():
    retrieved = ["compliance.pdf", "hr.pdf"]
    assert recall_at_k(retrieved, "hr.pdf", k=5) == 1.0


def test_recall_at_k_miss():
    retrieved = ["compliance.pdf", "sla.pdf"]
    assert recall_at_k(retrieved, "hr.pdf", k=5) == 0.0


def test_mrr_first_position():
    retrieved = ["hr.pdf", "sla.pdf"]
    assert mean_reciprocal_rank(retrieved, "hr.pdf") == 1.0


def test_mrr_second_position():
    retrieved = ["sla.pdf", "hr.pdf"]
    assert mean_reciprocal_rank(retrieved, "hr.pdf") == pytest.approx(0.5)


def test_mrr_no_hit():
    retrieved = ["sla.pdf", "compliance.pdf"]
    assert mean_reciprocal_rank(retrieved, "hr.pdf") == 0.0


def test_ndcg_at_k():
    retrieved = ["hr.pdf", "sla.pdf"]
    score = ndcg_at_k(retrieved, "hr.pdf", k=2)
    assert 0.0 <= score <= 1.0


def test_source_coverage():
    retrieved = ["hr.pdf", "hr.pdf", "sla.pdf"]
    cov = source_coverage(retrieved, "hr.pdf")
    assert cov == pytest.approx(2/3, abs=0.01)


# ─── Answer metrics ──────────────────────────────────────────────────────────────

def test_citation_accuracy_correct():
    answer = "According to Document: hr.pdf, the policy is clear."
    retrieved = ["hr.pdf"]
    assert citation_accuracy(answer, retrieved) == 1.0


def test_citation_accuracy_no_citations():
    answer = "The policy states employees get 20 days."
    assert citation_accuracy(answer, []) == 1.0


def test_no_answer_accuracy_correct_refusal():
    answer = "The uploaded documents do not contain enough information."
    assert no_answer_accuracy(answer, answerable=False) == 1


def test_no_answer_accuracy_wrong_refusal():
    answer = "The uploaded documents do not contain enough information."
    assert no_answer_accuracy(answer, answerable=True) == 0


def test_no_answer_accuracy_correct_answer():
    answer = "The policy grants 20 days of leave."
    assert no_answer_accuracy(answer, answerable=True) == 1


# ─── Failure classifier ──────────────────────────────────────────────────────────

def test_classify_retrieval_miss():
    result = {"source_hit": 0, "answer_relevance": 0.5, "faithfulness": 0.5,
              "citation_accuracy": 1.0, "no_answer_correct": 1, "answerable": True,
              "completeness": 0.5}
    assert classify_failure(result) == "retrieval_miss"


def test_classify_correct():
    result = {"source_hit": 1, "answer_relevance": 0.8, "faithfulness": 0.8,
              "citation_accuracy": 0.9, "no_answer_correct": 1, "answerable": True,
              "completeness": 0.7}
    assert classify_failure(result) == "correct"


def test_classify_no_answer_failure():
    result = {"source_hit": 1, "answer_relevance": 0.8, "faithfulness": 0.8,
              "citation_accuracy": 0.9, "no_answer_correct": 0, "answerable": False,
              "completeness": 0.7}
    assert classify_failure(result) == "no_answer_failure"


def test_hallucination_risk_range():
    risk = hallucination_risk("Some answer text.", ["Some context text."])
    assert 0.0 <= risk <= 1.0
