"""Tests for retrieval components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from backend.bm25_search import BM25Index
from backend.hybrid_retriever import _reciprocal_rank_fusion


# ─── BM25 ────────────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {"chunk_id": "doc1_0", "doc_id": "doc1", "doc_name": "hr.pdf",
     "text": "The annual leave policy grants 20 days to full-time employees.",
     "page_number": 1, "section_title": None},
    {"chunk_id": "doc1_1", "doc_id": "doc1", "doc_name": "hr.pdf",
     "text": "Sick leave is capped at 10 days per calendar year.",
     "page_number": 2, "section_title": None},
    {"chunk_id": "doc2_0", "doc_id": "doc2", "doc_name": "compliance.pdf",
     "text": "GDPR requires data retention for a minimum of 5 years.",
     "page_number": 1, "section_title": None},
]


def test_bm25_build_and_search():
    idx = BM25Index()
    idx.build(SAMPLE_CHUNKS)
    results = idx.search("annual leave", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "doc1_0"


def test_bm25_no_results_empty_index():
    idx = BM25Index()
    idx.bm25 = None
    idx.chunk_ids = []
    results = idx.search("anything")
    assert results == []


def test_bm25_filter():
    idx = BM25Index()
    idx.build(SAMPLE_CHUNKS)
    results = idx.search("leave", top_k=5, filters={"doc_name": "hr.pdf"})
    assert all(r["doc_name"] == "hr.pdf" for r in results)


# ─── RRF Fusion ──────────────────────────────────────────────────────────────────

def test_rrf_fusion_combines_results():
    list1 = [
        {"chunk_id": "a", "text": "AAA", "doc_name": "d1", "score": 0.9},
        {"chunk_id": "b", "text": "BBB", "doc_name": "d1", "score": 0.8},
    ]
    list2 = [
        {"chunk_id": "b", "text": "BBB", "doc_name": "d1", "score": 0.95},
        {"chunk_id": "c", "text": "CCC", "doc_name": "d2", "score": 0.7},
    ]
    fused = _reciprocal_rank_fusion([list1, list2])
    ids = [r["chunk_id"] for r in fused]
    # 'b' appears in both lists, should score highest
    assert ids[0] == "b"
    assert len(fused) == 3


def test_rrf_single_list():
    list1 = [
        {"chunk_id": "x", "text": "X", "doc_name": "d", "score": 1.0},
    ]
    fused = _reciprocal_rank_fusion([list1])
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "x"


def test_rrf_empty():
    assert _reciprocal_rank_fusion([]) == []
    assert _reciprocal_rank_fusion([[]]) == []
