"""Tests for the chunking module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from backend.chunking import fixed_chunks, section_aware_chunks


def test_fixed_chunks_basic():
    text = " ".join(["word"] * 200)
    chunks = fixed_chunks(text, size=50, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 50


def test_fixed_chunks_overlap():
    text = " ".join([str(i) for i in range(100)])
    chunks = fixed_chunks(text, size=20, overlap=5)
    # Check overlap: last words of chunk[i] should appear in chunk[i+1]
    if len(chunks) > 1:
        end_words = set(chunks[0].split()[-5:])
        start_words = set(chunks[1].split()[:10])
        assert end_words & start_words, "Expected overlap between consecutive chunks"


def test_fixed_chunks_empty():
    assert fixed_chunks("", size=100, overlap=10) == []


def test_section_aware_chunks():
    text = "INTRODUCTION\nThis is the intro.\n\nMETHODS\nThis is the methods section."
    sections = section_aware_chunks(text, size=50, overlap=5)
    assert len(sections) >= 1
    # Each item is (title, text)
    for title, body in sections:
        assert isinstance(body, str)


def test_chunk_sizes():
    text = " ".join(["token"] * 1000)
    for size in [300, 500, 800, 1000]:
        chunks = fixed_chunks(text, size=size, overlap=int(size * 0.1))
        for chunk in chunks:
            assert len(chunk.split()) <= size


def test_overlap_percentages():
    text = " ".join([str(i) for i in range(500)])
    for pct in [0.10, 0.15, 0.20]:
        size = 100
        overlap = int(size * pct)
        chunks = fixed_chunks(text, size=size, overlap=overlap)
        assert len(chunks) > 1
