"""Text chunking strategies."""
import re
import uuid
from datetime import datetime, timezone
from typing import Literal

from loguru import logger

from backend.config import settings
from database import db

ChunkStrategy = Literal["fixed", "overlap", "page_aware", "section_aware"]


def _token_count(text: str) -> int:
    """Rough word-based token estimate."""
    return len(text.split())


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def section_aware_chunks(text: str, size: int, overlap: int) -> list[tuple[str | None, str]]:
    """Split on headings, then apply fixed chunking within each section."""
    heading_pattern = re.compile(r"^(#{1,6}\s.+|[A-Z][A-Z\s]{3,}:?)$", re.MULTILINE)
    sections: list[tuple[str | None, str]] = []
    pos = 0
    current_title: str | None = None
    for m in heading_pattern.finditer(text):
        if m.start() > pos:
            sections.append((current_title, text[pos : m.start()].strip()))
        current_title = m.group().strip()
        pos = m.end()
    sections.append((current_title, text[pos:].strip()))

    result = []
    for title, body in sections:
        for chunk in fixed_chunks(body, size, overlap):
            result.append((title, chunk))
    return result


def chunk_document(
    doc_id: str,
    doc_name: str,
    pages: list[dict],
    strategy: ChunkStrategy = "overlap",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
    vector_store: str | None = None,
) -> list[dict]:
    """
    Chunk a document's pages and persist to SQLite.
    Returns list of chunk dicts ready for embedding.
    """
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP
    emb_model = embedding_model or settings.EMBEDDING_MODEL
    vs = vector_store or settings.VECTOR_STORE
    now = datetime.now(timezone.utc).isoformat()

    all_chunks: list[dict] = []
    chunk_index = 0

    for page_info in pages:
        page_num = page_info["page"]
        text = page_info["text"]

        if strategy == "page_aware":
            raw = [text]  # one chunk per page
            titles = [None]
        elif strategy == "section_aware":
            section_chunks = section_aware_chunks(text, size, overlap)
            raw = [c for _, c in section_chunks]
            titles = [t for t, _ in section_chunks]
        else:  # fixed / overlap (overlap handled by fixed_chunks)
            raw = fixed_chunks(text, size, overlap)
            titles = [None] * len(raw)

        for chunk_text, section_title in zip(raw, titles):
            if not chunk_text.strip():
                continue
            chunk_id = f"{doc_id}_{chunk_index}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "text": chunk_text,
                "page_number": page_num,
                "section_title": section_title,
                "chunk_index": chunk_index,
                "chunk_size": size,
                "overlap": overlap,
                "embedding_model": emb_model,
                "vector_store": vs,
                "created_at": now,
            })
            chunk_index += 1

    db.insert_chunks(all_chunks)
    db.update_document_status(doc_id, "chunks_ready", len(all_chunks))
    logger.info(f"Chunked doc {doc_id} → {len(all_chunks)} chunks (strategy={strategy})")
    return all_chunks
