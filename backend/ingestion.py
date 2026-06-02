"""Document ingestion: parse → clean → store metadata."""
import hashlib
import io
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

# ─── Optional imports (graceful fallback) ───────────────────────────────────────
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

import pandas as pd

from database import db


# ─── Text cleaning ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and strip junk."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)   # non-ASCII
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)
    return text.strip()


def is_low_quality(text: str, min_words: int = 10) -> bool:
    words = text.split()
    if len(words) < min_words:
        return True
    # More than 50 % numeric/punctuation → likely garbled
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(len(text), 1) < 0.3:
        return True
    return False


# ─── Parsers ─────────────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Return list of {page, text} dicts."""
    pages = []
    if HAS_FITZ:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page": page_num, "text": clean_text(text)})
        doc.close()
    elif HAS_PYPDF:
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": clean_text(text)})
    else:
        raise ImportError("Install PyMuPDF or pypdf to process PDFs")
    return pages


def parse_docx(file_bytes: bytes) -> list[dict]:
    if not HAS_DOCX:
        raise ImportError("Install python-docx to process DOCX files")
    doc = DocxDocument(io.BytesIO(file_bytes))
    pages: list[dict] = []
    current: list[str] = []
    page = 1
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            current.append(t)
        # Simulate page break heuristic every ~50 paragraphs
        if len(current) >= 50:
            pages.append({"page": page, "text": clean_text(" ".join(current))})
            current = []
            page += 1
    if current:
        pages.append({"page": page, "text": clean_text(" ".join(current))})
    return pages


def parse_txt(file_bytes: bytes) -> list[dict]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [{"page": 1, "text": clean_text(text)}]


def parse_html(file_bytes: bytes) -> list[dict]:
    if not HAS_BS4:
        raise ImportError("Install beautifulsoup4 to process HTML")
    soup = BeautifulSoup(file_bytes, "html.parser")
    text = soup.get_text(separator=" ")
    return [{"page": 1, "text": clean_text(text)}]


def parse_csv(file_bytes: bytes) -> list[dict]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    text = df.to_string(index=False)
    return [{"page": 1, "text": clean_text(text)}]


def parse_excel(file_bytes: bytes) -> list[dict]:
    sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    pages = []
    for i, (name, df) in enumerate(sheets.items(), start=1):
        pages.append({"page": i, "text": clean_text(f"Sheet: {name}\n{df.to_string(index=False)}")})
    return pages


PARSERS = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".txt":  parse_txt,
    ".html": parse_html,
    ".htm":  parse_html,
    ".csv":  parse_csv,
    ".xlsx": parse_excel,
    ".xls":  parse_excel,
}


# ─── Public API ──────────────────────────────────────────────────────────────────

def ingest_document(
    file_bytes: bytes,
    filename: str,
    category: str = "general",
) -> dict:
    """
    Parse a document, store metadata in SQLite, and return:
    {doc_id, name, file_type, pages: [{page, text}], raw_pages_count}
    """
    suffix = Path(filename).suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    doc_id = hashlib.md5(file_bytes).hexdigest()[:16]
    now = datetime.now(timezone.utc).isoformat()

    # Call parser with correct arguments
    if suffix == ".pdf":
        pages = parser(file_bytes, filename)
    else:
        pages = parser(file_bytes)

    # Filter low-quality pages
    pages = [p for p in pages if not is_low_quality(p["text"])]

    db.insert_document({
        "doc_id": doc_id,
        "name": filename,
        "file_type": suffix.lstrip("."),
        "source_path": filename,
        "category": category,
        "upload_date": now,
        "num_chunks": 0,
        "status": "pending",
    })

    logger.info(f"Ingested '{filename}' → doc_id={doc_id}, pages={len(pages)}")
    return {
        "doc_id": doc_id,
        "name": filename,
        "file_type": suffix.lstrip("."),
        "pages": pages,
        "raw_pages_count": len(pages),
    }
