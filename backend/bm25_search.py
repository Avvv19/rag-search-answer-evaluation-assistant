"""BM25 keyword search over indexed chunks."""
import pickle
from pathlib import Path

from loguru import logger
from rank_bm25 import BM25Okapi

from database import db

BM25_INDEX_PATH = Path("data/vector_indexes/bm25.pkl")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self._load()

    def _load(self):
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                state = pickle.load(f)
            self.bm25 = state["bm25"]
            self.chunk_ids = state["chunk_ids"]
            logger.info(f"BM25: loaded index with {len(self.chunk_ids)} docs")

    def _save(self):
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    def build(self, chunks: list[dict] | None = None):
        """Build BM25 index from provided chunks or all DB chunks."""
        if chunks is None:
            chunks = db.get_all_chunks()
        if not chunks:
            logger.warning("BM25: no chunks to index")
            return
        corpus = [_tokenize(c["text"]) for c in chunks]
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self._save()
        logger.info(f"BM25: built index with {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        if self.bm25 is None or not self.chunk_ids:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            if score <= 0:
                break
            chunk_id = self.chunk_ids[idx]
            chunk = db.get_chunk_by_id(chunk_id)
            if chunk is None:
                continue
            if filters and not _match_filters(chunk, filters):
                continue
            results.append({**chunk, "score": float(score), "retrieval_method": "bm25"})
            if len(results) >= top_k:
                break
        return results


def _match_filters(meta: dict, filters: dict) -> bool:
    for k, v in filters.items():
        if str(meta.get(k, "")) != str(v):
            return False
    return True


_bm25_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


def bm25_search(query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
    return get_bm25_index().search(query, top_k, filters)


def rebuild_bm25(chunks: list[dict] | None = None):
    idx = get_bm25_index()
    idx.build(chunks)
