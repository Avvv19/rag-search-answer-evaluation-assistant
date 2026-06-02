"""FAISS and ChromaDB vector store abstraction."""
import json
import pickle
from pathlib import Path
from typing import List, Literal

import numpy as np
from loguru import logger

from backend.config import settings
from backend.embeddings import embed_texts, embed_query

VECTOR_DIR = Path("data/vector_indexes")
VectorBackend = Literal["faiss", "chroma"]


# ─── FAISS ───────────────────────────────────────────────────────────────────────

class FAISSStore:
    def __init__(self, index_path: Path | None = None):
        import faiss
        self._faiss = faiss
        self.index_path = index_path or VECTOR_DIR / "faiss.index"
        self.meta_path = self.index_path.with_suffix(".meta.pkl")
        self.index: "faiss.Index | None" = None
        self.metadata: list[dict] = []
        self._load()

    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = self._faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"FAISS: loaded {self.index.ntotal} vectors")
        else:
            self.index = None
            self.metadata = []

    def _save(self):
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add(self, texts: list[str], metadatas: list[dict], model_name: str | None = None):
        embeddings = embed_texts(texts, model_name)
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = self._faiss.IndexFlatIP(dim)  # Inner product (cosine on L2-norm)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        self._save()
        logger.info(f"FAISS: added {len(texts)} vectors, total={self.index.ntotal}")

    def search(self, query: str, top_k: int = 5, model_name: str | None = None,
               filters: dict | None = None) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_emb = embed_query(query, model_name).reshape(1, -1)
        scores, indices = self.index.search(q_emb, min(top_k * 3, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if filters and not _match_filters(meta, filters):
                continue
            results.append({**meta, "score": float(score), "retrieval_method": "vector"})
            if len(results) >= top_k:
                break
        return results

    def delete_by_doc(self, doc_id: str):
        """Remove all vectors for a document (rebuild index)."""
        keep_mask = [m.get("doc_id") != doc_id for m in self.metadata]
        kept_meta = [m for m, keep in zip(self.metadata, keep_mask) if keep]
        if not kept_meta:
            self.index = None
            self.metadata = []
            self._save()
            return
        # Rebuild
        import faiss
        texts = [m["text"] for m in kept_meta]
        embeddings = embed_texts(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = kept_meta
        self._save()


# ─── ChromaDB ────────────────────────────────────────────────────────────────────

class ChromaStore:
    def __init__(self):
        import chromadb
        persist_dir = str(VECTOR_DIR / "chroma")
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="rag_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts: list[str], metadatas: list[dict], model_name: str | None = None):
        embeddings = embed_texts(texts, model_name).tolist()
        ids = [m["chunk_id"] for m in metadatas]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{k: str(v) if v is not None else "" for k, v in m.items() if k != "text"}
                       for m in metadatas],
        )
        logger.info(f"Chroma: upserted {len(texts)} chunks")

    def search(self, query: str, top_k: int = 5, model_name: str | None = None,
               filters: dict | None = None) -> list[dict]:
        q_emb = embed_query(query, model_name).tolist()
        where = None
        if filters:
            where = {k: {"$eq": str(v)} for k, v in filters.items()}
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
        )
        results = []
        for doc, meta, dist, id_ in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], res["ids"][0]
        ):
            results.append({
                **meta,
                "text": doc,
                "chunk_id": id_,
                "score": 1 - dist,
                "retrieval_method": "vector",
            })
        return results

    def delete_by_doc(self, doc_id: str):
        existing = self.collection.get(where={"doc_id": {"$eq": doc_id}})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _match_filters(meta: dict, filters: dict) -> bool:
    for k, v in filters.items():
        if str(meta.get(k, "")) != str(v):
            return False
    return True


# ─── Factory ─────────────────────────────────────────────────────────────────────

_store_cache: dict[str, FAISSStore | ChromaStore] = {}


def get_store(backend: str | None = None) -> FAISSStore | ChromaStore:
    b = backend or settings.VECTOR_STORE
    if b not in _store_cache:
        if b == "faiss":
            _store_cache[b] = FAISSStore()
        elif b == "chroma":
            _store_cache[b] = ChromaStore()
        else:
            raise ValueError(f"Unknown vector store: {b}")
    return _store_cache[b]


def index_chunks(chunks: list[dict], model_name: str | None = None, backend: str | None = None):
    store = get_store(backend)
    texts = [c["text"] for c in chunks]
    store.add(texts, chunks, model_name)


def vector_search(query: str, top_k: int | None = None, model_name: str | None = None,
                  backend: str | None = None, filters: dict | None = None) -> list[dict]:
    k = top_k or settings.TOP_K
    return get_store(backend).search(query, k, model_name, filters)
