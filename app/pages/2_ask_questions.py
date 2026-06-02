"""Page 2 – Ask natural-language questions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from database import db
from backend.pipeline import run_query_pipeline
from backend.feedback import save_feedback
from backend.config import settings
from backend.embeddings import SUPPORTED_MODELS
from datetime import datetime, timezone

st.set_page_config(page_title="Ask Questions", page_icon="💬", layout="wide")
st.title("💬 Ask Questions")

# ─── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    top_k = st.slider("Top-K chunks", 1, 20, settings.TOP_K)
    use_hybrid = st.checkbox("Hybrid retrieval (vector + BM25)", value=settings.USE_HYBRID)
    use_reranker = st.checkbox("Cross-encoder reranker", value=settings.USE_RERANKER)
    embedding_model = st.selectbox("Embedding model", SUPPORTED_MODELS,
                                   index=SUPPORTED_MODELS.index(settings.EMBEDDING_MODEL)
                                   if settings.EMBEDDING_MODEL in SUPPORTED_MODELS else 0)
    vector_store = st.selectbox("Vector store", ["faiss", "chroma"],
                                index=0 if settings.VECTOR_STORE == "faiss" else 1)
    llm_backend = st.selectbox("LLM backend", ["ollama", "huggingface", "stub"],
                               index=["ollama", "huggingface", "stub"].index(settings.LLM_BACKEND)
                               if settings.LLM_BACKEND in ["ollama", "huggingface", "stub"] else 2)

    st.markdown("---")
    st.subheader("🔎 Metadata Filters (optional)")
    docs = db.list_documents()
    doc_names = ["(all)"] + [d["name"] for d in docs]
    filter_doc = st.selectbox("Filter by document", doc_names)
    categories = ["(all)"] + list(set(d.get("category", "") for d in docs if d.get("category")))
    filter_cat = st.selectbox("Filter by category", categories)

filters = {}
if filter_doc != "(all)":
    filters["doc_name"] = filter_doc
if filter_cat != "(all)":
    filters["category"] = filter_cat

# ─── Question input ───────────────────────────────────────────────────────────────
query = st.text_area("Ask a question about your documents:", height=80,
                     placeholder="e.g. What is the vacation policy for new employees?")

col_ask, col_clear = st.columns([1, 4])
ask = col_ask.button("🔍 Ask", type="primary")

if ask and query.strip():
    if not docs:
        st.warning("No documents indexed. Please upload documents first.")
    else:
        with st.spinner("Retrieving and generating answer …"):
            result = run_query_pipeline(
                query=query,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                embedding_model=embedding_model,
                vector_store_backend=vector_store,
                llm_backend=llm_backend,
                filters=filters or None,
            )

        # ─── Answer card ─────────────────────────────────────────────────────────
        if result.get("is_no_answer"):
            st.error("🚫 " + result["answer"])
        else:
            st.success("✅ Answer")
            st.markdown(result["answer"])

        if result.get("warning"):
            st.warning(result["warning"])

        ev = result.get("evidence_strength", "unknown")
        col1, col2, col3 = st.columns(3)
        col1.metric("Evidence Strength", ev.upper())
        col2.metric("Latency", f"{result.get('total_latency_ms', 0):.0f} ms")
        col3.metric("LLM Backend", result.get("backend_used", "?"))

        # ─── Sources ──────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📎 Retrieved Sources")
        for i, src in enumerate(result.get("sources", []), 1):
            with st.expander(
                f"[{i}] {src['doc_name']} — Page {src['page_number']} "
                f"(score: {src['score']:.3f}, method: {src['retrieval_method']})"
            ):
                st.caption(f"Chunk ID: `{src['chunk_id']}`")
                st.text(src["text_snippet"])

        # ─── All retrieved chunks ─────────────────────────────────────────────────
        all_chunks = result.get("retrieved_chunks", [])
        if all_chunks:
            with st.expander(f"Show all {len(all_chunks)} retrieved chunks with scores"):
                import pandas as pd
                rows = []
                for c in all_chunks:
                    rows.append({
                        "chunk_id": c.get("chunk_id"),
                        "doc_name": c.get("doc_name"),
                        "page": c.get("page_number"),
                        "vector_score": round(c.get("score", 0), 4),
                        "rerank_score": round(c.get("rerank_score", 0), 4),
                        "method": c.get("retrieval_method"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ─── Feedback ─────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("👍 Was this helpful?")
        fb_col1, fb_col2 = st.columns(2)
        rating = fb_col1.slider("Rating (1–5)", 1, 5, 3)
        comment = fb_col2.text_input("Comment (optional)")
        helpful = st.radio("Helpful?", ["Yes", "No", "Partially"], horizontal=True)
        if st.button("Submit feedback"):
            save_feedback(
                query=query,
                answer=result["answer"],
                rating=rating,
                comment=comment,
                helpful=helpful == "Yes",
            )
            st.success("Feedback saved!")

elif ask and not query.strip():
    st.warning("Please enter a question.")
