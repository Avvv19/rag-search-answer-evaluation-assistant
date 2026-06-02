"""Page 1 – Upload & index documents."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from database import db
from backend.pipeline import run_ingestion_pipeline
from backend.embeddings import SUPPORTED_MODELS

st.set_page_config(page_title="Upload Documents", page_icon="📄", layout="wide")
st.title("📄 Upload Documents")

# ─── Sidebar settings ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ingestion Settings")
    category = st.text_input("Document category", value="general")
    strategy = st.selectbox(
        "Chunking strategy",
        ["overlap", "fixed", "page_aware", "section_aware"],
    )
    chunk_size = st.select_slider(
        "Chunk size (tokens)",
        options=[300, 500, 800, 1000],
        value=500,
    )
    overlap_pct = st.select_slider(
        "Chunk overlap %",
        options=[10, 15, 20],
        value=10,
    )
    chunk_overlap = int(chunk_size * overlap_pct / 100)
    st.caption(f"→ Overlap tokens: {chunk_overlap}")

    embedding_model = st.selectbox("Embedding model", SUPPORTED_MODELS)
    vector_store = st.selectbox("Vector store", ["faiss", "chroma"])

# ─── Upload ──────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT, CSV, XLSX, HTML)",
    type=["pdf", "docx", "txt", "csv", "xlsx", "xls", "html", "htm"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("🚀 Index uploaded files", type="primary"):
        results = []
        for uf in uploaded_files:
            with st.spinner(f"Indexing {uf.name} …"):
                try:
                    res = run_ingestion_pipeline(
                        file_bytes=uf.read(),
                        filename=uf.name,
                        category=category,
                        strategy=strategy,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        embedding_model=embedding_model,
                        vector_store_backend=vector_store,
                    )
                    results.append({"file": uf.name, "chunks": res["num_chunks"], "status": "✅ indexed"})
                except Exception as e:
                    results.append({"file": uf.name, "chunks": 0, "status": f"❌ {e}"})

        st.success("Indexing complete!")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

# ─── Document list ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📚 Indexed Documents")

docs = db.list_documents()
if not docs:
    st.info("No documents indexed yet. Upload some files above.")
else:
    df = pd.DataFrame(docs)
    display_cols = ["doc_id", "name", "file_type", "category", "upload_date", "num_chunks", "status"]
    df_display = df[[c for c in display_cols if c in df.columns]]
    st.dataframe(df_display, use_container_width=True)

    st.markdown("---")
    st.subheader("🗑️ Delete Document")
    doc_names = {d["name"]: d["doc_id"] for d in docs}
    to_delete = st.selectbox("Select document to delete", list(doc_names.keys()))
    if st.button("Delete", type="secondary"):
        doc_id = doc_names[to_delete]
        db.delete_document(doc_id)
        # Also clear from vector stores
        try:
            from backend.vector_store import get_store
            get_store().delete_by_doc(doc_id)
        except Exception:
            pass
        from backend.bm25_search import rebuild_bm25
        rebuild_bm25()
        st.success(f"Deleted '{to_delete}' and re-built indexes.")
        st.rerun()

# ─── Stats ───────────────────────────────────────────────────────────────────────
if docs:
    total_chunks = sum(d.get("num_chunks", 0) for d in docs)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", len(docs))
    col2.metric("Total Chunks", total_chunks)
    col3.metric("File Types", len(set(d["file_type"] for d in docs)))
