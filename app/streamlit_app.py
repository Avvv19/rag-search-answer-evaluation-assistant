"""Main Streamlit entry-point."""
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# Initialise DB on first run
from database.db import init_db
init_db()

st.set_page_config(
    page_title="RAG Evaluation Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 RAG Search & Answer Evaluation Assistant")
st.markdown(
    """
    A **personal RAG evaluation prototype** that lets you upload documents,
    search them semantically, generate source-grounded answers, and measure
    whether those answers are **trustworthy**.

    ---
    ### Navigate using the sidebar pages:
    | Page | Purpose |
    |------|---------|
    | 📄 **Upload Documents** | Ingest & index your files |
    | 💬 **Ask Questions** | Semantic search + grounded answers |
    | 🧪 **Run Evaluation** | Benchmark retrieval & answer quality |
    | 📊 **Metrics Dashboard** | Visualise evaluation metrics |
    | 🔎 **Failure Analysis** | Diagnose what went wrong |
    """
)

st.sidebar.success("Select a page above ↑")

with st.sidebar:
    st.markdown("---")
    st.caption("Built with Streamlit · FAISS · ChromaDB · BM25 · Sentence-Transformers")
