"""Page 3 – Evaluation Runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
import io
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from database import db
from backend.pipeline import run_query_pipeline
from backend.evaluator import evaluate_single
from backend.config import settings
from backend.embeddings import SUPPORTED_MODELS

st.set_page_config(page_title="Run Evaluation", page_icon="🧪", layout="wide")
st.title("🧪 Evaluation Runner")

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Eval Settings")
    top_k = st.slider("Top-K", 1, 20, settings.TOP_K)
    use_hybrid = st.checkbox("Hybrid retrieval", value=settings.USE_HYBRID)
    use_reranker = st.checkbox("Reranker", value=settings.USE_RERANKER)
    llm_backend = st.selectbox("LLM backend", ["ollama", "huggingface", "stub"],
                               index=2)

# ─── Load test questions ──────────────────────────────────────────────────────────
st.subheader("📋 Load Test Questions")
tab_csv, tab_db, tab_manual = st.tabs(["Upload CSV", "From Database", "Manual Entry"])

questions_df = None

with tab_csv:
    st.markdown("""
    CSV must have columns: `question`, `expected_answer`, `expected_doc`,
    `expected_page`, `answerable` (1/0), `question_type`, `difficulty`
    """)
    uf = st.file_uploader("Upload test CSV", type=["csv"])
    if uf:
        questions_df = pd.read_csv(uf)
        st.dataframe(questions_df.head(), use_container_width=True)

with tab_db:
    db_qs = db.list_eval_questions()
    if db_qs:
        questions_df = pd.DataFrame(db_qs)
        st.dataframe(questions_df.head(10), use_container_width=True)
        st.caption(f"{len(db_qs)} questions in database")
    else:
        st.info("No evaluation questions in DB. Upload a CSV or use the default set.")
        if st.button("Load default test questions"):
            from evaluation.run_eval import load_default_questions
            questions_df = load_default_questions()
            st.success(f"Loaded {len(questions_df)} default questions")

with tab_manual:
    st.markdown("Add a single test question:")
    m_q = st.text_input("Question")
    m_ea = st.text_input("Expected answer (or keywords)")
    m_ed = st.text_input("Expected document name")
    m_ep = st.number_input("Expected page", min_value=0, value=1)
    m_ans = st.checkbox("Answerable?", value=True)
    m_type = st.selectbox("Question type",
                          ["factual", "summary", "comparison", "policy",
                           "numeric", "multi_doc", "ambiguous", "unanswerable"])
    m_diff = st.selectbox("Difficulty", ["easy", "medium", "hard"])
    if st.button("Add question") and m_q:
        new_row = {
            "question": m_q,
            "expected_answer": m_ea,
            "expected_doc": m_ed,
            "expected_page": m_ep,
            "answerable": int(m_ans),
            "question_type": m_type,
            "difficulty": m_diff,
        }
        questions_df = pd.DataFrame([new_row])
        st.success("Question added!")

# ─── Run evaluation ───────────────────────────────────────────────────────────────
st.markdown("---")
if questions_df is not None and not questions_df.empty:
    st.success(f"Ready to evaluate **{len(questions_df)}** questions")
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if st.button("▶️ Run Evaluation", type="primary"):
        progress = st.progress(0)
        results = []
        detail_container = st.empty()

        for i, row in questions_df.iterrows():
            q = str(row.get("question", ""))
            expected_ans = str(row.get("expected_answer", ""))
            expected_doc = str(row.get("expected_doc", ""))
            expected_page = int(row.get("expected_page", 0))
            answerable = bool(int(row.get("answerable", 1)))

            with st.spinner(f"[{i+1}/{len(questions_df)}] {q[:60]}…"):
                import time
                t0 = time.time()
                pipeline_result = run_query_pipeline(
                    query=q,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_reranker=use_reranker,
                    llm_backend=llm_backend,
                )
                latency = (time.time() - t0) * 1000
                chunks = pipeline_result.get("retrieved_chunks", [])
                gen_answer = pipeline_result.get("answer", "")
                metrics = evaluate_single(
                    question=q,
                    expected_answer=expected_ans,
                    expected_doc=expected_doc,
                    generated_answer=gen_answer,
                    retrieved_chunks=chunks,
                    answerable=answerable,
                    k=top_k,
                )

                q_id = str(row.get("question_id", uuid.uuid4().hex[:8]))
                result_row = {
                    "run_id": run_id,
                    "experiment_id": None,
                    "question_id": q_id,
                    "question": q,
                    "generated_answer": gen_answer,
                    "expected_answer": expected_ans,
                    "retrieved_docs": [c.get("doc_name","") for c in chunks],
                    "expected_doc": expected_doc,
                    "latency_ms": latency,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **metrics,
                }
                db.insert_eval_result(result_row)
                results.append(result_row)

            progress.progress((i + 1) / len(questions_df))

        st.success(f"✅ Evaluation complete! Run ID: `{run_id}`")

        results_df = pd.DataFrame(results)
        st.dataframe(results_df[[
            "question", "generated_answer", "expected_answer",
            "source_hit", "answer_relevance", "faithfulness",
            "completeness", "hallucination_risk", "failure_category"
        ]], use_container_width=True)

        # Download
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download results CSV",
            data=csv_buf.getvalue(),
            file_name=f"eval_{run_id}.csv",
            mime="text/csv",
        )
else:
    st.info("Load test questions using one of the tabs above to start evaluation.")
