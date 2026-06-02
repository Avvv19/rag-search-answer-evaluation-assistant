"""Page 4 – Metrics Dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from database import db

st.set_page_config(page_title="Metrics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Metrics Dashboard")

results = db.get_eval_results()
if not results:
    st.info("No evaluation results yet. Run an evaluation on the **Run Evaluation** page first.")
    st.stop()

df = pd.DataFrame(results)

# ─── Run selector ─────────────────────────────────────────────────────────────────
runs = ["All runs"] + sorted(df["run_id"].unique().tolist(), reverse=True)
selected_run = st.selectbox("Select evaluation run", runs)
if selected_run != "All runs":
    df = df[df["run_id"] == selected_run]

st.caption(f"Showing {len(df)} evaluation records")

# ─── KPI tiles ────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Precision@K", f"{df['precision_at_k'].mean():.3f}" if 'precision_at_k' in df else "–")
col2.metric("Recall@K",    f"{df['recall_at_k'].mean():.3f}" if 'recall_at_k' in df else "–")
col3.metric("Source Hit Rate", f"{df['source_hit'].mean():.2%}" if 'source_hit' in df else "–")
col4.metric("Answer Relevance", f"{df['answer_relevance'].mean():.3f}" if 'answer_relevance' in df else "–")
col5.metric("Hallucination Risk", f"{df['hallucination_risk'].mean():.3f}" if 'hallucination_risk' in df else "–")

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("Faithfulness",     f"{df['faithfulness'].mean():.3f}" if 'faithfulness' in df else "–")
col7.metric("Completeness",     f"{df['completeness'].mean():.3f}" if 'completeness' in df else "–")
col8.metric("Citation Accuracy",f"{df['citation_accuracy'].mean():.3f}" if 'citation_accuracy' in df else "–")
col9.metric("No-Answer Acc.",   f"{df['no_answer_correct'].mean():.2%}" if 'no_answer_correct' in df else "–")
col10.metric("Avg Latency",     f"{df['latency_ms'].mean():.0f} ms" if 'latency_ms' in df else "–")

st.markdown("---")

# ─── Charts ───────────────────────────────────────────────────────────────────────
try:
    import altair as alt

    # Metric distributions
    st.subheader("📈 Metric Distributions")
    metric_cols = ["answer_relevance", "faithfulness", "completeness",
                   "citation_accuracy", "hallucination_risk"]
    metric_cols_present = [c for c in metric_cols if c in df.columns]

    if metric_cols_present:
        melt = df[metric_cols_present].melt(var_name="Metric", value_name="Score")
        chart = (
            alt.Chart(melt)
            .mark_boxplot(extent="min-max")
            .encode(x="Metric:N", y="Score:Q", color="Metric:N")
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    # Source hit by question type
    if "question_type" in df.columns and "source_hit" in df.columns:
        st.subheader("🎯 Source Hit Rate by Question Type")
        grp = df.groupby("question_type")["source_hit"].mean().reset_index()
        bar = (
            alt.Chart(grp)
            .mark_bar()
            .encode(
                x=alt.X("source_hit:Q", title="Hit Rate"),
                y=alt.Y("question_type:N", sort="-x"),
                color=alt.value("#4e79a7"),
            )
            .properties(height=250)
        )
        st.altair_chart(bar, use_container_width=True)

    # Latency histogram
    if "latency_ms" in df.columns:
        st.subheader("⏱️ Query Latency Distribution")
        lat_chart = (
            alt.Chart(df)
            .mark_bar(color="#f28e2b")
            .encode(
                x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=30), title="Latency (ms)"),
                y="count():Q",
            )
            .properties(height=200)
        )
        st.altair_chart(lat_chart, use_container_width=True)

except ImportError:
    st.info("Install altair for charts: `pip install altair`")

# ─── Full results table ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Full Results Table")
display_cols = [c for c in [
    "question", "source_hit", "precision_at_k", "recall_at_k", "mrr",
    "answer_relevance", "faithfulness", "completeness",
    "citation_accuracy", "hallucination_risk", "no_answer_correct",
    "failure_category", "latency_ms"
] if c in df.columns]
st.dataframe(df[display_cols], use_container_width=True)

# ─── Failure distribution ─────────────────────────────────────────────────────────
if "failure_category" in df.columns:
    st.markdown("---")
    st.subheader("🔴 Failure Category Distribution")
    fail_counts = df["failure_category"].value_counts().reset_index()
    fail_counts.columns = ["Category", "Count"]
    st.dataframe(fail_counts, use_container_width=True)
