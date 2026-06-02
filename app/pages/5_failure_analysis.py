"""Page 5 – Failure Analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st
from database import db

st.set_page_config(page_title="Failure Analysis", page_icon="🔎", layout="wide")
st.title("🔎 Failure Analysis")

results = db.get_eval_results()
if not results:
    st.info("No evaluation results. Run the evaluation first.")
    st.stop()

df = pd.DataFrame(results)
if "failure_category" not in df.columns:
    st.warning("No failure_category column found in results.")
    st.stop()

# ─── Filters ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
categories = ["All"] + sorted(df["failure_category"].dropna().unique().tolist())
sel_cat = col1.selectbox("Filter by failure category", categories)

runs = ["All runs"] + sorted(df["run_id"].unique().tolist(), reverse=True)
sel_run = col2.selectbox("Filter by run", runs)

view = df.copy()
if sel_cat != "All":
    view = view[view["failure_category"] == sel_cat]
if sel_run != "All runs":
    view = view[view["run_id"] == sel_run]

# Only failures (non-correct)
failures = view[view["failure_category"] != "correct"]
st.markdown(f"**{len(failures)} failed questions** (out of {len(view)} in selection)")

# ─── Summary table ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Failure Category Summary")
summary = (
    view.groupby("failure_category")
    .agg(count=("question", "count"),
         avg_relevance=("answer_relevance", "mean"),
         avg_faithfulness=("faithfulness", "mean"),
         avg_source_hit=("source_hit", "mean"))
    .reset_index()
    .sort_values("count", ascending=False)
)
st.dataframe(summary, use_container_width=True)

# ─── Individual failures ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Individual Failed Questions")

SUGGESTIONS = {
    "retrieval_miss": "Try hybrid retrieval or expand chunk size; the relevant document may not be indexed.",
    "low_relevance_retrieval": "Improve embedding model or add metadata filters to narrow the search.",
    "partial_context": "Reduce chunk overlap or increase top-K to capture more context.",
    "bad_chunking": "Try section-aware chunking to better preserve document structure.",
    "conflicting_sources": "Add document-level metadata filtering to limit conflicting results.",
    "unsupported_answer": "LLM is generating answers not grounded in context; lower temperature or use a stricter prompt.",
    "citation_mismatch": "Enable stricter citation checking; ensure doc names are consistent.",
    "no_answer_failure": "Tune the no-answer threshold; the model is answering unanswerable questions.",
    "ambiguous_query": "Ask users to reformulate; consider adding query rewriting.",
    "metadata_filtering_failure": "Check filter values match document metadata exactly.",
    "correct": "No issues.",
}

for _, row in failures.iterrows():
    cat = row.get("failure_category", "unknown")
    with st.expander(
        f"❌ [{cat}] {str(row.get('question',''))[:80]}"
    ):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Generated Answer:**")
            st.info(str(row.get("generated_answer", "–"))[:500])
        with col_b:
            st.markdown("**Expected Answer:**")
            st.success(str(row.get("expected_answer", "–"))[:500])

        col_c, col_d, col_e = st.columns(3)
        col_c.metric("Answer Relevance",   f"{row.get('answer_relevance', 0):.3f}")
        col_d.metric("Faithfulness",       f"{row.get('faithfulness', 0):.3f}")
        col_e.metric("Source Hit",         "✅" if row.get("source_hit") else "❌")

        st.markdown("**Expected Document:**")
        st.code(str(row.get("expected_doc", "–")))

        st.markdown("**Retrieved Documents:**")
        import json
        try:
            retrieved = json.loads(row.get("retrieved_docs", "[]"))
        except Exception:
            retrieved = []
        st.write(retrieved or "None")

        st.markdown(f"**💡 Suggestion:** {SUGGESTIONS.get(cat, 'Review retrieval and generation settings.')}")

# ─── Export ───────────────────────────────────────────────────────────────────────
st.markdown("---")
import io
csv_buf = io.StringIO()
failures.to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Export failure report CSV",
    data=csv_buf.getvalue(),
    file_name="failure_report.csv",
    mime="text/csv",
)
