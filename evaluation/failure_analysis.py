"""Failure analysis utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from database.db import get_eval_results


FAILURE_SUGGESTIONS = {
    "retrieval_miss": (
        "The correct document was not retrieved. "
        "Suggestions: (1) verify the document is indexed; "
        "(2) try hybrid retrieval; (3) increase top-K; "
        "(4) check metadata filters are not excluding the document."
    ),
    "low_relevance_retrieval": (
        "Retrieved chunks have low relevance scores. "
        "Suggestions: (1) switch to a stronger embedding model (bge-base-en); "
        "(2) enable hybrid search to capture exact keywords; "
        "(3) enable reranker to reorder chunks."
    ),
    "partial_context": (
        "Only partial context was retrieved. "
        "Suggestions: (1) increase chunk size; "
        "(2) increase overlap; "
        "(3) increase top-K; "
        "(4) use section-aware chunking."
    ),
    "bad_chunking": (
        "Chunking broke useful context. "
        "Suggestions: (1) use section-aware chunking; "
        "(2) try page-aware chunking for PDFs; "
        "(3) adjust chunk size downward."
    ),
    "conflicting_sources": (
        "Multiple conflicting sources retrieved. "
        "Suggestions: (1) add document-level metadata filters; "
        "(2) use reranker to select most relevant chunk; "
        "(3) increase reranker score threshold."
    ),
    "unsupported_answer": (
        "Answer not grounded in retrieved context (potential hallucination). "
        "Suggestions: (1) use a stricter generation prompt; "
        "(2) lower LLM temperature; "
        "(3) filter chunks by minimum relevance score."
    ),
    "citation_mismatch": (
        "Answer cites incorrect or non-existent documents. "
        "Suggestions: (1) ensure document names are clean and consistent; "
        "(2) validate chunk metadata; "
        "(3) enforce citation format in the prompt."
    ),
    "no_answer_failure": (
        "System answered an unanswerable question (or refused an answerable one). "
        "Suggestions: (1) calibrate the evidence strength threshold; "
        "(2) add confidence scoring to the generator; "
        "(3) review no-answer detection patterns."
    ),
    "ambiguous_query": (
        "Query is ambiguous or too vague. "
        "Suggestions: (1) implement query rewriting/expansion; "
        "(2) prompt user for clarification; "
        "(3) return multiple candidate answers."
    ),
    "metadata_filtering_failure": (
        "Metadata filters excluded the correct document. "
        "Suggestions: (1) verify filter values match document metadata; "
        "(2) normalise category and doc_name fields on ingestion."
    ),
    "correct": "No failure — question answered correctly.",
}


def load_failures(run_id: str | None = None) -> pd.DataFrame:
    results = get_eval_results(run_id)
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df[df.get("failure_category", pd.Series(dtype=str)) != "correct"]


def failure_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "failure_category" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("failure_category")
        .agg(count=("question", "count"),
             avg_relevance=("answer_relevance", "mean"),
             avg_faithfulness=("faithfulness", "mean"),
             avg_source_hit=("source_hit", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )


def export_failure_report(output_path: str = "reports/failure_report.csv", run_id: str | None = None):
    df = load_failures(run_id)
    df["suggestion"] = df["failure_category"].map(FAILURE_SUGGESTIONS)
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Failure report saved to {output_path}")
    return df


if __name__ == "__main__":
    df = export_failure_report()
    if not df.empty:
        print(failure_summary(df).to_string())
