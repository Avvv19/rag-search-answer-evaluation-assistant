"""Aggregated metrics computation from eval results."""
import pandas as pd
import numpy as np


def compute_aggregate_metrics(results_df: pd.DataFrame) -> dict:
    """Compute aggregate evaluation metrics from results DataFrame."""
    if results_df.empty:
        return {}

    numeric_metrics = [
        "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k",
        "source_coverage", "answer_relevance", "faithfulness",
        "completeness", "citation_accuracy", "hallucination_risk",
        "latency_ms",
    ]
    binary_metrics = ["source_hit", "no_answer_correct"]

    agg = {}
    for col in numeric_metrics:
        if col in results_df.columns:
            agg[f"mean_{col}"] = round(float(results_df[col].mean()), 4)
            agg[f"median_{col}"] = round(float(results_df[col].median()), 4)

    for col in binary_metrics:
        if col in results_df.columns:
            agg[f"{col}_rate"] = round(float(results_df[col].mean()), 4)

    if "failure_category" in results_df.columns:
        agg["failure_distribution"] = results_df["failure_category"].value_counts().to_dict()

    if "question_type" in results_df.columns and "source_hit" in results_df.columns:
        agg["source_hit_by_type"] = (
            results_df.groupby("question_type")["source_hit"].mean().round(4).to_dict()
        )

    agg["total_questions"] = len(results_df)
    agg["correct_count"] = int((results_df.get("failure_category", pd.Series(dtype=str)) == "correct").sum())

    return agg


def compute_experiment_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    """Group results by experiment_id and compute aggregate metrics."""
    if "experiment_id" not in results_df.columns:
        return pd.DataFrame()
    metric_cols = [
        c for c in ["precision_at_k", "recall_at_k", "answer_relevance",
                     "faithfulness", "completeness", "hallucination_risk",
                     "source_hit", "latency_ms"]
        if c in results_df.columns
    ]
    return results_df.groupby("experiment_id")[metric_cols].mean().round(4).reset_index()
