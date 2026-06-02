"""
Standalone evaluation runner.
Usage:
    python evaluation/run_eval.py [--questions path/to/questions.csv] [--top-k 5]
"""
import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline import run_query_pipeline
from backend.evaluator import evaluate_single
from database.db import init_db, insert_eval_result, insert_eval_questions

DEFAULT_QUESTIONS_PATH = Path(__file__).parent / "test_questions.csv"


def load_default_questions() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_QUESTIONS_PATH)


def run_evaluation(
    questions_df: pd.DataFrame,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    llm_backend: str = "stub",
    run_id: str | None = None,
) -> pd.DataFrame:
    run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    results = []

    # Persist questions to DB
    q_rows = []
    for _, row in questions_df.iterrows():
        q_rows.append({
            "question_id": str(row.get("question_id", uuid.uuid4().hex[:8])),
            "question": str(row.get("question", "")),
            "expected_answer": str(row.get("expected_answer", "")),
            "expected_doc": str(row.get("expected_doc", "")),
            "expected_page": int(row.get("expected_page", 0)),
            "expected_chunk": str(row.get("expected_chunk", "")),
            "answerable": int(row.get("answerable", 1)),
            "question_type": str(row.get("question_type", "")),
            "difficulty": str(row.get("difficulty", "")),
        })
    insert_eval_questions(q_rows)

    for i, (_, row) in enumerate(questions_df.iterrows(), 1):
        q = str(row.get("question", ""))
        expected_ans = str(row.get("expected_answer", ""))
        expected_doc = str(row.get("expected_doc", ""))
        answerable = bool(int(row.get("answerable", 1)))
        q_id = str(row.get("question_id", uuid.uuid4().hex[:8]))

        print(f"[{i}/{len(questions_df)}] {q[:70]}…", flush=True)

        import time
        t0 = time.time()
        try:
            pipeline_result = run_query_pipeline(
                query=q,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                llm_backend=llm_backend,
            )
        except Exception as e:
            print(f"  ⚠ pipeline error: {e}")
            continue

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

        result_row = {
            "run_id": run_id,
            "experiment_id": None,
            "question_id": q_id,
            "question": q,
            "generated_answer": gen_answer,
            "expected_answer": expected_ans,
            "retrieved_docs": [c.get("doc_name", "") for c in chunks],
            "expected_doc": expected_doc,
            "latency_ms": latency,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        insert_eval_result(result_row)
        results.append(result_row)

    df = pd.DataFrame(results)
    print(f"\n=== Evaluation complete: run_id={run_id} ===")
    if not df.empty:
        for col in ["precision_at_k", "recall_at_k", "answer_relevance",
                    "faithfulness", "completeness", "hallucination_risk"]:
            if col in df.columns:
                print(f"  {col}: {df[col].mean():.4f}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--llm-backend", default="stub")
    parser.add_argument("--output", default="reports/evaluation_summary.csv")
    args = parser.parse_args()

    init_db()
    questions = pd.read_csv(args.questions)
    results = run_evaluation(
        questions,
        top_k=args.top_k,
        use_hybrid=not args.no_hybrid,
        use_reranker=not args.no_reranker,
        llm_backend=args.llm_backend,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Results saved to {out}")
