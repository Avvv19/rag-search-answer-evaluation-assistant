"""SQLite database helpers."""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from loguru import logger

from backend.config import settings

DB_PATH = Path(settings.SQLITE_DB_PATH)
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_cursor():
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"DB error: {e}")
        raise
    finally:
        conn.close()


def init_db():
    """Create tables from schema.sql if they don't exist."""
    schema = SCHEMA_PATH.read_text()
    conn = get_connection()
    conn.executescript(schema)
    conn.commit()
    conn.close()
    logger.info(f"Database initialised at {DB_PATH}")


# ─── Documents ──────────────────────────────────────────────────────────────────

def insert_document(doc: dict):
    with db_cursor() as cur:
        cur.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, name, file_type, source_path, category, upload_date, num_chunks, status)
               VALUES (:doc_id,:name,:file_type,:source_path,:category,:upload_date,:num_chunks,:status)""",
            doc,
        )


def list_documents() -> list[dict]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM documents ORDER BY upload_date DESC")
        return [dict(r) for r in cur.fetchall()]


def delete_document(doc_id: str):
    with db_cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        cur.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))


def update_document_status(doc_id: str, status: str, num_chunks: int = 0):
    with db_cursor() as cur:
        cur.execute(
            "UPDATE documents SET status=?, num_chunks=? WHERE doc_id=?",
            (status, num_chunks, doc_id),
        )


# ─── Chunks ─────────────────────────────────────────────────────────────────────

def insert_chunks(chunks: list[dict]):
    with db_cursor() as cur:
        cur.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id,doc_id,doc_name,text,page_number,section_title,
                chunk_index,chunk_size,overlap,embedding_model,vector_store,created_at)
               VALUES (:chunk_id,:doc_id,:doc_name,:text,:page_number,:section_title,
                       :chunk_index,:chunk_size,:overlap,:embedding_model,:vector_store,:created_at)""",
            chunks,
        )


def get_chunks_by_doc(doc_id: str) -> list[dict]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,))
        return [dict(r) for r in cur.fetchall()]


def get_all_chunks() -> list[dict]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM chunks ORDER BY doc_id, chunk_index")
        return [dict(r) for r in cur.fetchall()]


def get_chunk_by_id(chunk_id: str) -> dict | None:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# ─── Eval Questions ─────────────────────────────────────────────────────────────

def insert_eval_questions(questions: list[dict]):
    with db_cursor() as cur:
        cur.executemany(
            """INSERT OR REPLACE INTO eval_questions
               (question_id,question,expected_answer,expected_doc,expected_page,
                expected_chunk,answerable,question_type,difficulty)
               VALUES (:question_id,:question,:expected_answer,:expected_doc,:expected_page,
                       :expected_chunk,:answerable,:question_type,:difficulty)""",
            questions,
        )


def list_eval_questions() -> list[dict]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM eval_questions")
        return [dict(r) for r in cur.fetchall()]


# ─── Eval Results ───────────────────────────────────────────────────────────────

def insert_eval_result(result: dict):
    if isinstance(result.get("retrieved_docs"), list):
        result = {**result, "retrieved_docs": json.dumps(result["retrieved_docs"])}
    with db_cursor() as cur:
        cur.execute(
            """INSERT INTO eval_results
               (run_id,experiment_id,question_id,question,generated_answer,expected_answer,
                retrieved_docs,expected_doc,source_hit,answer_relevance,faithfulness,
                completeness,citation_accuracy,hallucination_risk,no_answer_correct,
                failure_category,latency_ms,created_at)
               VALUES (:run_id,:experiment_id,:question_id,:question,:generated_answer,
                       :expected_answer,:retrieved_docs,:expected_doc,:source_hit,
                       :answer_relevance,:faithfulness,:completeness,:citation_accuracy,
                       :hallucination_risk,:no_answer_correct,:failure_category,
                       :latency_ms,:created_at)""",
            result,
        )


def get_eval_results(run_id: str | None = None) -> list[dict]:
    with db_cursor() as cur:
        if run_id:
            cur.execute("SELECT * FROM eval_results WHERE run_id=?", (run_id,))
        else:
            cur.execute("SELECT * FROM eval_results ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


# ─── Feedback ───────────────────────────────────────────────────────────────────

def insert_feedback(fb: dict):
    with db_cursor() as cur:
        cur.execute(
            """INSERT INTO feedback (query,answer,rating,comment,helpful,created_at)
               VALUES (:query,:answer,:rating,:comment,:helpful,:created_at)""",
            fb,
        )


# ─── Query Logs ─────────────────────────────────────────────────────────────────

def log_query(entry: dict):
    if isinstance(entry.get("retrieved_chunks"), (list, dict)):
        entry = {**entry, "retrieved_chunks": json.dumps(entry["retrieved_chunks"])}
    if isinstance(entry.get("config_snapshot"), dict):
        entry = {**entry, "config_snapshot": json.dumps(entry["config_snapshot"])}
    with db_cursor() as cur:
        cur.execute(
            """INSERT INTO query_logs (query,answer,retrieved_chunks,latency_ms,config_snapshot,created_at)
               VALUES (:query,:answer,:retrieved_chunks,:latency_ms,:config_snapshot,:created_at)""",
            entry,
        )


# ─── Experiments ────────────────────────────────────────────────────────────────

def insert_experiment(exp: dict):
    with db_cursor() as cur:
        cur.execute(
            """INSERT OR REPLACE INTO experiments
               (experiment_id,name,chunk_size,chunk_overlap,embedding_model,vector_store,
                top_k,use_hybrid,use_reranker,hybrid_alpha,created_at,notes)
               VALUES (:experiment_id,:name,:chunk_size,:chunk_overlap,:embedding_model,
                       :vector_store,:top_k,:use_hybrid,:use_reranker,:hybrid_alpha,:created_at,:notes)""",
            exp,
        )


def list_experiments() -> list[dict]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]
