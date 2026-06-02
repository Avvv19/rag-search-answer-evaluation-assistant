-- ─── Documents ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    file_type   TEXT    NOT NULL,
    source_path TEXT,
    category    TEXT    DEFAULT 'general',
    upload_date TEXT    NOT NULL,
    num_chunks  INTEGER DEFAULT 0,
    status      TEXT    DEFAULT 'pending'   -- pending | indexed | error
);

-- ─── Chunks ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id      TEXT    NOT NULL UNIQUE,
    doc_id        TEXT    NOT NULL REFERENCES documents(doc_id),
    doc_name      TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    page_number   INTEGER DEFAULT 0,
    section_title TEXT,
    chunk_index   INTEGER DEFAULT 0,
    chunk_size    INTEGER,
    overlap       INTEGER,
    embedding_model TEXT,
    vector_store  TEXT,
    created_at    TEXT    NOT NULL
);

-- ─── Experiments ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    chunk_size      INTEGER,
    chunk_overlap   INTEGER,
    embedding_model TEXT,
    vector_store    TEXT,
    top_k           INTEGER,
    use_hybrid      INTEGER DEFAULT 0,
    use_reranker    INTEGER DEFAULT 0,
    hybrid_alpha    REAL    DEFAULT 0.5,
    created_at      TEXT    NOT NULL,
    notes           TEXT
);

-- ─── Evaluation Questions ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_questions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id     TEXT    NOT NULL UNIQUE,
    question        TEXT    NOT NULL,
    expected_answer TEXT,
    expected_doc    TEXT,
    expected_page   INTEGER,
    expected_chunk  TEXT,
    answerable      INTEGER DEFAULT 1,   -- 1=yes, 0=no
    question_type   TEXT,
    difficulty      TEXT
);

-- ─── Evaluation Results ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL,
    experiment_id       TEXT,
    question_id         TEXT    NOT NULL REFERENCES eval_questions(question_id),
    question            TEXT    NOT NULL,
    generated_answer    TEXT,
    expected_answer     TEXT,
    retrieved_docs      TEXT,   -- JSON list
    expected_doc        TEXT,
    source_hit          INTEGER DEFAULT 0,
    answer_relevance    REAL    DEFAULT 0.0,
    faithfulness        REAL    DEFAULT 0.0,
    completeness        REAL    DEFAULT 0.0,
    citation_accuracy   REAL    DEFAULT 0.0,
    hallucination_risk  REAL    DEFAULT 0.0,
    no_answer_correct   INTEGER DEFAULT 0,
    failure_category    TEXT,
    latency_ms          REAL,
    created_at          TEXT    NOT NULL
);

-- ─── Feedback ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT    NOT NULL,
    answer      TEXT,
    rating      INTEGER,            -- 1-5
    comment     TEXT,
    helpful     INTEGER,            -- 1=yes, 0=no
    created_at  TEXT    NOT NULL
);

-- ─── Query Logs ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query           TEXT    NOT NULL,
    answer          TEXT,
    retrieved_chunks TEXT,  -- JSON
    latency_ms      REAL,
    config_snapshot TEXT,   -- JSON of settings used
    created_at      TEXT    NOT NULL
);
