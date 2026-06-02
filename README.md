<div align="center">

# 🔍 RAG Search & Answer Evaluation Assistant

### *Enterprise-grade RAG platform with built-in reliability measurement*

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge&logo=meta&logoColor=white"/>
<img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-F5A623?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>

<br/>

> **Most RAG chatbots just answer. This system also _evaluates_ whether those answers can be trusted.**

```
╔══════════════════════════════════════════════════════════════════════╗
║  Upload Docs → Semantic Search → Grounded Answers → Trust Scores    ║
╚══════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 🎯 What Makes This Different

| Ordinary RAG Chatbot | ✅ This System |
|----------------------|---------------|
| Retrieves chunks | Retrieves + measures **Precision@K, Recall@K, MRR, nDCG** |
| Generates an answer | Generates + measures **Faithfulness & Hallucination Risk** |
| Answers everything | Knows when to say **"Not enough information"** |
| No feedback loop | **10-category failure analysis** with fix suggestions |
| Single config | **Experiment tracking** across chunk sizes, models, top-K |

---

## 📸 Screenshots

### Page 1 — Upload Documents
*Upload PDF, DOCX, TXT, CSV, XLSX, HTML files. Configure chunking strategy, chunk size, overlap, embedding model, and vector store. View indexed document table with live status.*

<img src="https://github.com/Avvv19/rag-search-answer-evaluation-assistant/raw/main/docs/screenshots/page1_upload_documents.png" width="100%" alt="Upload Documents Page"/>

---

### Page 2 — Ask Questions
*Type a natural-language question. Get a grounded answer with source citations, similarity scores, evidence strength rating, and per-chunk reranking scores.*

<img src="https://github.com/Avvv19/rag-search-answer-evaluation-assistant/raw/main/docs/screenshots/page2_ask_questions.png" width="100%" alt="Ask Questions Page"/>

---

### Page 3 — Evaluation Runner
*Load 50 benchmark questions (CSV / DB / manual). Run end-to-end evaluation. See generated vs expected answers side-by-side with Precision@K, Faithfulness, and failure category per question.*

<img src="https://github.com/Avvv19/rag-search-answer-evaluation-assistant/raw/main/docs/screenshots/page3_run_evaluation.png" width="100%" alt="Run Evaluation Page"/>

---

### Page 4 — Metrics Dashboard
*10 KPI tiles + bar charts for answer quality metrics, source hit rate by question type, latency histogram, and failure category breakdown.*

<img src="https://github.com/Avvv19/rag-search-answer-evaluation-assistant/raw/main/docs/screenshots/page4_metrics_dashboard.png" width="100%" alt="Metrics Dashboard Page"/>

---

### Page 5 — Failure Analysis
*Per-failure drill-down: generated vs expected answer, root-cause category, retrieval metadata, and specific actionable improvement suggestion.*

<img src="https://github.com/Avvv19/rag-search-answer-evaluation-assistant/raw/main/docs/screenshots/page5_failure_analysis.png" width="100%" alt="Failure Analysis Page"/>

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Streamlit UI (5 pages)                  │
└────────────────────┬────────────────────────────────┘
                     │
              FastAPI Backend (optional)
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
Vector Search     BM25 Search    Metadata Filter
(FAISS/Chroma)  (rank-bm25)      (SQLite)
    │                │
    └────────┬───────┘
             ▼
      RRF Hybrid Fusion
             │
             ▼
     Cross-Encoder Reranker
             │
             ▼
     Answer Generator
   (Ollama / HuggingFace / Stub)
             │
             ▼
    Evaluation & Metrics Engine
    (12 metrics · 10 failure categories)
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Avvv19/rag-search-answer-evaluation-assistant.git
cd rag-search-answer-evaluation-assistant

# 2. Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure
cp .env.example .env       # edit LLM_BACKEND, EMBEDDING_MODEL, etc.

# 4. Launch
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

### Docker
```bash
docker build -t rag-eval .
docker run -p 8501:8501 -v $(pwd)/data:/app/data rag-eval
```

### FastAPI (optional)
```bash
uvicorn backend.api:app --reload --port 8000
# → http://localhost:8000/docs
```

---

## 📁 Project Structure

```
rag_search_answer_evaluation_assistant/
├── app/
│   ├── streamlit_app.py              ← Main entry point
│   └── pages/
│       ├── 1_upload_documents.py     ← Upload & index
│       ├── 2_ask_questions.py        ← Q&A + citations
│       ├── 3_run_evaluation.py       ← Eval runner
│       ├── 4_metrics_dashboard.py    ← Metrics viz
│       └── 5_failure_analysis.py     ← Failure diagnosis
├── backend/
│   ├── config.py           ingestion.py   chunking.py
│   ├── embeddings.py       vector_store.py  bm25_search.py
│   ├── hybrid_retriever.py reranker.py    generator.py
│   ├── evaluator.py        feedback.py    pipeline.py   api.py
├── database/
│   ├── schema.sql (6 tables)    db.py (CRUD helpers)
├── evaluation/
│   ├── test_questions.csv (50 questions)
│   ├── run_eval.py   metrics.py   failure_analysis.py
├── tests/
│   ├── test_chunking.py (6 tests)
│   ├── test_retrieval.py (8 tests)
│   └── test_evaluation.py (13 tests)
├── Dockerfile    requirements.txt    .env.example
```

---

## 🧠 Supported Models

### Embedding Models

| Model | Params | Best For |
|-------|--------|----------|
| `all-MiniLM-L6-v2` ⭐ | 22M | General purpose (default) |
| `multi-qa-MiniLM-L6-cos-v1` | 22M | Q&A retrieval |
| `BAAI/bge-small-en` | 33M | High quality, small |
| `BAAI/bge-base-en` | 109M | High quality, larger |
| `intfloat/e5-small-v2` | 33M | E5 family |
| `intfloat/e5-base-v2` | 109M | E5 family, larger |

### LLM Backends

| Backend | Config | Notes |
|---------|--------|-------|
| `ollama` 🦙 | `OLLAMA_MODEL=llama3` | 100% local, requires Ollama |
| `huggingface` 🤗 | `HF_MODEL=google/flan-t5-base` | Auto-downloads |
| `stub` 🧪 | — | Testing/CI mode |

---

## 📊 Evaluation Metrics

### Retrieval
```
Precision@K   = (relevant docs in top-K) / K
Recall@K      = 1 if expected doc in top-K else 0
MRR           = 1 / rank_of_first_relevant_result
nDCG@K        = DCG@K / IDCG@K
Source Coverage = matching chunks / total chunks
```

### Answer Quality
```
Answer Relevance  = cosine_sim(question, answer)
Faithfulness      = avg sim(answer sentences, context)
Completeness      = 0.4×token_overlap + 0.6×semantic_sim
Citation Accuracy = correct citations / total citations
Hallucination Risk = 1 - Faithfulness
No-Answer Accuracy = correct refusal/answer rate
```

---

## 🔴 Failure Categories

| Category | Meaning | Fix |
|----------|---------|-----|
| `retrieval_miss` | Correct doc not retrieved | Check indexing; try hybrid; ↑ top-K |
| `low_relevance_retrieval` | Low similarity scores | Use `bge-base-en`; enable reranker |
| `partial_context` | Only partial answer in context | ↑ chunk size; ↑ overlap; ↑ top-K |
| `bad_chunking` | Context split across chunks | Switch to `section_aware` |
| `conflicting_sources` | Multiple conflicting docs | Add metadata filters |
| `unsupported_answer` | Answer not grounded | ↓ LLM temperature; stricter prompt |
| `citation_mismatch` | Wrong docs cited | Normalise doc names on ingest |
| `no_answer_failure` | Wrong refusal/answer decision | Tune evidence threshold |
| `ambiguous_query` | Query too vague | Add query rewriting |
| `metadata_filtering_failure` | Filters excluded correct doc | Verify metadata values |

---

## 🧪 Tests

```bash
pytest tests/ -v
# 27 tests — chunking, BM25/RRF retrieval, all 12 evaluation metrics
```

---

## 📋 Benchmark Dataset

50 questions across 8 types:
- ✅ Factual lookup · Policy · Summary · Comparison
- ✅ Numeric/date · Multi-document · Ambiguous
- ✅ **Unanswerable** (tests no-answer refusal)

---

## 🔧 Configuration

```env
LLM_BACKEND=ollama          # ollama | huggingface | stub
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE=faiss           # faiss | chroma
TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
USE_RERANKER=true
USE_HYBRID=true
```

---

## 📜 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**Built with Streamlit · FAISS · ChromaDB · Sentence-Transformers · BM25 · Ollama**

⭐ Star this repo if it helped you build more reliable RAG systems!

</div>
