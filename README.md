# 🔍 RAG Search & Answer Evaluation Assistant

> An enterprise-grade **Retrieval-Augmented Generation (RAG)** platform that lets you upload private documents, search them semantically, generate source-grounded answers, and **measure whether those answers are trustworthy**.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Project Does

Most RAG chatbots just *answer*. This system also *evaluates*:

| Question | What the system measures |
|----------|--------------------------|
| Did we retrieve the right evidence? | Precision@K, Recall@K, MRR, nDCG, Source Hit Rate |
| Is the answer grounded in the sources? | Faithfulness, Hallucination Risk, Citation Accuracy |
| Should we refuse to answer? | No-Answer Accuracy, Evidence Strength |
| What went wrong? | 10-category Failure Analysis with improvement suggestions |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│  Upload │ Ask │ Evaluate │ Metrics │ Failures    │
└────────────────────┬────────────────────────────┘
                     │
              FastAPI Backend (optional)
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
Vector Search     BM25 Search    Metadata Filter
(FAISS/Chroma)  (rank-bm25)      (SQLite)
    │                │
    └────────┬───────┘
             ▼
      Hybrid Fusion (RRF)
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
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-search-answer-evaluation-assistant.git
cd rag-search-answer-evaluation-assistant
```

### 2. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env to set your LLM_BACKEND (ollama | huggingface | stub)
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 5. (Optional) Run the FastAPI backend

```bash
uvicorn backend.api:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

---

## 🐳 Docker

```bash
docker build -t rag-eval-assistant .
docker run -p 8501:8501 -v $(pwd)/data:/app/data rag-eval-assistant
```

---

## 📁 Project Structure

```
rag_search_answer_evaluation_assistant/
│
├── app/
│   ├── streamlit_app.py              # Main Streamlit entry-point
│   └── pages/
│       ├── 1_upload_documents.py     # Upload & index documents
│       ├── 2_ask_questions.py        # Q&A with source citations
│       ├── 3_run_evaluation.py       # Evaluation runner
│       ├── 4_metrics_dashboard.py    # Metrics visualisation
│       └── 5_failure_analysis.py     # Failure diagnosis
│
├── backend/
│   ├── config.py           # Pydantic settings (reads .env)
│   ├── ingestion.py        # PDF/DOCX/TXT/CSV/XLSX/HTML parsing
│   ├── chunking.py         # Fixed/overlap/page/section chunking
│   ├── embeddings.py       # Sentence-Transformers wrapper
│   ├── vector_store.py     # FAISS + ChromaDB abstraction
│   ├── bm25_search.py      # BM25 keyword search
│   ├── hybrid_retriever.py # RRF fusion of vector + BM25
│   ├── reranker.py         # Cross-encoder reranker
│   ├── generator.py        # Ollama / HF / stub answer generation
│   ├── evaluator.py        # All retrieval + answer metrics
│   ├── feedback.py         # User feedback collection
│   ├── pipeline.py         # End-to-end orchestration
│   └── api.py              # FastAPI REST endpoints
│
├── database/
│   ├── schema.sql           # SQLite table definitions
│   └── db.py                # DB helpers (CRUD)
│
├── evaluation/
│   ├── test_questions.csv   # 50 benchmark questions
│   ├── run_eval.py          # CLI evaluation runner
│   ├── metrics.py           # Aggregate metrics computation
│   └── failure_analysis.py  # Failure export & analysis
│
├── data/
│   ├── raw_documents/       # Uploaded source files
│   ├── processed_chunks/    # Intermediate chunk files
│   └── vector_indexes/      # FAISS index + BM25 pickle
│
├── reports/
│   ├── evaluation_summary.csv
│   └── failure_report.csv
│
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_evaluation.py
│
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📄 Supported File Types

| Format | Parser |
|--------|--------|
| PDF | PyMuPDF (primary), pypdf (fallback) |
| DOCX | python-docx |
| TXT | Built-in |
| CSV | pandas |
| XLSX/XLS | pandas + openpyxl |
| HTML/HTM | BeautifulSoup4 |

---

## 🤖 Embedding Models

Configure via `EMBEDDING_MODEL` in `.env`:

| Model | Size | Best For |
|-------|------|----------|
| `all-MiniLM-L6-v2` | 80MB | General purpose (default) |
| `multi-qa-MiniLM-L6-cos-v1` | 80MB | Q&A retrieval |
| `BAAI/bge-small-en` | 130MB | High quality, small |
| `BAAI/bge-base-en` | 440MB | High quality, larger |
| `intfloat/e5-small-v2` | 130MB | E5 family |
| `intfloat/e5-base-v2` | 440MB | E5 family, larger |

---

## 🔍 LLM Backends

| Backend | Config | Notes |
|---------|--------|-------|
| `ollama` | `OLLAMA_MODEL=llama3` | Requires [Ollama](https://ollama.ai) running locally |
| `huggingface` | `HF_MODEL=google/flan-t5-base` | Downloads model on first use |
| `stub` | — | Returns templated responses; useful for testing |

---

## 📊 Evaluation Metrics

### Retrieval Metrics
- **Precision@K** – fraction of top-K results that match expected source
- **Recall@K** – whether the expected source appears in top-K
- **MRR** – Mean Reciprocal Rank
- **nDCG@K** – Normalised Discounted Cumulative Gain
- **Source Coverage** – fraction of all retrieved docs matching expected source

### Answer Metrics
- **Answer Relevance** – semantic similarity between question and answer
- **Faithfulness** – how grounded the answer is in retrieved context
- **Completeness** – comparison against expected answer
- **Citation Accuracy** – whether cited docs match retrieved docs
- **Hallucination Risk** – `1 - faithfulness`
- **No-Answer Accuracy** – correct refusal rate for unanswerable questions

### Failure Categories
`retrieval_miss` · `low_relevance_retrieval` · `partial_context` · `bad_chunking` · `conflicting_sources` · `unsupported_answer` · `citation_mismatch` · `no_answer_failure` · `ambiguous_query` · `metadata_filtering_failure`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🖥️ Streamlit Pages

| Page | Features |
|------|----------|
| **📄 Upload Documents** | Multi-file upload, chunking config, embedding model selector, delete/re-index |
| **💬 Ask Questions** | Q&A with source citations, reranking scores, metadata filters, feedback |
| **🧪 Run Evaluation** | CSV/DB question loading, per-question metrics, download results |
| **📊 Metrics Dashboard** | KPI tiles, metric distributions, latency histogram, failure breakdown |
| **🔎 Failure Analysis** | Per-failure drill-down, improvement suggestions, export report |

---

## ⚙️ Chunking Strategies

| Strategy | Best For |
|----------|----------|
| `overlap` | General documents (default) |
| `fixed` | Uniform text without structure |
| `page_aware` | PDFs where page boundaries matter |
| `section_aware` | Documents with clear headings |

---

## 🔧 Environment Variables

See `.env.example` for all options. Key variables:

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

## 📋 Benchmark Questions

The `evaluation/test_questions.csv` file contains **50 benchmark questions** covering:

- ✅ Factual lookup (easy/medium)
- ✅ Policy questions (easy/medium)
- ✅ Summary questions (hard)
- ✅ Comparison questions (medium/hard)
- ✅ Numeric/date questions
- ✅ Multi-document questions (hard)
- ✅ Ambiguous questions
- ✅ Unanswerable questions (no-answer test)

---

## 🛡️ No-Answer Handling

When retrieved evidence is insufficient, the system responds:

> *"The uploaded documents do not contain enough information to answer this confidently."*

This is measured via **No-Answer Accuracy** in the evaluation framework.

---

## 📦 CLI Evaluation

```bash
python evaluation/run_eval.py \
  --questions evaluation/test_questions.csv \
  --top-k 5 \
  --llm-backend stub \
  --output reports/evaluation_summary.csv
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) for BM25 search
- [Ollama](https://ollama.ai/) for local LLM serving
- [Streamlit](https://streamlit.io/) for the UI framework
