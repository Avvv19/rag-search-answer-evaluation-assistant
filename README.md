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

<br/>

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

## 📸 Screenshots & Proof of Work

### 📁 Project Structure (48 files · 2,918 lines of Python)

```
rag_search_answer_evaluation_assistant/
│
├── 📱 app/
│   ├── streamlit_app.py                 ← Main entry point
│   └── pages/
│       ├── 1_upload_documents.py        ← Page 1: Upload & Index
│       ├── 2_ask_questions.py           ← Page 2: Q&A with citations
│       ├── 3_run_evaluation.py          ← Page 3: Eval runner
│       ├── 4_metrics_dashboard.py       ← Page 4: Metrics viz
│       └── 5_failure_analysis.py        ← Page 5: Failure drill-down
│
├── 🔧 backend/
│   ├── config.py          ← Pydantic settings (.env)
│   ├── ingestion.py       ← PDF/DOCX/TXT/CSV/XLSX/HTML parsers
│   ├── chunking.py        ← 4 chunking strategies
│   ├── embeddings.py      ← Sentence-Transformers (6 models)
│   ├── vector_store.py    ← FAISS + ChromaDB abstraction
│   ├── bm25_search.py     ← BM25 keyword index
│   ├── hybrid_retriever.py← RRF fusion (vector + BM25)
│   ├── reranker.py        ← Cross-encoder reranker
│   ├── generator.py       ← Ollama / HuggingFace / Stub LLM
│   ├── evaluator.py       ← All 12 retrieval+answer metrics
│   ├── feedback.py        ← User rating collection
│   ├── pipeline.py        ← End-to-end orchestration
│   └── api.py             ← FastAPI REST endpoints
│
├── 🗄️ database/
│   ├── schema.sql         ← 6 SQLite tables
│   └── db.py              ← CRUD helpers
│
├── 📊 evaluation/
│   ├── test_questions.csv ← 50 benchmark questions
│   ├── run_eval.py        ← CLI evaluation runner
│   ├── metrics.py         ← Aggregate metric computation
│   └── failure_analysis.py← Failure export & categorisation
│
├── 🧪 tests/
│   ├── test_chunking.py   ← 6 chunking tests
│   ├── test_retrieval.py  ← 8 retrieval tests (BM25 + RRF)
│   └── test_evaluation.py ← 13 metric unit tests
│
├── 🐳 Dockerfile
├── 📋 requirements.txt    ← 25+ packages
└── ⚙️  .env.example
```

---

### 🖥️ Page 1 — Upload Documents

```
┌─────────────────────────────────────────────────────────────────────┐
│  📄 Upload Documents                                                 │
├────────────────────────────┬────────────────────────────────────────┤
│  ⚙️ SIDEBAR SETTINGS        │  📂 DROP FILES HERE                    │
│                            │  ┌────────────────────────────────┐   │
│  Category:  [general     ] │  │  PDF  DOCX  TXT  CSV           │   │
│  Strategy:  [overlap   ▼] │  │  XLSX  HTML  HTM               │   │
│  Chunk size: [500 ◄────►] │  └────────────────────────────────┘   │
│  Overlap %:  [10% ◄────►] │                                        │
│  Embedding:  [MiniLM-L6 ▼]│  [🚀 Index uploaded files]             │
│  Vector DB:  [faiss     ▼]│                                        │
│                            ├────────────────────────────────────────┤
│                            │  📚 Indexed Documents                  │
│                            │  ┌──────────┬──────┬───────┬───────┐  │
│                            │  │ Name     │Chunks│ Type  │Status │  │
│                            │  ├──────────┼──────┼───────┼───────┤  │
│                            │  │ hr.pdf   │  124 │  pdf  │✅ OK  │  │
│                            │  │ sla.docx │   87 │ docx  │✅ OK  │  │
│                            │  └──────────┴──────┴───────┴───────┘  │
│                            │                                        │
│                            │  📊  3 Docs │ 311 Chunks │ 2 Types     │
└────────────────────────────┴────────────────────────────────────────┘
```

---

### 🖥️ Page 2 — Ask Questions

```
┌─────────────────────────────────────────────────────────────────────┐
│  💬 Ask Questions                                                    │
├────────────────────────────┬────────────────────────────────────────┤
│  ⚙️ RETRIEVAL SETTINGS      │  ┌──────────────────────────────────┐ │
│                            │  │ What is the annual leave policy? │ │
│  Top-K:  [5 ◄──────────►] │  └──────────────────────────────────┘ │
│  ☑ Hybrid retrieval        │  [🔍 Ask]                             │
│  ☑ Reranker               │                                        │
│  Model: [MiniLM-L6-v2  ▼] │  ✅ Answer                            │
│  LLM:   [ollama        ▼] │  ┌──────────────────────────────────┐ │
│                            │  │ Full-time employees are entitled │ │
│  🔎 FILTERS (optional)     │  │ to 20 days annual leave per year.│ │
│  Doc: [(all)           ▼] │  │ [Source: hr_policy.pdf, Page 3]  │ │
│  Cat: [(all)           ▼] │  └──────────────────────────────────┘ │
│                            │                                        │
│                            │  📊 STRONG │ 234 ms │ ollama          │
│                            │                                        │
│                            │  📎 Retrieved Sources                  │
│                            │  ▼ [1] hr_policy.pdf — Page 3 (0.92) │
│                            │     "Full-time employees receive 20…" │
│                            │  ▼ [2] hr_policy.pdf — Page 4 (0.81) │
│                            │     "Leave accrues at 1.67 days per…" │
└────────────────────────────┴────────────────────────────────────────┘
```

---

### 🖥️ Page 3 — Evaluation Runner

```
┌─────────────────────────────────────────────────────────────────────┐
│  🧪 Run Evaluation                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Load Test Questions: [Upload CSV] [From Database] [Manual Entry]   │
│                                                                     │
│  ✅ Ready to evaluate 50 questions                                  │
│                                                                     │
│  [▶️ Run Evalu