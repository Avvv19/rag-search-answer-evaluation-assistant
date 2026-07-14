# RAG Search and Answer Evaluation Assistant

A personal evaluation prototype for experimenting with document ingestion, hybrid retrieval, answer generation, and repeatable quality checks. The project is an evaluation harness, not proof of quality improvement or a deployed knowledge product.

## Implemented scope

- PDF, DOCX, HTML, spreadsheet, and text ingestion paths
- Configurable chunking and overlap
- FAISS or Chroma vector storage
- Optional BM25 hybrid retrieval and reranking
- Ollama, Hugging Face, or deterministic stub answer backends
- FastAPI and Streamlit surfaces
- SQLite persistence
- Heuristic and embedding-based evaluation
- A canonical 25-pair JSON golden set

The evaluator is not an LLM-as-judge system. Scores are produced from implemented lexical, retrieval, and embedding-oriented methods. No statistically validated improvement claim is made because complete versioned run outputs are not yet included.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux
```

The default `.env.example` selects Ollama. Choose `LLM_BACKEND=stub` for a deterministic path that does not require a model service.

## Run and test

Follow the entry points in the repository for the API or Streamlit UI, then run:

```bash
pytest -q
```

## Evaluation contract

The 25-pair JSON golden set is the canonical regression dataset. A publishable evaluation should record:

- commit hash and configuration
- corpus version
- golden-set version
- retrieval and answer metrics
- per-item failures
- model/backend and embedding versions
- exact command and environment

## Limitations

- No external users or production traffic are claimed.
- No saved benchmark establishes superiority over another system.
- Golden-set coverage is limited and should be expanded by failure category.
- Model and embedding downloads can make results environment-dependent.

## Future work

- Save machine-readable evaluation reports with each release
- Separate retrieval failures from generation failures
- Add human review labels and calibration guidance
- Document both golden datasets if a distinct second use case is retained
