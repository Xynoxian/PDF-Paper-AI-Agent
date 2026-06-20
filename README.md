

# PDF-Papers AI Agent

Hybrid Retrieval + GraphRAG with Online Learning and AutoML (CSAI415).

## Architecture

```
PDF Files (papers/)
    |
    +-> [ingest.py] PyMuPDF extraction -> text chunking (500 char, 100 overlap)
    |
    +-> [sentence-transformers] BAAI/bge-small-en-v1.5 -> 384-dim embeddings
    |
    +-> MongoDB: chunk metadata (chunk_id, paper_id, title, authors, text, pages)
    |
    +-> Qdrant: dense vector index (cosine similarity)
    |
    +-> [hybrid_search.py] BM25 + Dense -> RRF fusion -> ranked results
    |
    +-> [graph_build.py] Neo4j knowledge graph
    |       Paper --(ABOUT)--> Topic
    |       Author --(WROTE)--> Paper
    |       Paper --(CITES)--> Paper
    |
    +-> [online_learning.py] River adaptive weights + ADWIN drift detection
    |
    +-> [automl_tuner.py] Optuna hyperparameter search
    |
    +-> [app.py] FastAPI REST API
            /ingest, /search, /feedback, /graph/query, /stats, /health
```

## Quick Start

```bash
# 1. Start backing services
docker compose up -d mongodb qdrant neo4j

# 2. Install dependencies
pip install -r requirements.txt

# 3. Seed with sample data
python seed_data.py

# 4. Run the API server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# API docs at http://localhost:8000/docs
```

Or with Docker Compose (all-in-one):

```bash
docker compose up -d --build
```

## Project Structure

```
.
├── app.py                  # FastAPI REST API (search, ingest, feedback, graph)
├── ingest.py               # PDF -> chunks -> MongoDB + Qdrant
├── hybrid_search.py        # BM25 + Dense + RRF retrieval with eval metrics
├── graph_build.py          # Neo4j knowledge graph builder
├── online_learning.py      # River online learner + ADWIN drift detection (D1)
├── automl_tuner.py         # Optuna AutoML for retriever hyperparameters (D1)
├── seed_data.py            # Data seeding and quick evaluation
├── docker-compose.yml      # MongoDB, Qdrant, Neo4j, FastAPI
├── Dockerfile              # App container image
├── requirements.txt        # All Python dependencies (D1 + D2 merged)
├── .env.example            # Environment variable template
├── papers/                 # Sample arXiv PDFs
├── configs/
│   └── run_card.yaml       # AutoML winning config + metrics (D1)
└── notebooks/
    └── D1_automl_streaming.ipynb  # Original D1 notebook (preserved)
```

## API Endpoints

| Endpoint          | Method | Description                                      |
|-------------------|--------|--------------------------------------------------|
| `/ingest`         | POST   | Ingest PDFs from a directory                     |
| `/search`         | POST   | Hybrid BM25+Dense search with RRF fusion         |
| `/feedback`       | POST   | Submit user feedback to River online learner      |
| `/feedback/stats` | GET    | Online learning stats + prequential log           |
| `/graph/query`    | POST   | Execute Cypher queries against Neo4j              |
| `/stats`          | GET    | System-wide statistics                            |
| `/health`         | GET    | Health check for all backing services             |

## Components by Deliverable

### D1 — Streaming Learner & AutoML
- `online_learning.py`: River LogisticRegression with ADWIN drift detection
- `automl_tuner.py`: Optuna TPE search over retriever hyperparameters
- `configs/run_card.yaml`: Winning config (k=11, alpha=0.115, SVD 256-dim)
- `notebooks/D1_automl_streaming.ipynb`: Original notebook (preserved)

### D2 — Retrieval Stack & Graph Build
- `ingest.py`: PDF extraction, chunking, embedding, MongoDB + Qdrant storage
- `hybrid_search.py`: BM25 + Dense + Reciprocal Rank Fusion
- `graph_build.py`: Neo4j graph (Paper, Author, Topic nodes + relationships)
- `app.py`: FastAPI with all endpoints
- `docker-compose.yml`: Full infrastructure stack

### D3 — GraphRAG Executor (planned)
Extension point: the `graph_build.py` Cypher queries + `hybrid_search.py` fusion +
`online_learning.py` adaptive weights form the foundation for the GraphRAG executor.

## Team

- **Mahmood**: Retrieval engine (BM25, dense search, RRF, evaluation metrics)
- **Afzal**: Backend infrastructure (FastAPI, Docker Compose, lazy loading)
- **Abdulsalam**: Data ingestion & graph layer (PyMuPDF, MongoDB, Neo4j)
