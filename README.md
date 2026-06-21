# PDF-Papers AI Agent

Hybrid Retrieval + GraphRAG with Online Learning, AutoML, and Provenance Safety (CSAI415).

---

> **FASTEST WAY TO RUN THIS PROJECT:**
> 1. Have **Docker Desktop** running and **Python 3.11+** installed
> 2. Double-click **`start.bat`**
> 3. The browser opens automatically. Done.
>
> That's it. Everything else in this README is details. See [Quick Start (One Command)](#quick-start-one-command) below.

---

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
    |       Loads k=11, alpha=0.1154 from configs/run_card.yaml (AutoML winning params)
    |       Supports paper_ids filtering for graph-guided retrieval
    |
    +-> [graph_build.py] Neo4j knowledge graph
    |       Paper --(HAS_TOPIC)--> Topic
    |       Author --(WROTE)--> Paper
    |       Paper --(CITES)--> Paper
    |
    +-> [graphrag_executor.py] D3 GraphRAG Pipeline (Method A -- Pre-Filter)
    |       Query -> LLM Cypher generation -> Neo4j subgraph -> filtered search
    |       -> LLM answer generation -> provenance safety filtering
    |
    +-> [online_learning.py] River adaptive weights + ADWIN drift detection
    |
    +-> [automl_tuner.py] Optuna hyperparameter search
    |
    +-> [evaluate.py] Ablation harness (vector_only / graph_only / hybrid)
    |
    +-> [app.py] FastAPI REST API
            /ingest, /search, /graphrag, /feedback, /feedback/stats,
            /graph/query, /stats, /health
```

## Prerequisites

- **Docker Desktop** (or Docker Engine + Docker Compose v2)
- **Python 3.11+**
- **Git**
- **An LLM API key** (OpenAI, or any OpenAI-compatible provider) for the GraphRAG pipeline

## Quick Start (One Command)

Make sure **Docker Desktop** is running and **Python 3.11+** is installed, then:

```
start.bat
```

That's it. The script will:
1. Start MongoDB, Qdrant, and Neo4j in Docker
2. Create a virtual environment and install dependencies
3. Seed 5 sample arXiv papers into the system
4. Launch the server and open **http://localhost:8001** in your browser

> **Note:** On first run, dependency installation and paper downloads may take a couple of minutes.
> If you want GraphRAG (LLM-powered answers), edit `.env` and set `LLM_API_KEY` before running.
> Hybrid Search works without an API key.

To stop the server, press `Ctrl+C` in the terminal window.

### Manual Setup (Step by Step)

If you prefer to run each step yourself:

```bash
# 1. Clone the repo
git clone https://github.com/Xynoxian/PDF-Paper-AI-Agent.git
cd PDF-Paper-AI-Agent

# 2. Start database services
docker compose up -d mongodb qdrant neo4j

# 3. Create venv and install dependencies
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

# 4. Configure environment (set LLM_API_KEY for GraphRAG)
copy .env.example .env

# 5. Seed sample data
python seed_data.py

# 6. Start the server
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Open **http://localhost:8001** in your browser.

> **Tip:** Swagger API docs are also available at **http://localhost:8001/docs** if needed.

---

## Demo Walkthrough

Everything is done through the web UI at **http://localhost:8001**. No terminal commands needed after setup. The sidebar on the left has four tabs:

### 1. Chat (D2 + D3)

This is the main feature. Type a question and press Enter.

**Try with Hybrid Search mode** (select from the dropdown):
```
What is the attention mechanism in transformers?
```
This runs BM25 + Dense retrieval with RRF fusion (D2). You get ranked chunks from the ingested papers.

**Try with GraphRAG mode** (select from the dropdown):
```
What papers discuss attention mechanisms?
```
This runs the full GraphRAG pipeline (D3): LLM generates a Cypher query, retrieves a subgraph from Neo4j, filters results, and generates a grounded answer with citations. You will see:
- The LLM answer
- Expandable **Sources** with verified/dropped citations and a provenance score
- Metadata tags showing whether graph filtering was applied

After each GraphRAG answer, click the **thumbs up/down** buttons to submit feedback. This triggers the online learning loop (D1): River model weight update + ADWIN drift detection.

### 2. Ingest PDFs (D2)

Click **Ingest PDFs** in the sidebar.

- **Upload**: drag and drop any PDF file onto the drop zone (or click to browse). The file is saved and ingested automatically. You will see the chunk count and processing time.
- **Directory ingestion**: type a folder path (default: `papers`) and click Start Ingestion to ingest all PDFs in that folder.

After uploading a new PDF, go back to the Chat tab and ask questions about it.

### 3. Graph Query (D2)

Click **Graph Query** in the sidebar. Paste a Cypher query and click Execute.

**Example — see all papers and their topics:**
```
MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic) RETURN p.title AS paper, collect(t.name) AS topics
```

**Example — see which authors wrote which papers:**
```
MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name AS author, p.title AS paper LIMIT 10
```

**Example — count papers per topic:**
```
MATCH (t:Topic)<-[:HAS_TOPIC]-(p:Paper) RETURN t.name AS topic, count(p) AS num_papers ORDER BY num_papers DESC
```

### 4. Stats (D1)

Click **Stats** in the sidebar.

- **System Statistics** — shows counts across all stores (MongoDB documents, Qdrant vectors, Neo4j nodes) and the current online learning state.
- **Online Learning** — shows the River/ADWIN feedback stats: total steps, accuracy, drift count, and the prequential log. This updates each time you submit feedback via the thumbs up/down buttons in the Chat tab.

---

## Evaluation & Ablation Harness (D3)

Run from the terminal (not the web UI):

```bash
# Single mode
python evaluate.py --csv eval_queries.csv --mode hybrid --top-k 5

# Full ablation (compares all three modes)
python evaluate.py --csv eval_queries.csv --ablation --top-k 5
```

| Mode | Description |
|------|-------------|
| `vector_only` | BM25 + Dense hybrid search only, no graph filtering |
| `graph_only` | Only graph-filtered chunks, no unfiltered fallback |
| `hybrid` | Full GraphRAG pipeline (graph filter with fallback) |

Output: metrics table printed to stdout + results saved to `eval_results.json`.
A sample `eval_queries.csv` with 5 queries is included in the repo.

---

## Project Structure

```
.
├── app.py                  # FastAPI REST API (all endpoints including /graphrag)
├── graphrag_executor.py    # D3 GraphRAG pipeline + provenance safety filter
├── hybrid_search.py        # BM25 + Dense + RRF (loads AutoML config, paper_ids filter)
├── graph_build.py          # Neo4j knowledge graph builder
├── online_learning.py      # River online learner + ADWIN drift detection (D1)
├── automl_tuner.py         # Optuna AutoML for retriever hyperparameters (D1)
├── evaluate.py             # D3 evaluation & ablation harness
├── ingest.py               # PDF -> chunks -> MongoDB + Qdrant
├── seed_data.py            # Data seeding and quick evaluation
├── eval_queries.csv        # Sample evaluation queries
├── start.bat               # One-click launcher (Windows)
├── docker-compose.yml      # MongoDB, Qdrant, Neo4j, FastAPI
├── Dockerfile              # App container image
├── requirements.txt        # All Python dependencies (D1 + D2 + D3)
├── .env.example            # Environment variable template
├── static/
│   └── index.html          # Web UI (chatbot frontend)
├── papers/                 # Sample arXiv PDFs (auto-downloaded by seed_data.py)
├── configs/
│   └── run_card.yaml       # AutoML winning config + metrics (D1)
└── Notebooks/
    ├── D1.ipynb                   # Original D1 notebook (streaming learner + AutoML)
    ├── D2_retrieval_stack.ipynb   # D2 demo: ingestion, hybrid search, graph, eval
    └── D3_graphrag_executor.ipynb # D3 demo: GraphRAG, provenance, feedback, ablation
```

## API Endpoints

| Endpoint          | Method | Tag             | Description                                          |
|-------------------|--------|-----------------|------------------------------------------------------|
| `/`               | GET    | Frontend        | Web UI (chatbot interface)                           |
| `/upload`         | POST   | Ingestion       | Upload and ingest a single PDF file                  |
| `/ingest`         | POST   | Ingestion       | Ingest PDFs from a directory                         |
| `/search`         | POST   | Search          | Hybrid BM25+Dense search with RRF fusion             |
| `/graphrag`       | POST   | GraphRAG        | Full GraphRAG pipeline with provenance safety        |
| `/feedback`       | POST   | Online Learning | Submit y/n helpfulness feedback, triggers ADWIN      |
| `/feedback/stats` | GET    | Online Learning | Online learning stats + prequential log              |
| `/graph/query`    | POST   | Graph           | Execute raw Cypher queries against Neo4j             |
| `/stats`          | GET    | System          | System-wide statistics (all stores + online learner) |
| `/health`         | GET    | System          | Health check for all backing services                |

## Components by Deliverable

### D1 -- Streaming Learner & AutoML

- `online_learning.py`: River LogisticRegression with ADWIN drift detection (delta=0.002)
- `automl_tuner.py`: Optuna TPE search over retriever hyperparameters (40 trials)
- `configs/run_card.yaml`: Winning config (k=11, alpha=0.1154, SVD 256-dim, cosine)
- `Notebooks/D1.ipynb`: Original D1 notebook (preserved)

### D2 -- Retrieval Stack & Graph Build

- `ingest.py`: PDF extraction (PyMuPDF), chunking (500 char, 100 overlap), embedding (BGE), MongoDB + Qdrant storage
- `hybrid_search.py`: BM25 + Dense + Reciprocal Rank Fusion, auto-loads winning config from run_card.yaml
- `graph_build.py`: Neo4j graph (Paper, Author, Topic nodes + WROTE, HAS_TOPIC, CITES relationships)
- `app.py`: FastAPI with all endpoints
- `docker-compose.yml`: Full infrastructure stack (MongoDB, Qdrant, Neo4j, FastAPI)

### D3 -- GraphRAG Executor, Evaluation & Safety

- `graphrag_executor.py`: Core pipeline (Method A -- Pre-Filter)
  1. LLM generates Cypher from user question
  2. Cypher runs against Neo4j -> paper_ids (subgraph)
  3. Filtered BM25 + Dense + RRF retrieval (fallback to unfiltered if empty)
  4. LLM generates grounded answer with page-range citations
  5. Provenance filter verifies all citations against retrieved metadata
- `evaluate.py`: Ablation harness with three modes (vector_only, graph_only, hybrid)
  - Metrics: p95 latency, Faithfulness, Answer-Relevance, Provenance score
- `/feedback` endpoint: Synchronous River/ADWIN feedback loop for the GraphRAG pipeline
- Parameter injection: `hybrid_search.py` auto-loads AutoML winning config from YAML

## Environment Variables

| Variable        | Default                       | Description                           |
|-----------------|-------------------------------|---------------------------------------|
| `MONGO_URI`     | `mongodb://localhost:27017`   | MongoDB connection string             |
| `MONGO_DB`      | `pdf_rag`                     | MongoDB database name                 |
| `QDRANT_HOST`   | `localhost`                   | Qdrant server host                    |
| `QDRANT_PORT`   | `6333`                       | Qdrant server port                    |
| `NEO4J_URI`     | `bolt://localhost:7687`       | Neo4j Bolt connection URI             |
| `NEO4J_USER`    | `neo4j`                      | Neo4j username                        |
| `NEO4J_PASSWORD`| `password`                    | Neo4j password                        |
| `LLM_API_KEY`   | *(required for D3)*          | API key for the LLM provider          |
| `LLM_BASE_URL`  | `https://api.openai.com/v1`  | Base URL for OpenAI-compatible API    |
| `LLM_MODEL`     | `gpt-4o-mini`                | Model identifier for LLM calls        |
| `LLM_TEMPERATURE`| `0.0`                       | Sampling temperature for LLM          |
| `ADWIN_DELTA`   | `0.002`                      | ADWIN drift detection sensitivity     |
| `RIVER_LR`      | `0.01`                       | River model learning rate             |
| `RIVER_L2`      | `1e-4`                       | River model L2 regularization         |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Neo4j won't start | Check port 7687 isn't in use. Run `docker compose logs neo4j` |
| `seed_data.py` can't download PDFs | It falls back to synthetic data automatically |
| Qdrant connection refused | Wait 10-15 seconds after `docker compose up` |
| `ModuleNotFoundError` | Activate your venv and run `pip install -r requirements.txt` |
| Embedding model download slow | First run downloads ~130MB for BGE. Cached afterward |
| `/graphrag` returns 500 | Check that `LLM_API_KEY` is set in your `.env` file |
| Port 8000 conflict (Splunk, etc.) | Run uvicorn on a different port: `--port 8001`. Check with `netstat -ano \| findstr :8000` |
| `/ingest` returns `num_chunks: 0` | Data was already ingested (e.g. by `seed_data.py`). This is normal — duplicates are skipped |
| Provenance score is low | The LLM may hallucinate titles. Lower temperature helps (default 0.0) |

## Team

- **Mahmood**: Retrieval engine (BM25, dense search, RRF, evaluation metrics)
- **Afzal**: Backend infrastructure (FastAPI, Docker Compose, lazy loading)
- **Abdulsalam**: Data ingestion & graph layer (PyMuPDF, MongoDB, Neo4j)
