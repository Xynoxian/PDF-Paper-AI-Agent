# PDF-Papers AI Agent

Hybrid Retrieval + GraphRAG with Online Learning, AutoML, and Provenance Safety (CSAI415).

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

## Quick Start (Local Python + Dockerized Services)

This is the recommended setup for development.

### Step 1: Clone and switch to branch

```bash
git clone https://github.com/Xynoxian/PDF-Paper-AI-Agent.git
cd PDF-Paper-AI-Agent
git checkout Testing_Zone
```

### Step 2: Start backing services

```bash
docker compose up -d mongodb qdrant neo4j
```

Wait ~15 seconds for all three services to initialize. You can check with:

```bash
docker compose ps
```

All three containers should show `running` (and `healthy` for MongoDB/Neo4j).

### Step 3: Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (CMD):
.venv\Scripts\activate.bat

# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 4: Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your LLM API key:

```
LLM_API_KEY=sk-your-actual-api-key-here
```

The other defaults work as-is for local Docker services. If you use a non-OpenAI
provider, also update `LLM_BASE_URL` and `LLM_MODEL`.

### Step 5: Seed sample data

```bash
python seed_data.py
```

This downloads 5 arXiv PDFs (Transformer, BERT, RAG, LLaMA, LLaVA), ingests them
into MongoDB + Qdrant, builds the Neo4j knowledge graph, and runs a quick retrieval
evaluation. If PDF downloads fail, it falls back to synthetic data automatically.

### Step 6: Run the API server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** If port 8000 is taken (e.g. by Splunk, which defaults to 8000), use a
> different port: `uvicorn app:app --host 0.0.0.0 --port 8001 --reload`

Swagger UI is at **http://localhost:8000/docs** (or whichever port you chose).

## Quick Start (Full Docker Compose)

Runs everything (MongoDB, Qdrant, Neo4j, and the FastAPI app) in containers:

```bash
cp .env.example .env
# Edit .env to set LLM_API_KEY

docker compose up -d --build
```

Wait ~2 minutes for the embedding model to download during first build. Then:

```bash
docker compose exec app python seed_data.py
```

API is at **http://localhost:8000/docs**.

## Testing the Endpoints

> **Windows PowerShell note:** PowerShell aliases `curl` to `Invoke-WebRequest`, which
> uses different syntax. The examples below show both **Bash** (Linux/Mac/Git Bash) and
> **PowerShell** commands. In PowerShell, use `curl.exe` to call the real curl, or use
> `Invoke-RestMethod` as shown.

### Health check

**Bash:**
```bash
curl http://localhost:8000/health
```

**PowerShell:**
```powershell
Invoke-RestMethod http://localhost:8000/health
```

All three services (mongo, qdrant, neo4j) should return `"ok"`.

### Ingest PDFs

**Bash:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_dir": "papers"}'
```

**PowerShell:**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ingest -ContentType "application/json" -Body '{"pdf_dir": "papers"}'
```

### Hybrid search (D2)

**Bash:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanism in transformers", "top_k": 5}'
```

**PowerShell:**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/search -ContentType "application/json" -Body '{"query": "attention mechanism in transformers", "top_k": 5}'
```

Returns BM25 + Dense results fused via RRF, using the AutoML-optimized k=11 from
`configs/run_card.yaml`.

### GraphRAG query (D3)

**Bash:**
```bash
curl -X POST http://localhost:8000/graphrag \
  -H "Content-Type: application/json" \
  -d '{"query": "What papers has Vaswani written about attention?", "top_k": 5}'
```

**PowerShell:**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/graphrag -ContentType "application/json" -Body '{"query": "What papers has Vaswani written about attention?", "top_k": 5}'
```

Full pipeline response includes:
- `answer` — LLM-generated answer with page-range citations
- `verified_citations` — citations that passed provenance filtering
- `dropped_citations` — citations removed for failing provenance check
- `provenance_score` — fraction of citations verified (1.0 = all valid)
- `cypher_generated` — the Cypher query the LLM produced
- `graph_filter_applied` — whether graph filtering was used
- `fallback` — whether it fell back to unfiltered search
- `bm25_top_score` / `dense_top_score` — for passing to `/feedback`

### Submit feedback (D1 + D3)

After a `/graphrag` query, submit helpfulness feedback:

**Bash:**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What papers has Vaswani written about attention?",
    "helpful": 1,
    "bm25_top_score": 12.5,
    "dense_top_score": 0.87
  }'
```

**PowerShell:**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/feedback -ContentType "application/json" -Body '{"query": "What papers has Vaswani written about attention?", "helpful": 1, "bm25_top_score": 12.5, "dense_top_score": 0.87}'
```

This synchronously:
1. Updates the River LogisticRegression model weights
2. Triggers ADWIN drift detection on the error stream
3. Returns current accuracy and drift status

### Check online learning stats

**Bash:**
```bash
curl http://localhost:8000/feedback/stats
```

**PowerShell:**
```powershell
Invoke-RestMethod http://localhost:8000/feedback/stats
```

Returns total steps, accuracy, drift count, and the full prequential log.

### Graph query (raw Cypher)

**Bash:**
```bash
curl -X POST http://localhost:8000/graph/query \
  -H "Content-Type: application/json" \
  -d '{"cypher": "MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name, p.title LIMIT 10"}'
```

**PowerShell:**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/graph/query -ContentType "application/json" -Body '{"cypher": "MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name, p.title LIMIT 10"}'
```

### System stats

**Bash:**
```bash
curl http://localhost:8000/stats
```

**PowerShell:**
```powershell
Invoke-RestMethod http://localhost:8000/stats
```

## Running the Evaluation & Ablation Harness (D3)

The evaluation script tests the pipeline across three modes and reports
p95 latency, Faithfulness, and Answer-Relevance.

### Single mode

```bash
python evaluate.py --csv eval_queries.csv --mode hybrid --top-k 5
```

### Full ablation (all three modes compared)

```bash
python evaluate.py --csv eval_queries.csv --ablation --top-k 5
```

Modes:
| Mode | Description |
|------|-------------|
| `vector_only` | BM25 + Dense hybrid search only, no graph filtering |
| `graph_only` | Only graph-filtered chunks, no unfiltered fallback |
| `hybrid` | Full GraphRAG pipeline (graph filter with fallback) |

Output: metrics table printed to stdout + results saved to `eval_results.json`.

### CSV format

Minimal (just queries):
```csv
query
"How does attention work in transformers?"
"What is BERT pre-training?"
```

With expected keywords (for answer-relevance scoring):
```csv
query,expected_keywords
"How does attention work?","attention,self-attention,multi-head,scaled dot-product"
"What is BERT?","masked language model,pre-training,bidirectional"
```

A sample `eval_queries.csv` with 5 queries is included in the repo.

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
├── docker-compose.yml      # MongoDB, Qdrant, Neo4j, FastAPI
├── Dockerfile              # App container image
├── requirements.txt        # All Python dependencies (D1 + D2 + D3)
├── .env.example            # Environment variable template
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
