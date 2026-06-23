"""
FastAPI Application
====================
REST API exposing the ingestion pipeline, hybrid search, graph queries,
online learning feedback, and system statistics.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Lifespan -- background BM25 pre-warm
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the BM25 index in the background as soon as the server starts."""
    def _prewarm() -> None:
        try:
            logger.info("Pre-warm: building BM25 index in background ...")
            searcher = _get_searcher()
            searcher._ensure_bm25()
            logger.info("Pre-warm: BM25 index ready")
        except Exception as exc:
            logger.warning("Pre-warm failed (will retry on first /search): %s", exc)

    t = threading.Thread(target=_prewarm, daemon=True, name="bm25-prewarm")
    t.start()
    yield

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF-Papers AI Agent API",
    description=(
        "Hybrid retrieval (BM25+Dense+RRF), Neo4j graph queries, "
        "online learning feedback (River+ADWIN), and PDF ingestion."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_searcher = None
_graph = None
_adapter = None
_executor = None


def _get_searcher():
    """Lazy singleton for the HybridSearcher (BM25 + Dense + RRF)."""
    global _searcher
    if _searcher is None:
        from hybrid_search import HybridSearcher
        _searcher = HybridSearcher()
    return _searcher


def _get_graph():
    """Lazy singleton for the Neo4j KnowledgeGraph."""
    global _graph
    if _graph is None:
        from graph_build import KnowledgeGraph
        _graph = KnowledgeGraph()
    return _graph


def _get_adapter():
    """Lazy singleton for the River online learning adapter (D1)."""
    global _adapter
    if _adapter is None:
        from online_learning import RiverHybridAdapter
        _adapter = RiverHybridAdapter()
    return _adapter


def _get_executor():
    """Lazy singleton for the GraphRAG executor (D3).

    Shares the searcher and graph instances with the rest of the app
    to avoid duplicate connections.
    """
    global _executor
    if _executor is None:
        from graphrag_executor import GraphRAGExecutor
        _executor = GraphRAGExecutor(
            searcher=_get_searcher(),
            graph=_get_graph(),
        )
    return _executor


def _after_ingest() -> None:
    """Sync Neo4j graph and refresh the BM25 index after new papers are ingested."""
    try:
        from graph_build import populate_from_mongodb
        populate_from_mongodb(graph=_get_graph())
        logger.info("Neo4j graph synced after ingestion.")
    except Exception as exc:
        logger.warning("Graph sync after ingest failed: %s", exc)

    try:
        _get_searcher().invalidate_bm25()
    except Exception as exc:
        logger.warning("BM25 invalidate after ingest failed: %s", exc)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    pdf_dir: str = Field(..., description="Absolute path to directory containing PDFs")
    batch_size: int = Field(default=32, description="Embedding batch size")


class IngestResponse(BaseModel):
    status: str
    num_chunks: int
    elapsed_seconds: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class SearchHit(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    text: str
    score: float
    page_start: int
    page_end: int
    authors: list[str]
    source: str
    citation: str


class SearchResponse(BaseModel):
    query: str
    num_results: int
    elapsed_ms: float
    results: list[SearchHit]


class GraphQueryRequest(BaseModel):
    cypher: str = Field(..., description="Cypher query to execute")
    params: dict[str, Any] = Field(default_factory=dict, description="Query parameters")


class GraphQueryResponse(BaseModel):
    records: list[dict[str, Any]]
    num_records: int
    elapsed_ms: float


class FeedbackRequest(BaseModel):
    """User feedback on a search result."""
    query: str = Field(..., description="The original search query")
    helpful: int = Field(..., ge=0, le=1, description="1 if helpful, 0 otherwise")
    bm25_top_score: float = Field(default=0.0, description="Top BM25 score from the search")
    dense_top_score: float = Field(default=0.0, description="Top dense score from the search")


class FeedbackResponse(BaseModel):
    status: str
    step: int
    current_accuracy: float
    drift_detected: bool
    total_drifts: int


class AdapterStatsResponse(BaseModel):
    total_steps: int
    current_accuracy: float
    n_drifts: int
    last_drift_step: int
    prequential: list[dict]


class GraphRAGRequest(BaseModel):
    """Request body for the GraphRAG pipeline endpoint."""
    query: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class GraphRAGResponseModel(BaseModel):
    """Response from the GraphRAG pipeline with provenance metadata."""
    answer: str
    citations: list[str]
    verified_citations: list[str]
    dropped_citations: list[str]
    chunks_used: int
    graph_filter_applied: bool
    graph_papers_found: int
    cypher_generated: Optional[str]
    intent: str
    fallback: bool
    provenance_score: float
    bm25_top_score: float
    dense_top_score: float
    elapsed_ms: float


class StatsResponse(BaseModel):
    mongo_papers: int
    mongo_chunks: int
    qdrant_vectors: int
    graph: dict[str, int]
    online_learning: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    mongo: str
    qdrant: str
    neo4j: str
    papers: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdfs(req: IngestRequest) -> IngestResponse:
    """Ingest all PDFs from a directory."""
    from ingest import ingest_directory

    t0 = time.time()
    try:
        chunks = ingest_directory(req.pdf_dir, batch_size=req.batch_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.time() - t0
    _after_ingest()
    return IngestResponse(status="ok", num_chunks=len(chunks), elapsed_seconds=round(elapsed, 2))


_PAPERS_DIR = Path(__file__).resolve().parent / "papers"


@app.post("/upload", tags=["Ingestion"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and ingest it immediately."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    from ingest import ingest_pdf

    _PAPERS_DIR.mkdir(exist_ok=True)
    dest = _PAPERS_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    t0 = time.time()
    try:
        chunks = ingest_pdf(str(dest))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.time() - t0
    _after_ingest()
    return {
        "status": "ok",
        "filename": file.filename,
        "num_chunks": len(chunks),
        "elapsed_seconds": round(elapsed, 2),
    }


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def hybrid_search(req: SearchRequest) -> SearchResponse:
    """Hybrid BM25 + Dense search with Reciprocal Rank Fusion."""
    searcher = _get_searcher()

    t0 = time.perf_counter()
    try:
        results = searcher.search(req.query, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    hits = [
        SearchHit(
            chunk_id=r.chunk_id,
            paper_id=r.paper_id,
            title=r.title,
            text=r.text,
            score=round(r.score, 6),
            page_start=r.page_start,
            page_end=r.page_end,
            authors=r.authors,
            source=r.source,
            citation=r.citation(),
        )
        for r in results
    ]

    return SearchResponse(
        query=req.query,
        num_results=len(hits),
        elapsed_ms=round(elapsed_ms, 2),
        results=hits,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Online Learning"])
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Submit y/n helpfulness feedback for a specific query.

    This endpoint synchronously calls the River online learner to:
      1. Update the model weights based on the feedback.
      2. Trigger ADWIN drift detection on the error stream.
      3. Return the current accuracy and drift status.

    The adapter learns from (bm25_score, dense_score, query_features)
    to predict whether results will be helpful. ADWIN monitors the
    error stream for concept drift, signaling when the retrieval
    strategy needs re-evaluation.

    Tip: after calling /graphrag, pass the returned bm25_top_score and
    dense_top_score into this endpoint for accurate feature learning.
    """
    adapter = _get_adapter()

    try:
        entry = adapter.learn(
            query_text=req.query,
            bm25_top_score=req.bm25_top_score,
            dense_top_score=req.dense_top_score,
            helpful=req.helpful,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback learning failed: {exc}",
        )

    return FeedbackResponse(
        status="ok",
        step=entry.step,
        current_accuracy=round(adapter.current_accuracy, 4),
        drift_detected=entry.drift_detected,
        total_drifts=adapter.n_drifts,
    )


@app.get("/feedback/stats", response_model=AdapterStatsResponse, tags=["Online Learning"])
async def adapter_stats() -> AdapterStatsResponse:
    """Get River online learning adapter statistics and prequential log."""
    adapter = _get_adapter()
    summary = adapter.get_summary()

    return AdapterStatsResponse(
        total_steps=summary["total_steps"],
        current_accuracy=summary["current_accuracy"],
        n_drifts=summary["n_drifts"],
        last_drift_step=summary["last_drift_step"],
        prequential=adapter.get_prequential_data(),
    )


@app.post("/graph/query", response_model=GraphQueryResponse, tags=["Graph"])
async def graph_query(req: GraphQueryRequest) -> GraphQueryResponse:
    """Execute an arbitrary Cypher query against the Neo4j graph."""
    graph = _get_graph()

    t0 = time.perf_counter()
    try:
        with graph._driver.session() as session:
            result = session.run(req.cypher, **req.params)
            records = [dict(record) for record in result]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cypher error: {exc}")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return GraphQueryResponse(
        records=records,
        num_records=len(records),
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.post("/graphrag", response_model=GraphRAGResponseModel, tags=["GraphRAG"])
async def graphrag_query(req: GraphRAGRequest) -> GraphRAGResponseModel:
    """GraphRAG pipeline: graph-filtered retrieval + LLM answer generation.

    End-to-end pipeline (Method A -- Pre-Filter):
      1. LLM generates Cypher from the question
      2. Neo4j returns matching paper IDs (subgraph extraction)
      3. Hybrid search runs filtered to those papers (fallback if empty)
      4. LLM generates a grounded answer with page-range citations
      5. Provenance filter verifies citations against retrieved metadata
    """
    executor = _get_executor()

    t0 = time.perf_counter()
    try:
        resp = executor.query(req.query, top_k=req.top_k)
    except Exception as exc:
        logger.error("GraphRAG query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return GraphRAGResponseModel(
        answer=resp.answer,
        citations=resp.citations,
        verified_citations=resp.verified_citations,
        dropped_citations=resp.dropped_citations,
        chunks_used=resp.chunks_used,
        graph_filter_applied=resp.graph_filter_applied,
        graph_papers_found=resp.graph_papers_found,
        cypher_generated=resp.cypher_generated,
        intent=resp.intent,
        fallback=resp.fallback,
        provenance_score=resp.provenance_score,
        bm25_top_score=resp.bm25_top_score,
        dense_top_score=resp.dense_top_score,
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.post("/graphrag/tuned", response_model=GraphRAGResponseModel, tags=["GraphRAG"])
async def graphrag_tuned_query(req: GraphRAGRequest) -> GraphRAGResponseModel:
    """GraphRAG pipeline using the QLoRA fine-tuned SLM for answer generation.

    Same retrieval pipeline as /graphrag but routes answer generation through
    the locally fine-tuned small language model instead of the cloud LLM.
    """
    from graphrag_executor import GraphRAGExecutor

    executor = GraphRAGExecutor(
        searcher=_get_searcher(),
        graph=_get_graph(),
        use_tuned_slm=True,
    )

    t0 = time.perf_counter()
    try:
        resp = executor.query(req.query, top_k=req.top_k)
    except Exception as exc:
        logger.error("GraphRAG (tuned) query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return GraphRAGResponseModel(
        answer=resp.answer,
        citations=resp.citations,
        verified_citations=resp.verified_citations,
        dropped_citations=resp.dropped_citations,
        chunks_used=resp.chunks_used,
        graph_filter_applied=resp.graph_filter_applied,
        graph_papers_found=resp.graph_papers_found,
        cypher_generated=resp.cypher_generated,
        intent=resp.intent,
        fallback=resp.fallback,
        provenance_score=resp.provenance_score,
        bm25_top_score=resp.bm25_top_score,
        dense_top_score=resp.dense_top_score,
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def system_stats() -> StatsResponse:
    """Return counts of papers, chunks, graph nodes, and online learning state."""
    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    from ingest import (
        MONGO_COLLECTION,
        MONGO_DB,
        MONGO_URI,
        QDRANT_COLLECTION,
        QDRANT_HOST,
        QDRANT_PORT,
        count_papers,
    )

    mongo_papers = -1
    mongo_count = -1
    try:
        mongo = MongoClient(MONGO_URI)
        col = mongo[MONGO_DB][MONGO_COLLECTION]
        mongo_count = col.count_documents({})
        mongo_papers = count_papers(col)
    except Exception:
        pass

    try:
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        info = qc.get_collection(QDRANT_COLLECTION)
        qdrant_count = info.points_count or 0
    except Exception:
        qdrant_count = -1

    try:
        graph = _get_graph()
        graph_stats = graph.stats()
    except Exception:
        graph_stats = {}

    adapter = _get_adapter()
    ol_stats = adapter.get_summary()

    return StatsResponse(
        mongo_papers=mongo_papers,
        mongo_chunks=mongo_count,
        qdrant_vectors=qdrant_count,
        graph=graph_stats,
        online_learning=ol_stats,
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check -- verify connectivity to all backing services."""
    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    from ingest import MONGO_URI, MONGO_DB, MONGO_COLLECTION, QDRANT_HOST, QDRANT_PORT, count_papers

    papers_count = -1
    try:
        mc = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        mc.server_info()
        mongo_status = "ok"
        papers_count = count_papers(mc[MONGO_DB][MONGO_COLLECTION])
    except Exception as exc:
        mongo_status = f"error: {exc}"

    try:
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
        qc.get_collections()
        qdrant_status = "ok"
    except Exception as exc:
        qdrant_status = f"error: {exc}"

    try:
        g = _get_graph()
        with g._driver.session() as session:
            session.run("RETURN 1")
        neo4j_status = "ok"
    except Exception as exc:
        neo4j_status = f"error: {exc}"

    overall = "healthy" if all(s == "ok" for s in [mongo_status, qdrant_status, neo4j_status]) else "degraded"
    return HealthResponse(
        status=overall,
        mongo=mongo_status,
        qdrant=qdrant_status,
        neo4j=neo4j_status,
        papers=max(papers_count, 0),
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
async def frontend_root():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
