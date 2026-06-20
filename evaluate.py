"""
Evaluation & Ablation Harness (D3)
====================================
Standalone script that evaluates the GraphRAG pipeline against a CSV of
queries and measures p95 latency, Faithfulness, and Answer-Relevance.

Ablation modes:
  - vector_only : BM25 + Dense hybrid search only (no graph filtering)
  - graph_only  : Graph-filtered chunks only (no unfiltered fallback)
  - hybrid      : Full GraphRAG pipeline (graph filter with fallback)

CSV format (minimal):
    query
    "How does attention work in transformers?"
    "What is BERT pre-training?"

CSV format (with expected answers for relevance scoring):
    query,expected_keywords
    "How does attention work?","attention,self-attention,multi-head"
    "What is BERT?","masked language model,pre-training,bidirectional"

Usage:
    python evaluate.py --csv queries.csv --mode hybrid
    python evaluate.py --csv queries.csv --ablation        # all three modes
    python evaluate.py --csv queries.csv --ablation --top-k 5

Output: metrics table printed to stdout + results saved to eval_results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation metrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Metrics for a single query evaluation."""

    query: str
    mode: str
    latency_ms: float
    faithfulness: float
    answer_relevance: float
    chunks_used: int
    graph_papers_found: int
    provenance_score: float
    verified_citations: int
    dropped_citations: int
    answer_length: int


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all queries for a single mode."""

    mode: str
    n_queries: int
    p95_latency_ms: float
    mean_latency_ms: float
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_provenance_score: float
    mean_chunks_used: float
    total_verified_citations: int
    total_dropped_citations: int


# ---------------------------------------------------------------------------
# Faithfulness & Answer-Relevance scorers
# ---------------------------------------------------------------------------


def compute_faithfulness(
    answer: str,
    verified_count: int,
    total_citation_count: int,
) -> float:
    """Compute faithfulness as the fraction of citations that pass provenance.

    Faithfulness = verified_citations / total_citations_found.
    If no citations are present, returns 1.0 (vacuously faithful).

    Args:
        answer: The generated answer text.
        verified_count: Number of citations passing provenance check.
        total_citation_count: Total citations found in the answer.

    Returns:
        Float in [0.0, 1.0].
    """
    if total_citation_count == 0:
        return 1.0
    return verified_count / total_citation_count


def compute_answer_relevance(
    query: str,
    answer: str,
    expected_keywords: list[str] | None = None,
) -> float:
    """Compute answer relevance using keyword overlap.

    If expected_keywords are provided, measures what fraction of them
    appear in the answer. Otherwise, measures overlap between query
    tokens and answer tokens as a proxy.

    Args:
        query: The original query string.
        answer: The generated answer text.
        expected_keywords: Optional list of keywords the answer should contain.

    Returns:
        Float in [0.0, 1.0].
    """
    answer_lower = answer.lower()

    if expected_keywords:
        keywords = [kw.strip().lower() for kw in expected_keywords if kw.strip()]
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw in answer_lower)
        return hits / len(keywords)

    query_tokens = set(query.lower().split())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                 "at", "to", "for", "of", "with", "by", "from", "and", "or",
                 "how", "what", "which", "who", "where", "when", "does", "do"}
    query_tokens -= stopwords

    if not query_tokens:
        return 0.0

    hits = sum(1 for token in query_tokens if token in answer_lower)
    return hits / len(query_tokens)


# ---------------------------------------------------------------------------
# Pipeline runners for each ablation mode
# ---------------------------------------------------------------------------


def run_vector_only(query: str, top_k: int) -> dict[str, Any]:
    """Run vector-only mode: BM25 + Dense hybrid search, no graph filtering.

    Args:
        query: The search query.
        top_k: Number of results to retrieve.

    Returns:
        Dict with answer, chunks, latency, and citation metadata.
    """
    from graphrag_executor import GraphRAGExecutor, provenance_filter

    executor = GraphRAGExecutor()

    t0 = time.perf_counter()

    chunks = executor._searcher.search(query, top_k=top_k)

    if chunks:
        raw_answer = executor._generate_answer(query, chunks)
        filtered_answer, verified, dropped, prov_score = provenance_filter(
            raw_answer, chunks,
        )
    else:
        filtered_answer = "No relevant documents found."
        verified, dropped, prov_score = [], [], 1.0

    latency_ms = (time.perf_counter() - t0) * 1000
    executor.close()

    return {
        "answer": filtered_answer,
        "chunks": chunks,
        "latency_ms": latency_ms,
        "verified": verified,
        "dropped": dropped,
        "provenance_score": prov_score,
        "graph_papers_found": 0,
    }


def run_graph_only(query: str, top_k: int) -> dict[str, Any]:
    """Run graph-only mode: only graph-filtered chunks, no fallback.

    Args:
        query: The search query.
        top_k: Number of results to retrieve.

    Returns:
        Dict with answer, chunks, latency, and citation metadata.
    """
    from graphrag_executor import GraphRAGExecutor, provenance_filter

    executor = GraphRAGExecutor()

    t0 = time.perf_counter()

    cypher, intent = executor._generate_cypher(query)
    paper_ids: list[str] = []
    if cypher:
        paper_ids = executor._execute_cypher(cypher)

    if paper_ids:
        chunks = executor._searcher.search(query, top_k=top_k, paper_ids=paper_ids)
    else:
        chunks = []

    if chunks:
        raw_answer = executor._generate_answer(query, chunks)
        filtered_answer, verified, dropped, prov_score = provenance_filter(
            raw_answer, chunks,
        )
    else:
        filtered_answer = "No graph-filtered documents found for this query."
        verified, dropped, prov_score = [], [], 1.0

    latency_ms = (time.perf_counter() - t0) * 1000
    executor.close()

    return {
        "answer": filtered_answer,
        "chunks": chunks,
        "latency_ms": latency_ms,
        "verified": verified,
        "dropped": dropped,
        "provenance_score": prov_score,
        "graph_papers_found": len(paper_ids),
    }


def run_hybrid(query: str, top_k: int) -> dict[str, Any]:
    """Run hybrid mode: full GraphRAG pipeline (graph filter with fallback).

    Args:
        query: The search query.
        top_k: Number of results to retrieve.

    Returns:
        Dict with answer, chunks, latency, and citation metadata.
    """
    from graphrag_executor import GraphRAGExecutor

    executor = GraphRAGExecutor()

    t0 = time.perf_counter()
    resp = executor.query(query, top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    executor.close()

    return {
        "answer": resp.answer,
        "chunks": [],
        "latency_ms": latency_ms,
        "verified": resp.verified_citations,
        "dropped": resp.dropped_citations,
        "provenance_score": resp.provenance_score,
        "graph_papers_found": resp.graph_papers_found,
        "chunks_used": resp.chunks_used,
    }


MODE_RUNNERS = {
    "vector_only": run_vector_only,
    "graph_only": run_graph_only,
    "hybrid": run_hybrid,
}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def load_queries(csv_path: str) -> list[dict[str, Any]]:
    """Load evaluation queries from a CSV file.

    The CSV must have a 'query' column. Optionally includes
    'expected_keywords' (comma-separated keywords for relevance scoring).

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of dicts with 'query' and optional 'expected_keywords'.
    """
    queries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry: dict[str, Any] = {"query": row["query"].strip()}
            if "expected_keywords" in row and row["expected_keywords"]:
                entry["expected_keywords"] = [
                    kw.strip() for kw in row["expected_keywords"].split(",")
                ]
            else:
                entry["expected_keywords"] = None
            queries.append(entry)
    return queries


def evaluate_mode(
    queries: list[dict[str, Any]],
    mode: str,
    top_k: int = 5,
) -> tuple[list[EvalResult], AggregateMetrics]:
    """Run all queries through a specific pipeline mode and collect metrics.

    Args:
        queries: List of query dicts from load_queries().
        mode: One of 'vector_only', 'graph_only', 'hybrid'.
        top_k: Number of chunks to retrieve per query.

    Returns:
        Tuple of (per_query_results, aggregate_metrics).
    """
    runner = MODE_RUNNERS[mode]
    results: list[EvalResult] = []

    print(f"\n{'='*72}")
    print(f"  Evaluating mode: {mode.upper()}")
    print(f"{'='*72}")

    for i, q in enumerate(queries, 1):
        query_text = q["query"]
        expected_kw = q.get("expected_keywords")

        print(f"  [{i}/{len(queries)}] {query_text[:60]}...", end="", flush=True)

        try:
            out = runner(query_text, top_k)
        except Exception as exc:
            logger.error("Query failed: %s -- %s", query_text[:40], exc)
            results.append(EvalResult(
                query=query_text, mode=mode, latency_ms=0.0,
                faithfulness=0.0, answer_relevance=0.0, chunks_used=0,
                graph_papers_found=0, provenance_score=0.0,
                verified_citations=0, dropped_citations=0, answer_length=0,
            ))
            print(" FAILED")
            continue

        total_cites = len(out["verified"]) + len(out["dropped"])
        faith = compute_faithfulness(out["answer"], len(out["verified"]), total_cites)
        relevance = compute_answer_relevance(query_text, out["answer"], expected_kw)

        chunks_used = out.get("chunks_used", len(out.get("chunks", [])))

        result = EvalResult(
            query=query_text,
            mode=mode,
            latency_ms=round(out["latency_ms"], 2),
            faithfulness=round(faith, 4),
            answer_relevance=round(relevance, 4),
            chunks_used=chunks_used,
            graph_papers_found=out["graph_papers_found"],
            provenance_score=round(out["provenance_score"], 4),
            verified_citations=len(out["verified"]),
            dropped_citations=len(out["dropped"]),
            answer_length=len(out["answer"]),
        )
        results.append(result)
        print(f" {result.latency_ms:.0f}ms | faith={faith:.2f} | rel={relevance:.2f}")

    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    agg = AggregateMetrics(
        mode=mode,
        n_queries=len(results),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        mean_latency_ms=round(float(np.mean(latencies)), 2) if latencies else 0.0,
        mean_faithfulness=round(float(np.mean([r.faithfulness for r in results])), 4),
        mean_answer_relevance=round(float(np.mean([r.answer_relevance for r in results])), 4),
        mean_provenance_score=round(float(np.mean([r.provenance_score for r in results])), 4),
        mean_chunks_used=round(float(np.mean([r.chunks_used for r in results])), 2),
        total_verified_citations=sum(r.verified_citations for r in results),
        total_dropped_citations=sum(r.dropped_citations for r in results),
    )

    return results, agg


def print_summary_table(aggregates: list[AggregateMetrics]) -> None:
    """Print a formatted comparison table across ablation modes.

    Args:
        aggregates: List of AggregateMetrics from each mode.
    """
    print(f"\n{'='*90}")
    print(f"  {'Mode':<15} {'p95 Lat(ms)':>12} {'Mean Lat':>10} "
          f"{'Faith':>8} {'Relevance':>10} {'Provenance':>11} {'Chunks':>8}")
    print(f"{'-'*90}")
    for a in aggregates:
        print(f"  {a.mode:<15} {a.p95_latency_ms:>12.1f} {a.mean_latency_ms:>10.1f} "
              f"{a.mean_faithfulness:>8.3f} {a.mean_answer_relevance:>10.3f} "
              f"{a.mean_provenance_score:>11.3f} {a.mean_chunks_used:>8.1f}")
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the evaluation harness from the command line."""
    parser = argparse.ArgumentParser(
        description="GraphRAG Evaluation & Ablation Harness",
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to CSV file with evaluation queries",
    )
    parser.add_argument(
        "--mode", type=str, default="hybrid",
        choices=["vector_only", "graph_only", "hybrid"],
        help="Pipeline mode to evaluate (default: hybrid)",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run all three modes for ablation comparison",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json",
        help="Path to save JSON results (default: eval_results.json)",
    )
    args = parser.parse_args()

    queries = load_queries(args.csv)
    print(f"Loaded {len(queries)} queries from {args.csv}")

    modes = ["vector_only", "graph_only", "hybrid"] if args.ablation else [args.mode]

    all_results: list[dict] = []
    all_aggregates: list[AggregateMetrics] = []

    for mode in modes:
        results, agg = evaluate_mode(queries, mode, top_k=args.top_k)
        all_results.extend([asdict(r) for r in results])
        all_aggregates.append(agg)

    print_summary_table(all_aggregates)

    output = {
        "config": {
            "csv": args.csv,
            "top_k": args.top_k,
            "modes": modes,
            "n_queries": len(queries),
        },
        "aggregates": [asdict(a) for a in all_aggregates],
        "per_query": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
