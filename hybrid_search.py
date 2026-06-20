"""
Hybrid Retrieval (BM25 + Dense + RRF)
======================================
Implements sparse (BM25) and dense (Qdrant) retrieval with Reciprocal Rank
Fusion for merging ranked lists.

Includes evaluation helpers: Recall@K, MRR, nDCG@K.

Usage:
    from hybrid_search import HybridSearcher, SearchResult
    searcher = HybridSearcher()
    results = searcher.search("attention mechanism", top_k=5)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from pymongo import MongoClient
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from ingest import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MONGO_COLLECTION,
    MONGO_DB,
    MONGO_URI,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    get_embedding_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single search hit."""

    chunk_id: str
    paper_id: str
    title: str
    text: str
    score: float
    page_start: int
    page_end: int
    authors: list[str] = field(default_factory=list)
    source: str = ""  # "bm25", "dense", or "rrf"

    def citation(self) -> str:
        """Return a short citation string."""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        return f'[{author_str}] "{self.title}" (pp. {self.page_start}-{self.page_end})'


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------


class HybridSearcher:
    """BM25 + Dense vector search with Reciprocal Rank Fusion."""

    def __init__(
        self,
        mongo_uri: str = MONGO_URI,
        mongo_db: str = MONGO_DB,
        mongo_collection: str = MONGO_COLLECTION,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        qdrant_collection: str = QDRANT_COLLECTION,
    ) -> None:
        self._mongo = MongoClient(mongo_uri)
        self._db = self._mongo[mongo_db]
        self._col = self._db[mongo_collection]

        self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._qdrant_collection = qdrant_collection

        self._embed_model: Optional[Any] = None

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[dict] = []

    def _ensure_bm25(self) -> None:
        """Build the BM25 index from MongoDB if not already done."""
        if self._bm25 is not None:
            return

        logger.info("Building BM25 index from MongoDB ...")
        docs = list(self._col.find({}, {"_id": 0}))
        if not docs:
            logger.warning("No documents in MongoDB -- BM25 index will be empty.")
            self._bm25_docs = []
            self._bm25 = BM25Okapi([[""]])
            return

        self._bm25_docs = docs
        tokenised = [doc["text"].lower().split() for doc in docs]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built with %d documents.", len(docs))

    def _get_embed_model(self):
        """Return the cached SentenceTransformer model."""
        if self._embed_model is None:
            self._embed_model = get_embedding_model()
        return self._embed_model

    def _embed_query(self, query: str) -> list[float]:
        """Encode a single query string."""
        model = self._get_embed_model()
        vec = model.encode([query], normalize_embeddings=True)[0]
        return vec.tolist()

    def bm25_search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Sparse (BM25) search over the MongoDB chunk corpus."""
        self._ensure_bm25()
        if not self._bm25_docs:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            doc = self._bm25_docs[idx]
            results.append(
                SearchResult(
                    chunk_id=doc["chunk_id"],
                    paper_id=doc["paper_id"],
                    title=doc.get("title", ""),
                    text=doc["text"],
                    score=float(scores[idx]),
                    page_start=doc.get("page_start", 0),
                    page_end=doc.get("page_end", 0),
                    authors=doc.get("authors", []),
                    source="bm25",
                )
            )
        return results

    def dense_search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Dense vector search via Qdrant."""
        query_vec = self._embed_query(query)
        hits = self._qdrant.query_points(
            collection_name=self._qdrant_collection,
            query=query_vec,
            limit=top_k,
        ).points

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    paper_id=payload.get("paper_id", ""),
                    title=payload.get("title", ""),
                    text=payload.get("text", ""),
                    score=float(hit.score),
                    page_start=payload.get("page_start", 0),
                    page_end=payload.get("page_end", 0),
                    authors=payload.get("authors", []),
                    source="dense",
                )
            )
        return results

    @staticmethod
    def rrf_fuse(
        results_lists: list[list[SearchResult]],
        k: int = 60,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion (RRF).

        RRF_score(d) = sum( 1 / (k + rank_i(d)) )
        """
        scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}

        for rlist in results_lists:
            for rank, result in enumerate(rlist, start=1):
                cid = result.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in best_result or result.score > best_result[cid].score:
                    best_result[cid] = result

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

        fused: list[SearchResult] = []
        for cid in sorted_ids:
            r = best_result[cid]
            fused.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    paper_id=r.paper_id,
                    title=r.title,
                    text=r.text,
                    score=scores[cid],
                    page_start=r.page_start,
                    page_end=r.page_end,
                    authors=r.authors,
                    source="rrf",
                )
            )
        return fused

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Hybrid search: BM25 + Dense fused with RRF."""
        candidate_k = top_k * 4

        bm25_results = self.bm25_search(query, top_k=candidate_k)
        dense_results = self.dense_search(query, top_k=candidate_k)

        fused = self.rrf_fuse([bm25_results, dense_results], top_k=top_k)
        return fused


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K -- fraction of relevant documents in the top-K results."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank for a single query."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain up to position k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K (binary relevance)."""
    relevances = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
    dcg = _dcg(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def run_quick_eval(
    searcher: HybridSearcher,
    eval_queries: list[dict],
) -> None:
    """Run a mini evaluation and print a metrics table."""
    print("\n" + "=" * 72)
    print(f" {'Query':<30} {'Recall@5':>10} {'MRR':>10} {'nDCG@5':>10}")
    print("-" * 72)

    all_recall, all_mrr, all_ndcg = [], [], []

    for eq in eval_queries:
        query = eq["query"]
        relevant = set(eq["relevant_ids"])

        results = searcher.search(query, top_k=10)
        retrieved_ids = [r.paper_id for r in results]

        r_at_5 = recall_at_k(retrieved_ids, relevant, k=5)
        m = mrr(retrieved_ids, relevant)
        n_at_5 = ndcg_at_k(retrieved_ids, relevant, k=5)

        all_recall.append(r_at_5)
        all_mrr.append(m)
        all_ndcg.append(n_at_5)

        short_q = (query[:27] + "...") if len(query) > 28 else query
        print(f" {short_q:<30} {r_at_5:>10.3f} {m:>10.3f} {n_at_5:>10.3f}")

    print("-" * 72)
    print(
        f" {'MEAN':<30} {np.mean(all_recall):>10.3f} "
        f"{np.mean(all_mrr):>10.3f} {np.mean(all_ndcg):>10.3f}"
    )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run an interactive search loop or a single query."""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Search")
    parser.add_argument("--query", type=str, default=None, help="Single query to run")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    searcher = HybridSearcher()

    if args.query:
        results = searcher.search(args.query, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score={r.score:.4f}, source={r.source}) ---")
            print(f"Title : {r.title}")
            print(f"Pages : {r.page_start}-{r.page_end}")
            print(f"Text  : {r.text[:200]}...")
            print(f"Cite  : {r.citation()}")
        return

    if args.interactive:
        print("Hybrid Search (type 'quit' to exit)")
        while True:
            query = input("\nQuery> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            results = searcher.search(query, top_k=args.top_k)
            for i, r in enumerate(results, 1):
                print(f"  [{i}] (score={r.score:.4f}) {r.title} -- pp. {r.page_start}-{r.page_end}")
                print(f"       {r.text[:150]}...")


if __name__ == "__main__":
    main()
