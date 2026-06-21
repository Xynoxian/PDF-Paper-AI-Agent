"""
AutoML Hyperparameter Tuning with Optuna
==========================================
Extracted from D1: searches over hybrid retrieval hyperparameters
(k, alpha, SVD dim, normalization, distance metric) using Optuna TPE.

Can be run standalone against the in-memory retriever (D1 style) or
used as a library to tune parameters for D2's database-backed searcher.

Usage:
    python automl_tuner.py --n-trials 40

The winning config is saved to configs/run_card.yaml.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import yaml
from optuna.samplers import TPESampler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED: int = 42
TOP_K_EVAL: int = 5
DEFAULT_N_TRIALS: int = 40


# ---------------------------------------------------------------------------
# In-memory hybrid retriever (for AutoML search)
# ---------------------------------------------------------------------------


class InMemoryHybridRetriever:
    """BM25 + Dense hybrid retriever with configurable hyperparameters.

    This is the D1 retriever adapted for Optuna search. It operates
    on in-memory numpy arrays rather than database backends.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        svd_dim: Optional[int] = None,
        norm: str = "l2",
        metric: str = "cosine",
        k: int = 5,
    ):
        self.alpha = alpha
        self.svd_dim = svd_dim
        self.norm = norm
        self.metric = metric
        self.k = k
        self._svd: Optional[TruncatedSVD] = None
        self._index: Optional[NearestNeighbors] = None
        self._dense: Optional[np.ndarray] = None

    def fit(self, dense_matrix: np.ndarray) -> "InMemoryHybridRetriever":
        d = dense_matrix.copy()
        if self.svd_dim and self.svd_dim < d.shape[1]:
            self._svd = TruncatedSVD(n_components=self.svd_dim, random_state=SEED)
            d = self._svd.fit_transform(d)
        if self.norm == "l2":
            d = normalize(d, norm="l2")
        self._dense = d
        nn_metric = "cosine" if self.metric in ("cosine", "dot") else "euclidean"
        self._index = NearestNeighbors(
            n_neighbors=self.k, metric=nn_metric, algorithm="brute"
        )
        self._index.fit(d)
        return self

    def query(
        self,
        q_text: str,
        q_emb: np.ndarray,
        bm25_scores_fn,
        doc_ids: list[str],
    ) -> list[str]:
        """Retrieve top-k documents by hybrid fusion.

        Args:
            q_text: Query text (for BM25).
            q_emb: Query embedding vector.
            bm25_scores_fn: Callable that returns normalized BM25 scores array.
            doc_ids: List of document IDs corresponding to corpus indices.

        Returns:
            List of retrieved document IDs.
        """
        qd = q_emb.copy().reshape(1, -1)
        if self._svd:
            qd = self._svd.transform(qd)
        if self.norm == "l2":
            qd = normalize(qd, norm="l2")

        dense_dists, dense_idx = self._index.kneighbors(qd, n_neighbors=self.k)
        dense_scores = np.zeros(len(self._dense))
        dense_scores[dense_idx[0]] = 1 - dense_dists[0]

        lex_scores = bm25_scores_fn(q_text)

        fused = self.alpha * lex_scores + (1 - self.alpha) * dense_scores
        ranked = np.argsort(fused)[::-1][: self.k]
        return [doc_ids[i] for i in ranked]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: str) -> float:
    return float(relevant in retrieved)


def ndcg_at_k(retrieved: list[str], relevant: str, k: int = 5) -> float:
    if relevant not in retrieved[:k]:
        return 0.0
    rank = retrieved[:k].index(relevant) + 1
    return 1.0 / np.log2(rank + 1)


def evaluate_retriever(
    retriever: InMemoryHybridRetriever,
    queries_text: list[str],
    queries_emb: np.ndarray,
    gold_ids: list[str],
    bm25_scores_fn,
    doc_ids: list[str],
    top_k: int = TOP_K_EVAL,
) -> dict[str, float]:
    """Evaluate a retriever on gold queries."""
    recalls, ndcgs, latencies = [], [], []
    for qt, qe, gid in zip(queries_text, queries_emb, gold_ids):
        t0 = time.perf_counter()
        retrieved = retriever.query(qt, qe, bm25_scores_fn, doc_ids)
        latencies.append(time.perf_counter() - t0)
        recalls.append(recall_at_k(retrieved, gid))
        ndcgs.append(ndcg_at_k(retrieved, gid, k=top_k))
    return {
        f"Recall@{top_k}": float(np.mean(recalls)),
        f"NDCG@{top_k}": float(np.mean(ndcgs)),
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
    }


# ---------------------------------------------------------------------------
# Optuna search
# ---------------------------------------------------------------------------


def run_automl_search(
    corpus_emb: np.ndarray,
    query_emb: np.ndarray,
    gold_queries: list[str],
    gold_doc_ids: list[str],
    doc_ids: list[str],
    bm25_scores_fn,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run Optuna TPE search over hybrid retriever hyperparameters.

    Returns:
        Dict with keys: best_params, best_metrics, baseline_metrics, study.
    """
    np.random.seed(seed)

    baseline = InMemoryHybridRetriever(
        alpha=0.5, svd_dim=None, norm="l2", metric="cosine", k=5
    ).fit(corpus_emb)
    baseline_metrics = evaluate_retriever(
        baseline, gold_queries, query_emb, gold_doc_ids,
        bm25_scores_fn, doc_ids,
    )
    logger.info("Baseline metrics: %s", baseline_metrics)

    def objective(trial: optuna.Trial) -> float:
        k = trial.suggest_int("k", 1, 20)
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        svd_dim = trial.suggest_categorical("svd_dim", [None, 64, 128, 256])
        norm = trial.suggest_categorical("norm", ["l2", "none"])
        metric = trial.suggest_categorical("metric", ["cosine", "euclidean"])

        ret = InMemoryHybridRetriever(
            alpha=alpha, svd_dim=svd_dim, norm=norm, metric=metric, k=k
        )
        try:
            ret.fit(corpus_emb)
            m = evaluate_retriever(
                ret, gold_queries, query_emb, gold_doc_ids,
                bm25_scores_fn, doc_ids,
            )
        except Exception:
            return 0.0

        latency_penalty = max(0, m["p95_latency_ms"] - 500) / 1000
        score = (m[f"NDCG@{TOP_K_EVAL}"] + m[f"Recall@{TOP_K_EVAL}"]) / 2 \
                - 0.05 * latency_penalty

        trial.set_user_attr("NDCG", m[f"NDCG@{TOP_K_EVAL}"])
        trial.set_user_attr("Recall", m[f"Recall@{TOP_K_EVAL}"])
        trial.set_user_attr("p95_ms", m["p95_latency_ms"])
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name="knn_hybrid_automl",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_metrics = {
        f"NDCG@{TOP_K_EVAL}": best.user_attrs["NDCG"],
        f"Recall@{TOP_K_EVAL}": best.user_attrs["Recall"],
        "p95_latency_ms": best.user_attrs["p95_ms"],
    }

    logger.info("Best params: %s", best.params)
    logger.info("Best metrics: %s", best_metrics)

    return {
        "best_params": best.params,
        "best_metrics": best_metrics,
        "baseline_metrics": baseline_metrics,
        "study": study,
    }


# ---------------------------------------------------------------------------
# Run card export
# ---------------------------------------------------------------------------


def save_run_card(
    result: dict[str, Any],
    adapter_summary: Optional[dict] = None,
    output_path: str | Path = "configs/run_card.yaml",
) -> None:
    """Save a YAML run card from AutoML results."""
    run_card = {
        "run_card": {
            "experiment": "AutoML-kNN-Optuna",
            "date": time.strftime("%Y-%m-%d"),
            "seed": SEED,
            "n_trials": len(result["study"].trials),
            "embed_model": "BAAI/bge-small-en-v1.5",
            "eval_k": TOP_K_EVAL,
        },
        "search_space": {
            "k": {"type": "int", "low": 1, "high": 20},
            "alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "svd_dim": {"type": "categorical", "choices": [None, 64, 128, 256]},
            "norm": {"type": "categorical", "choices": ["l2", "none"]},
            "metric": {"type": "categorical", "choices": ["cosine", "euclidean"]},
        },
        "winning_config": {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in result["best_params"].items()
        },
        "metrics": {
            "baseline": {
                k: round(float(v), 4) for k, v in result["baseline_metrics"].items()
            },
            "automl_best": {
                k: round(float(v), 4) for k, v in result["best_metrics"].items()
            },
        },
    }

    if adapter_summary:
        run_card["online_learning"] = adapter_summary

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(run_card, f, default_flow_style=False, sort_keys=False)

    logger.info("Run card saved to %s", output_path)
