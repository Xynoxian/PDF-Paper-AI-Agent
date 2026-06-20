"""
Online Learning with River + ADWIN Drift Detection
====================================================
Provides adaptive hybrid fusion weight learning from user feedback,
with concept drift detection via ADWIN.

This module was extracted from D1 and adapted to work with D2's retrieval
infrastructure. It provides:

  - RiverHybridAdapter: incrementally learns from binary feedback (helpful y/n)
    to predict optimal retrieval fusion parameters per query.
  - ADWIN drift detection: flags when feedback patterns shift, triggering
    re-adaptation. This is critical for D3's GraphRAG executor, where the
    adapter will learn when to favor graph-guided vs vector-only retrieval.

Usage:
    from online_learning import RiverHybridAdapter

    adapter = RiverHybridAdapter()

    # On each query-feedback pair:
    adapter.learn(query_text, bm25_top_score, dense_top_score, helpful=1)

    # Get predicted fusion alpha for a new query:
    alpha = adapter.predict_alpha(query_text, bm25_top_score, dense_top_score)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from river import drift, linear_model, metrics, optim, preprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ADWIN_DELTA: float = float(os.getenv("ADWIN_DELTA", "0.002"))
LEARNING_RATE: float = float(os.getenv("RIVER_LR", "0.01"))
L2_REGULARIZATION: float = float(os.getenv("RIVER_L2", "1e-4"))


# ---------------------------------------------------------------------------
# Prequential log entry
# ---------------------------------------------------------------------------


@dataclass
class FeedbackEntry:
    """A single feedback step in the prequential log."""
    step: int
    accuracy: float
    drift_detected: bool
    predicted: Optional[int] = None
    actual: Optional[int] = None


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------


class RiverHybridAdapter:
    """Incrementally learns adaptive retrieval parameters from binary feedback.

    The adapter takes per-query features (BM25 scores, dense scores, query
    properties) and learns to predict whether results will be helpful.
    ADWIN monitors the error stream for concept drift.

    This feeds into the retrieval pipeline in two ways:
      1. predict_alpha() returns an adaptive BM25/dense fusion weight.
      2. drift_detected property signals when the system should re-evaluate
         its retrieval strategy (used by D3's GraphRAG executor).
    """

    def __init__(
        self,
        adwin_delta: float = ADWIN_DELTA,
        learning_rate: float = LEARNING_RATE,
        l2: float = L2_REGULARIZATION,
    ) -> None:
        self.model = (
            preprocessing.StandardScaler()
            | linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=learning_rate),
                l2=l2,
            )
        )
        self.adwin = drift.ADWIN(delta=adwin_delta)
        self.metric = metrics.Accuracy()
        self.prequential: list[FeedbackEntry] = []
        self.n_drifts: int = 0
        self._last_drift_step: int = -1

    @property
    def total_steps(self) -> int:
        return len(self.prequential)

    @property
    def current_accuracy(self) -> float:
        return self.metric.get()

    @property
    def drift_detected(self) -> bool:
        """Whether drift was detected on the most recent step."""
        if not self.prequential:
            return False
        return self.prequential[-1].drift_detected

    @staticmethod
    def _build_features(
        query_text: str,
        bm25_top_score: float,
        dense_top_score: float,
        bm25_mean_score: float = 0.0,
        dense_mean_score: float = 0.0,
        dense_std_score: float = 0.0,
        bm25_nonzero_count: float = 0.0,
    ) -> dict[str, float]:
        """Build the feature dict for the River model."""
        tokens = query_text.split()
        return {
            "bm25_top": bm25_top_score,
            "bm25_mean": bm25_mean_score,
            "bm25_nonzero": bm25_nonzero_count,
            "dense_top": dense_top_score,
            "dense_mean": dense_mean_score,
            "dense_std": dense_std_score,
            "q_len": float(len(tokens)),
            "q_avg_word_len": float(np.mean([len(t) for t in tokens]) if tokens else 0),
        }

    def learn(
        self,
        query_text: str,
        bm25_top_score: float,
        dense_top_score: float,
        helpful: int,
        bm25_mean_score: float = 0.0,
        dense_mean_score: float = 0.0,
        dense_std_score: float = 0.0,
        bm25_nonzero_count: float = 0.0,
    ) -> FeedbackEntry:
        """Learn from one feedback instance.

        Args:
            query_text: The original query string.
            bm25_top_score: Top BM25 score for this query.
            dense_top_score: Top dense/cosine score for this query.
            helpful: 1 if user found result helpful, 0 otherwise.
            bm25_mean_score: Mean BM25 score across results.
            dense_mean_score: Mean dense score across results.
            dense_std_score: Std dev of dense scores.
            bm25_nonzero_count: Number of non-zero BM25 scores.

        Returns:
            The FeedbackEntry for this step.
        """
        x = self._build_features(
            query_text, bm25_top_score, dense_top_score,
            bm25_mean_score, dense_mean_score, dense_std_score,
            bm25_nonzero_count,
        )

        y_pred = self.model.predict_one(x)
        self.model.learn_one(x, helpful)

        drift_flag = False
        if y_pred is not None:
            self.metric.update(helpful, y_pred)
            self.adwin.update(int(helpful != y_pred))
            if self.adwin.drift_detected:
                self.n_drifts += 1
                self._last_drift_step = len(self.prequential)
                drift_flag = True
                logger.info(
                    "ADWIN drift detected at step %d (total drifts: %d)",
                    len(self.prequential), self.n_drifts,
                )

        entry = FeedbackEntry(
            step=len(self.prequential),
            accuracy=self.metric.get(),
            drift_detected=drift_flag,
            predicted=int(y_pred) if y_pred is not None else None,
            actual=helpful,
        )
        self.prequential.append(entry)
        return entry

    def predict_alpha(
        self,
        query_text: str,
        bm25_top_score: float,
        dense_top_score: float,
        bm25_mean_score: float = 0.0,
        dense_mean_score: float = 0.0,
        dense_std_score: float = 0.0,
        bm25_nonzero_count: float = 0.0,
    ) -> float:
        """Predict the optimal BM25 fusion weight (alpha) for a query.

        Returns a value in [0.2, 0.8] representing the BM25 weight.
        Higher = more BM25, lower = more dense.
        """
        x = self._build_features(
            query_text, bm25_top_score, dense_top_score,
            bm25_mean_score, dense_mean_score, dense_std_score,
            bm25_nonzero_count,
        )
        prob = self.model.predict_proba_one(x)
        if prob is None:
            return 0.5
        return 0.2 + 0.6 * prob.get(1, 0.5)

    def get_summary(self) -> dict[str, Any]:
        """Return a summary dict of the adapter's state."""
        return {
            "total_steps": self.total_steps,
            "current_accuracy": round(self.current_accuracy, 4),
            "n_drifts": self.n_drifts,
            "last_drift_step": self._last_drift_step,
        }

    def get_prequential_data(self) -> list[dict]:
        """Return prequential log as a list of dicts (for plotting/export)."""
        return [
            {
                "step": e.step,
                "accuracy": round(e.accuracy, 4),
                "drift": e.drift_detected,
            }
            for e in self.prequential
        ]

    def save_state(self, path: str | Path) -> None:
        """Save the prequential log and summary to JSON."""
        data = {
            "summary": self.get_summary(),
            "prequential": self.get_prequential_data(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Adapter state saved to %s", path)


# ---------------------------------------------------------------------------
# Helper: extract scores from search results for the adapter
# ---------------------------------------------------------------------------


def extract_retrieval_scores(
    bm25_results: list,
    dense_results: list,
) -> dict[str, float]:
    """Extract summary scores from BM25 and dense result lists.

    Works with any objects that have a .score attribute
    (e.g. hybrid_search.SearchResult).

    Returns dict with keys matching RiverHybridAdapter.learn() kwargs.
    """
    bm25_scores = [r.score for r in bm25_results] if bm25_results else [0.0]
    dense_scores = [r.score for r in dense_results] if dense_results else [0.0]

    return {
        "bm25_top_score": max(bm25_scores),
        "bm25_mean_score": float(np.mean(bm25_scores)),
        "bm25_nonzero_count": float(sum(1 for s in bm25_scores if s > 0)),
        "dense_top_score": max(dense_scores),
        "dense_mean_score": float(np.mean(dense_scores)),
        "dense_std_score": float(np.std(dense_scores)),
    }
