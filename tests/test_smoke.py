"""
Smoke Tests (D4)
=================
Lightweight pytest smoke tests that verify imports, configuration loading,
data pipeline classes, and basic functionality without requiring running
database services. These run offline and fast (~2s total).

Usage:
    pytest tests/test_smoke.py -v
    pytest tests/test_smoke.py -v -k "test_imports"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Import tests — verify all modules load without crashing
# ---------------------------------------------------------------------------

class TestImports:
    """Verify every project module can be imported."""

    def test_import_ingest(self):
        import ingest
        assert hasattr(ingest, "ingest_pdf")
        assert hasattr(ingest, "ingest_directory")

    def test_import_hybrid_search(self):
        import hybrid_search
        assert hasattr(hybrid_search, "HybridSearcher")
        assert hasattr(hybrid_search, "SearchResult")

    def test_import_graph_build(self):
        import graph_build
        assert hasattr(graph_build, "KnowledgeGraph")

    def test_import_graphrag_executor(self):
        import graphrag_executor
        assert hasattr(graphrag_executor, "GraphRAGExecutor")
        assert hasattr(graphrag_executor, "provenance_filter")

    def test_import_online_learning(self):
        import online_learning
        assert hasattr(online_learning, "RiverHybridAdapter")

    def test_import_evaluate(self):
        import evaluate
        assert hasattr(evaluate, "compute_faithfulness")
        assert hasattr(evaluate, "compute_answer_relevance")

    def test_import_seed_data(self):
        import seed_data
        assert hasattr(seed_data, "generate_synthetic_data")

    def test_import_slm_tuner(self):
        import slm_tuner
        assert hasattr(slm_tuner, "fine_tune")
        assert hasattr(slm_tuner, "generate_training_dataset")
        assert hasattr(slm_tuner, "TuningCard")


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    """Verify config files exist and parse correctly."""

    def test_env_example_exists(self):
        assert (ROOT / ".env.example").exists()

    def test_env_example_has_required_keys(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        for key in ["MONGO_URI", "QDRANT_HOST", "NEO4J_URI", "LLM_API_KEY"]:
            assert key in text, f"Missing {key} in .env.example"

    def test_run_card_yaml_loads(self):
        import yaml
        card_path = ROOT / "configs" / "run_card.yaml"
        assert card_path.exists(), "configs/run_card.yaml missing"
        with open(card_path) as f:
            data = yaml.safe_load(f)
        assert "winning_config" in data or "best" in data or isinstance(data, dict)

    def test_eval_queries_csv_exists(self):
        csv_path = ROOT / "eval_queries.csv"
        assert csv_path.exists()

    def test_eval_queries_csv_has_header(self):
        import csv
        csv_path = ROOT / "eval_queries.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
        assert "query" in fields, "eval_queries.csv must have a 'query' column"

    def test_requirements_txt_exists(self):
        assert (ROOT / "requirements.txt").exists()

    def test_docker_compose_exists(self):
        assert (ROOT / "docker-compose.yml").exists()


# ---------------------------------------------------------------------------
# Evaluation metric tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    """Unit tests for the evaluation scoring functions."""

    def test_faithfulness_all_verified(self):
        from evaluate import compute_faithfulness
        assert compute_faithfulness("answer", 5, 5) == 1.0

    def test_faithfulness_none_verified(self):
        from evaluate import compute_faithfulness
        assert compute_faithfulness("answer", 0, 3) == 0.0

    def test_faithfulness_no_citations(self):
        from evaluate import compute_faithfulness
        assert compute_faithfulness("answer", 0, 0) == 1.0

    def test_relevance_with_keywords(self):
        from evaluate import compute_answer_relevance
        score = compute_answer_relevance(
            "What is attention?",
            "The attention mechanism uses self-attention and multi-head projections.",
            ["attention", "self-attention", "multi-head"],
        )
        assert score > 0.5

    def test_relevance_no_keywords(self):
        from evaluate import compute_answer_relevance
        score = compute_answer_relevance(
            "transformer architecture",
            "The transformer uses attention mechanisms.",
        )
        assert 0.0 <= score <= 1.0

    def test_relevance_empty_answer(self):
        from evaluate import compute_answer_relevance
        score = compute_answer_relevance("test query", "", ["keyword"])
        assert score == 0.0


# ---------------------------------------------------------------------------
# Provenance filter tests
# ---------------------------------------------------------------------------

class TestProvenanceFilter:
    """Unit tests for the citation provenance safety filter."""

    def test_no_citations_returns_full_score(self):
        from graphrag_executor import provenance_filter
        from hybrid_search import SearchResult
        chunks = [SearchResult(
            chunk_id="c1", paper_id="p1", title="Test Paper",
            text="Some text", score=0.9, page_start=1, page_end=5,
            authors=["Author A"], source="dense",
        )]
        answer = "This is a plain answer without citations."
        filtered, verified, dropped, score = provenance_filter(answer, chunks)
        assert score == 1.0
        assert len(dropped) == 0

    def test_valid_citation_passes(self):
        from graphrag_executor import provenance_filter
        from hybrid_search import SearchResult
        chunks = [SearchResult(
            chunk_id="c1", paper_id="p1", title="Test Paper",
            text="Some text", score=0.9, page_start=1, page_end=5,
            authors=["Author A"], source="dense",
        )]
        answer = 'The model works well [Author A] "Test Paper" (pp. 1-5).'
        filtered, verified, dropped, score = provenance_filter(answer, chunks)
        assert score == 1.0
        assert len(verified) == 1

    def test_invalid_citation_dropped(self):
        from graphrag_executor import provenance_filter
        from hybrid_search import SearchResult
        chunks = [SearchResult(
            chunk_id="c1", paper_id="p1", title="Test Paper",
            text="Some text", score=0.9, page_start=1, page_end=5,
            authors=["Author A"], source="dense",
        )]
        answer = 'Fake claim [X] "Nonexistent Paper" (pp. 99-100).'
        filtered, verified, dropped, score = provenance_filter(answer, chunks)
        assert score == 0.0
        assert len(dropped) == 1
        assert "[CITATION REMOVED" in filtered


# ---------------------------------------------------------------------------
# Data pipeline tests (no DB required)
# ---------------------------------------------------------------------------

class TestDataPipeline:
    """Verify data structures and helpers work without databases."""

    def test_search_result_citation_format(self):
        from hybrid_search import SearchResult
        r = SearchResult(
            chunk_id="c1", paper_id="p1",
            title="Attention Is All You Need",
            text="test", score=0.95,
            page_start=3, page_end=7,
            authors=["Vaswani", "Shazeer"],
            source="bm25",
        )
        cite = r.citation()
        assert "Attention Is All You Need" in cite
        assert "3" in cite and "7" in cite

    def test_synthetic_data_generation(self):
        from seed_data import generate_synthetic_data
        papers = generate_synthetic_data()
        assert len(papers) == 5
        for p in papers:
            assert "paper_id" in p
            assert "title" in p
            assert "text" in p

    def test_curated_qa_pairs_exist(self):
        from slm_tuner import CURATED_QA_PAIRS
        assert len(CURATED_QA_PAIRS) >= 10
        for pair in CURATED_QA_PAIRS:
            assert "question" in pair
            assert "answer" in pair
            assert len(pair["answer"]) > 20


# ---------------------------------------------------------------------------
# SLM tuner dataset generation test
# ---------------------------------------------------------------------------

class TestSLMTuner:
    """Verify SLM tuner dataset generation (no GPU required)."""

    def test_generate_dataset(self, tmp_path):
        from slm_tuner import generate_training_dataset
        out = tmp_path / "test_qa.json"
        generate_training_dataset(out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) >= 10
        for item in data:
            assert "instruction" in item
            assert "output" in item

    def test_tuning_card_dataclass(self):
        from slm_tuner import TuningCard
        card = TuningCard(
            base_model="test-model", dataset_path="test.json",
            dataset_size=10, epochs=1, learning_rate=1e-4,
            batch_size=4, max_seq_len=512, lora_r=16,
            lora_alpha=32, lora_dropout=0.05,
            lora_target_modules=["q_proj"], quantization="4-bit",
            hardware="test", gpu_name="none",
            training_time_seconds=1.0, output_dir="test_out",
            license="MIT", timestamp="2024-01-01",
        )
        assert card.dataset_size == 10
        assert card.lora_r == 16


# ---------------------------------------------------------------------------
# Online learning tests (no DB required)
# ---------------------------------------------------------------------------

class TestOnlineLearning:
    """Verify River online learning adapter works standalone."""

    def test_adapter_init(self):
        from online_learning import RiverHybridAdapter
        adapter = RiverHybridAdapter()
        assert adapter.current_accuracy == 0.0
        assert adapter.n_drifts == 0

    def test_adapter_learn_and_predict(self):
        from online_learning import RiverHybridAdapter
        adapter = RiverHybridAdapter()
        entry = adapter.learn("test query", bm25_top_score=0.8, dense_top_score=0.6, helpful=1)
        assert entry.step >= 0
        alpha = adapter.predict_alpha("test query", bm25_top_score=0.8, dense_top_score=0.6)
        assert 0.0 <= alpha <= 1.0


# ---------------------------------------------------------------------------
# App model tests (Pydantic validation)
# ---------------------------------------------------------------------------

class TestAppModels:
    """Verify Pydantic request/response models validate correctly."""

    def test_search_request_valid(self):
        from app import SearchRequest
        req = SearchRequest(query="test query", top_k=5)
        assert req.query == "test query"

    def test_search_request_rejects_empty(self):
        from app import SearchRequest
        with pytest.raises(Exception):
            SearchRequest(query="", top_k=5)

    def test_feedback_request_valid(self):
        from app import FeedbackRequest
        req = FeedbackRequest(query="test", helpful=1, bm25_top_score=0.5, dense_top_score=0.3)
        assert req.helpful == 1

    def test_graphrag_request_valid(self):
        from app import GraphRAGRequest
        req = GraphRAGRequest(query="What is attention?", top_k=5)
        assert req.top_k == 5
