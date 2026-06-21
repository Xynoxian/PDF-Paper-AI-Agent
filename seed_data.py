"""
Seed Data Script
=================
Downloads sample arXiv PDFs, runs the full pipeline (ingest -> graph -> eval),
and prints a metrics summary.

Usage:
    python seed_data.py
    python seed_data.py --pdf-dir ./my_papers
    python seed_data.py --synthetic
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample arXiv papers
# ---------------------------------------------------------------------------

SAMPLE_PAPERS: list[dict[str, str]] = [
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
    },
    {
        "id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
    },
    {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/pdf/2005.11401.pdf",
    },
    {
        "id": "2302.13971",
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "url": "https://arxiv.org/pdf/2302.13971.pdf",
    },
    {
        "id": "2304.08485",
        "title": "Visual Instruction Tuning (LLaVA)",
        "url": "https://arxiv.org/pdf/2304.08485.pdf",
    },
]

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_pdfs(dest_dir: str | Path, papers: list[dict[str, str]] | None = None) -> Path:
    """Download sample PDFs to *dest_dir*."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    papers = papers or SAMPLE_PAPERS

    for paper in papers:
        filename = f"{paper['id'].replace('/', '_')}.pdf"
        filepath = dest / filename
        if filepath.exists():
            logger.info("Already downloaded: %s", filename)
            continue

        logger.info("Downloading '%s' -> %s ...", paper["title"], filename)
        try:
            resp = requests.get(paper["url"], timeout=60, stream=True)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("  Saved (%d KB)", filepath.stat().st_size // 1024)
        except Exception as exc:
            logger.warning("  Failed to download %s: %s", paper["id"], exc)

    return dest


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------


def generate_synthetic_data() -> list[dict]:
    """Generate synthetic paper data for testing when PDFs are unavailable."""
    papers = [
        {
            "paper_id": f"synthetic_{i:03d}",
            "title": title,
            "authors": authors,
            "text": abstract,
            "topics": topics,
        }
        for i, (title, authors, abstract, topics) in enumerate(
            [
                (
                    "Attention Is All You Need",
                    ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                    "The dominant sequence transduction models are based on complex recurrent or "
                    "convolutional neural networks. We propose a new simple network architecture, "
                    "the Transformer, based solely on attention mechanisms, dispensing with recurrence "
                    "and convolutions entirely. The Transformer uses multi-head self-attention to "
                    "compute representations of its input and output.",
                    ["Natural Language Processing", "Machine Learning"],
                ),
                (
                    "BERT: Pre-training of Deep Bidirectional Transformers",
                    ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
                    "We introduce a new language representation model called BERT, which stands for "
                    "Bidirectional Encoder Representations from Transformers. BERT is designed to "
                    "pre-train deep bidirectional representations from unlabeled text by jointly "
                    "conditioning on both left and right context.",
                    ["Natural Language Processing", "Machine Learning"],
                ),
                (
                    "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
                    ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
                    "Large pre-trained language models have shown to store factual knowledge in their "
                    "parameters. We explore a general-purpose fine-tuning recipe for retrieval-augmented "
                    "generation (RAG) -- models which combine pre-trained parametric and non-parametric "
                    "memory for language generation.",
                    ["Information Retrieval", "Natural Language Processing"],
                ),
                (
                    "ImageNet Classification with Deep Convolutional Neural Networks",
                    ["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey Hinton"],
                    "We trained a large, deep convolutional neural network to classify the 1.2 million "
                    "high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different "
                    "classes. On the test data, we achieved top-1 and top-5 error rates that are "
                    "considerably better than the previous state-of-the-art.",
                    ["Computer Vision", "Machine Learning"],
                ),
                (
                    "Playing Atari with Deep Reinforcement Learning",
                    ["Volodymyr Mnih", "Koray Kavukcuoglu", "David Silver"],
                    "We present the first deep learning model to successfully learn control policies "
                    "directly from high-dimensional sensory input using reinforcement learning. The "
                    "model is a convolutional neural network, trained with a variant of Q-learning.",
                    ["Reinforcement Learning", "Machine Learning"],
                ),
            ]
        )
    ]
    return papers


def seed_synthetic(
    mongo_uri: str | None = None,
    qdrant_host: str | None = None,
) -> None:
    """Seed the system with synthetic data (no PDFs needed)."""
    import uuid

    from pymongo import MongoClient
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from ingest import (
        EMBEDDING_DIM,
        MONGO_COLLECTION,
        MONGO_DB,
        MONGO_URI,
        QDRANT_COLLECTION,
        QDRANT_HOST,
        QDRANT_PORT,
        embed_texts,
    )

    mu = mongo_uri or MONGO_URI
    qh = qdrant_host or QDRANT_HOST

    papers = generate_synthetic_data()

    client = MongoClient(mu)
    col = client[MONGO_DB][MONGO_COLLECTION]

    qc = QdrantClient(host=qh, port=QDRANT_PORT)
    collections = [c.name for c in qc.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        qc.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    texts = [p["text"] for p in papers]
    embeddings = embed_texts(texts)

    for paper, emb in zip(papers, embeddings):
        chunk_id = str(uuid.uuid4())
        doc = {
            "chunk_id": chunk_id,
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "text": paper["text"],
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
        }
        col.insert_one(doc)

        qc.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=emb,
                    payload=doc,
                )
            ],
        )

    logger.info("Inserted %d synthetic chunks into Mongo + Qdrant.", len(papers))

    from graph_build import populate_from_mongodb
    populate_from_mongodb()
    logger.info("Graph populated from synthetic data.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full seed pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Seed Data")
    parser.add_argument("--pdf-dir", type=str, default="./papers", help="Directory for PDFs")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of downloading PDFs")
    parser.add_argument("--skip-download", action="store_true", help="Skip PDF download, use existing files")
    args = parser.parse_args()

    t0 = time.time()

    if args.synthetic:
        print("Seeding with synthetic data ...")
        seed_synthetic()
    else:
        if not args.skip_download:
            print("Downloading sample PDFs ...")
            download_pdfs(args.pdf_dir)

        pdf_dir = Path(args.pdf_dir)
        if not list(pdf_dir.glob("*.pdf")):
            print("No PDFs found -- falling back to synthetic data.")
            seed_synthetic()
        else:
            print("\nRunning ingestion pipeline ...")
            from ingest import ingest_directory
            chunks = ingest_directory(pdf_dir)
            print(f"  -> {len(chunks)} chunks ingested")

            print("\nBuilding knowledge graph ...")
            from graph_build import populate_from_mongodb
            populate_from_mongodb()

    print("\nRunning quick evaluation ...")
    from hybrid_search import HybridSearcher, run_quick_eval

    searcher = HybridSearcher()

    if args.synthetic:
        eval_queries = [
            {"query": "transformer attention mechanism", "relevant_ids": {"synthetic_000", "synthetic_001"}},
            {"query": "retrieval augmented generation", "relevant_ids": {"synthetic_002"}},
            {"query": "reinforcement learning Atari", "relevant_ids": {"synthetic_004"}},
            {"query": "image classification convolutional", "relevant_ids": {"synthetic_003"}},
        ]
    else:
        eval_queries = [
            {"query": "transformer attention mechanism", "relevant_ids": {"1706.03762", "1810.04805"}},
            {"query": "retrieval augmented generation", "relevant_ids": {"2005.11401"}},
            {"query": "reinforcement learning Atari", "relevant_ids": {"2302.13971"}},
            {"query": "image classification convolutional", "relevant_ids": {"2304.08485"}},
        ]

    run_quick_eval(searcher, eval_queries)

    elapsed = time.time() - t0
    print(f"\nSeed pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
