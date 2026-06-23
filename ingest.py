"""
PDF Ingestion Pipeline
=======================
Reads PDFs -> chunks text -> stores metadata in MongoDB -> embeds with
BAAI/bge-small-en-v1.5 -> indexes vectors in Qdrant.

Usage:
    python ingest.py --pdf-dir ./papers
    python ingest.py --pdf-dir ./papers --batch-size 32
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, cast

import fitz  # PyMuPDF
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB: str = os.getenv("MONGO_DB", "pdf_rag")
MONGO_COLLECTION: str = "chunks"

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = "paper_chunks"

EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM: int = 384

CHUNK_SIZE: int = 500          # characters
CHUNK_OVERLAP: int = 100       # characters

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChunkDocument:
    """A single text chunk with its metadata and optional embedding."""

    chunk_id: str
    paper_id: str
    title: str
    authors: list[str]
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def to_mongo_dict(self) -> dict:
        """Return a dict suitable for MongoDB insertion (no embedding)."""
        d = asdict(self)
        d.pop("embedding", None)
        return d


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def extract_text_with_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Extract text from every page of a PDF.

    Returns:
        List of (page_number, page_text) tuples.  Page numbers are 1-indexed.
    """
    pages: list[tuple[int, str]] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(1, len(doc) + 1):
            page = doc[page_num - 1]
            text = cast(str, page.get_text("text"))
            if text.strip():
                pages.append((page_num, text))
    return pages


def _guess_title(first_page_text: str) -> str:
    """Heuristic: the first non-empty line is usually the title."""
    for line in first_page_text.splitlines():
        stripped = line.strip()
        if len(stripped) > 5:
            return stripped
    return "Untitled"


def _guess_authors(first_page_text: str) -> list[str]:
    """Heuristic: look for lines near the top that resemble author names."""
    lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]
    authors: list[str] = []
    for line in lines[1:6]:
        if len(line) > 120 or line.lower().startswith("abstract"):
            break
        if 3 < len(line) < 100 and not line.endswith("."):
            parts = re.split(r",|(?:\band\b)", line)
            for part in parts:
                name = part.strip()
                if 2 < len(name) < 60:
                    authors.append(name)
    return authors if authors else ["Unknown"]


def _paper_id_from_path(pdf_path: Path) -> str:
    """Use the filename stem directly as the paper_id (e.g. '1706.03762')."""
    return pdf_path.stem


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text_with_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split page-level text into overlapping character-level chunks.

    Each returned dict has keys:
        text, page_start, page_end, chunk_index
    """
    char_page: list[tuple[str, int]] = []
    for page_num, text in pages:
        for ch in text:
            char_page.append((ch, page_num))

    full_text = "".join(ch for ch, _ in char_page)
    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end]

        page_start = char_page[start][1]
        page_end = char_page[end - 1][1]

        chunks.append(
            {
                "text": chunk_text,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": idx,
            }
        )
        start += chunk_size - overlap
        idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load (and cache) the SentenceTransformer model."""
    if model_name not in _model_cache:
        logger.info("Loading embedding model '%s' ...", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts and return vectors as lists of floats."""
    model = get_embedding_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return [vec.tolist() for vec in embeddings]


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _get_mongo_collection():
    """Return the MongoDB collection handle."""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    collection.create_index("paper_id")
    collection.create_index("chunk_id", unique=True)
    return collection


def _get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client and ensure the collection exists."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", QDRANT_COLLECTION)
    return client


def _delete_paper_chunks(
    paper_id: str,
    mongo_col,
    qdrant_client: QdrantClient,
) -> None:
    """Remove all stored chunks for a paper (used before re-ingestion)."""
    mongo_col.delete_many({"paper_id": paper_id})
    try:
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
        )
    except Exception as exc:
        logger.warning("Could not delete Qdrant vectors for '%s': %s", paper_id, exc)


def count_papers(mongo_col=None) -> int:
    """Return the number of distinct ingested papers in MongoDB."""
    if mongo_col is None:
        mongo_col = _get_mongo_collection()
    return len(mongo_col.distinct("paper_id"))


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def ingest_pdf(
    pdf_path: str | Path,
    mongo_col=None,
    qdrant_client: QdrantClient | None = None,
    batch_size: int = 32,
) -> list[ChunkDocument]:
    """Ingest a single PDF end-to-end."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Ingesting '%s' ...", pdf_path.name)

    pages = extract_text_with_pages(pdf_path)
    if not pages:
        logger.warning("No text extracted from '%s' -- skipping.", pdf_path.name)
        return []

    first_page_text = pages[0][1]
    title = _guess_title(first_page_text)
    authors = _guess_authors(first_page_text)
    paper_id = _paper_id_from_path(pdf_path)

    if mongo_col is None:
        mongo_col = _get_mongo_collection()
    if qdrant_client is None:
        qdrant_client = _get_qdrant_client()

    if mongo_col.count_documents({"paper_id": paper_id}, limit=1):
        logger.info("  -> Replacing existing chunks for paper '%s'", paper_id)
        _delete_paper_chunks(paper_id, mongo_col, qdrant_client)

    raw_chunks = chunk_text_with_pages(pages)
    logger.info("  -> %d chunks", len(raw_chunks))

    chunks: list[ChunkDocument] = []
    for rc in raw_chunks:
        chunk = ChunkDocument(
            chunk_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{paper_id}:{rc['chunk_index']}")),
            paper_id=paper_id,
            title=title,
            authors=authors,
            text=rc["text"],
            page_start=rc["page_start"],
            page_end=rc["page_end"],
            chunk_index=rc["chunk_index"],
        )
        chunks.append(chunk)

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, batch_size=batch_size)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    if mongo_col is None:
        mongo_col = _get_mongo_collection()

    if qdrant_client is None:
        qdrant_client = _get_qdrant_client()

    mongo_docs = [c.to_mongo_dict() for c in chunks]
    if mongo_docs:
        mongo_col.insert_many(mongo_docs)
        logger.info("  -> Stored %d chunks in MongoDB", len(mongo_docs))

    points = [
        PointStruct(
            id=c.chunk_id,
            vector=c.embedding,
            payload={
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "title": c.title,
                "authors": c.authors,
                "text": c.text,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_index": c.chunk_index,
            },
        )
        for c in chunks
    ]
    qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logger.info("  -> Indexed %d vectors in Qdrant", len(points))

    return chunks


def ingest_directory(
    pdf_dir: str | Path,
    batch_size: int = 32,
) -> list[ChunkDocument]:
    """Ingest every PDF in *pdf_dir*."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in '%s'", pdf_dir)
        return []

    logger.info("Found %d PDFs in '%s'", len(pdfs), pdf_dir)

    mongo_col = _get_mongo_collection()
    qdrant_client = _get_qdrant_client()

    existing_ids = set(mongo_col.distinct("paper_id"))

    all_chunks: list[ChunkDocument] = []
    for pdf_path in pdfs:
        paper_id = _paper_id_from_path(pdf_path)
        if paper_id in existing_ids:
            logger.info("Re-ingesting '%s' (already in database).", pdf_path.name)
        try:
            chunks = ingest_pdf(pdf_path, mongo_col, qdrant_client, batch_size)
            all_chunks.extend(chunks)
            existing_ids.add(paper_id)
        except Exception as exc:
            logger.error("Failed to ingest '%s': %s", pdf_path.name, exc)

    logger.info("Ingestion complete -- %d total chunks from %d PDFs", len(all_chunks), len(pdfs))
    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline")
    parser.add_argument("--pdf-dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    chunks = ingest_directory(args.pdf_dir, batch_size=args.batch_size)
    print(f"\nIngested {len(chunks)} chunks total.")


if __name__ == "__main__":
    main()
