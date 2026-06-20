"""
Neo4j Knowledge Graph Builder
===============================
Creates a knowledge graph with Paper, Author, and Topic nodes
plus WROTE, HAS_TOPIC, and CITES relationships.

Includes five reusable Cypher-query functions and a populate_from_mongodb
helper that reads ingested data from MongoDB and materialises the graph.

Usage:
    python graph_build.py                 # populate from MongoDB
    python graph_build.py --clear         # wipe graph first
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any

from neo4j import GraphDatabase
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB: str = os.getenv("MONGO_DB", "pdf_rag")
MONGO_COLLECTION: str = "chunks"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic extraction helpers
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Machine Learning": ["machine learning", "deep learning", "neural network", "gradient descent"],
    "Natural Language Processing": ["nlp", "natural language", "language model", "transformer", "bert", "gpt", "tokeniz"],
    "Computer Vision": ["computer vision", "image classification", "object detection", "convolutional", "cnn"],
    "Reinforcement Learning": ["reinforcement learning", "reward", "policy gradient", "q-learning", "mdp"],
    "Information Retrieval": ["information retrieval", "search engine", "ranking", "bm25", "retrieval"],
    "Knowledge Graphs": ["knowledge graph", "ontology", "triple", "entity linking"],
    "Generative AI": ["generative", "diffusion", "gan", "variational autoencoder", "vae"],
    "Optimization": ["optimization", "adam", "sgd", "learning rate", "convergence"],
    "Robotics": ["robot", "robotic", "manipulation", "locomotion"],
    "Ethics & Fairness": ["bias", "fairness", "ethics", "responsible ai"],
}


def extract_topics(text: str, top_n: int = 3) -> list[str]:
    """Extract up to *top_n* topics from *text* using keyword matching."""
    text_lower = text.lower()
    scores: Counter[str] = Counter()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[topic] += 1
    if not scores:
        return ["General"]
    return [t for t, _ in scores.most_common(top_n)]


# ---------------------------------------------------------------------------
# Graph manager
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Wrapper around a Neo4j driver with helpers for the project schema."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connected to Neo4j at %s", uri)

    def close(self) -> None:
        self._driver.close()

    # ---- schema -----------------------------------------------------------

    def create_constraints(self) -> None:
        """Create uniqueness constraints and indexes."""
        queries = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
        ]
        with self._driver.session() as session:
            for q in queries:
                session.run(q)
        logger.info("Constraints and indexes created.")

    def clear_graph(self) -> None:
        """Delete all nodes and relationships."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared.")

    # ---- node creation ----------------------------------------------------

    def merge_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str = "",
        year: int | None = None,
        venue: str = "",
    ) -> None:
        query = """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title    = $title,
            p.abstract = $abstract,
            p.year     = $year,
            p.venue    = $venue
        """
        with self._driver.session() as session:
            session.run(query, paper_id=paper_id, title=title, abstract=abstract, year=year, venue=venue)

    def merge_author(self, name: str) -> None:
        query = "MERGE (a:Author {name: $name})"
        with self._driver.session() as session:
            session.run(query, name=name)

    def merge_topic(self, name: str) -> None:
        query = "MERGE (t:Topic {name: $name})"
        with self._driver.session() as session:
            session.run(query, name=name)

    # ---- relationship creation --------------------------------------------

    def add_wrote(self, author_name: str, paper_id: str) -> None:
        query = """
        MATCH (a:Author {name: $author_name})
        MATCH (p:Paper  {paper_id: $paper_id})
        MERGE (a)-[:WROTE]->(p)
        """
        with self._driver.session() as session:
            session.run(query, author_name=author_name, paper_id=paper_id)

    def add_has_topic(self, paper_id: str, topic_name: str) -> None:
        query = """
        MATCH (p:Paper {paper_id: $paper_id})
        MATCH (t:Topic {name: $topic_name})
        MERGE (p)-[:HAS_TOPIC]->(t)
        """
        with self._driver.session() as session:
            session.run(query, paper_id=paper_id, topic_name=topic_name)

    def add_cites(self, citing_id: str, cited_id: str) -> None:
        query = """
        MATCH (a:Paper {paper_id: $citing_id})
        MATCH (b:Paper {paper_id: $cited_id})
        MERGE (a)-[:CITES]->(b)
        """
        with self._driver.session() as session:
            session.run(query, citing_id=citing_id, cited_id=cited_id)

    # ---- five example Cypher queries --------------------------------------

    def find_papers_by_author(self, name: str) -> list[dict[str, Any]]:
        """Query 1: Find all papers written by an author."""
        query = """
        MATCH (a:Author {name: $name})-[:WROTE]->(p:Paper)
        RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year
        ORDER BY p.year DESC
        """
        with self._driver.session() as session:
            result = session.run(query, name=name)
            return [dict(record) for record in result]

    def find_papers_by_topic(self, topic: str) -> list[dict[str, Any]]:
        """Query 2: Find papers about a specific topic."""
        query = """
        MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic {name: $topic})
        RETURN p.paper_id AS paper_id, p.title AS title
        ORDER BY p.title
        """
        with self._driver.session() as session:
            result = session.run(query, topic=topic)
            return [dict(record) for record in result]

    def find_collaborators(self, author_name: str) -> list[dict[str, Any]]:
        """Query 3: Find all co-authors of a given author."""
        query = """
        MATCH (a:Author {name: $name})-[:WROTE]->(p:Paper)<-[:WROTE]-(coauthor:Author)
        WHERE coauthor.name <> $name
        RETURN DISTINCT coauthor.name AS collaborator,
               count(p) AS shared_papers
        ORDER BY shared_papers DESC
        """
        with self._driver.session() as session:
            result = session.run(query, name=author_name)
            return [dict(record) for record in result]

    def count_papers_per_topic(self) -> list[dict[str, Any]]:
        """Query 4: Topic distribution (number of papers per topic)."""
        query = """
        MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
        RETURN t.name AS topic, count(p) AS paper_count
        ORDER BY paper_count DESC
        """
        with self._driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def find_related_papers(self, paper_id: str) -> list[dict[str, Any]]:
        """Query 5: Papers that share topics with a given paper."""
        query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(related:Paper)
        WHERE related.paper_id <> $paper_id
        RETURN DISTINCT related.paper_id AS paper_id,
               related.title AS title,
               collect(DISTINCT t.name) AS shared_topics
        ORDER BY size(shared_topics) DESC
        LIMIT 10
        """
        with self._driver.session() as session:
            result = session.run(query, paper_id=paper_id)
            return [dict(record) for record in result]

    # ---- graph stats ------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return counts of nodes and relationships."""
        queries = {
            "papers": "MATCH (p:Paper) RETURN count(p) AS n",
            "authors": "MATCH (a:Author) RETURN count(a) AS n",
            "topics": "MATCH (t:Topic) RETURN count(t) AS n",
            "wrote_edges": "MATCH ()-[r:WROTE]->() RETURN count(r) AS n",
            "topic_edges": "MATCH ()-[r:HAS_TOPIC]->() RETURN count(r) AS n",
            "cites_edges": "MATCH ()-[r:CITES]->() RETURN count(r) AS n",
        }
        result: dict[str, int] = {}
        with self._driver.session() as session:
            for key, q in queries.items():
                record = session.run(q).single()
                result[key] = record["n"] if record else 0
        return result


# ---------------------------------------------------------------------------
# MongoDB -> Neo4j sync
# ---------------------------------------------------------------------------


def populate_from_mongodb(
    graph: KnowledgeGraph | None = None,
    mongo_uri: str = MONGO_URI,
    mongo_db: str = MONGO_DB,
    mongo_collection: str = MONGO_COLLECTION,
) -> None:
    """Read papers/chunks from MongoDB and populate the Neo4j graph."""
    own_graph = graph is None
    if own_graph:
        graph = KnowledgeGraph()

    graph.create_constraints()

    client = MongoClient(mongo_uri)
    col = client[mongo_db][mongo_collection]

    pipeline = [
        {"$sort": {"chunk_index": 1}},
        {
            "$group": {
                "_id": "$paper_id",
                "title": {"$first": "$title"},
                "authors": {"$first": "$authors"},
                "first_text": {"$first": "$text"},
            }
        },
    ]

    papers = list(col.aggregate(pipeline))
    logger.info("Found %d unique papers in MongoDB.", len(papers))

    for paper in papers:
        paper_id = paper["_id"]
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", [])
        first_text = paper.get("first_text", "")

        graph.merge_paper(paper_id=paper_id, title=title, abstract=first_text[:500])

        for author in authors:
            graph.merge_author(author)
            graph.add_wrote(author, paper_id)

        topics = extract_topics(first_text)
        for topic in topics:
            graph.merge_topic(topic)
            graph.add_has_topic(paper_id, topic)

    stats = graph.stats()
    logger.info("Graph populated -- %s", stats)

    if own_graph:
        graph.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j Graph Builder")
    parser.add_argument("--clear", action="store_true", help="Clear the graph before populating")
    args = parser.parse_args()

    graph = KnowledgeGraph()

    if args.clear:
        graph.clear_graph()

    populate_from_mongodb(graph)

    print("\nGraph Stats:")
    for k, v in graph.stats().items():
        print(f"  {k}: {v}")

    print("\nTopic Distribution:")
    for row in graph.count_papers_per_topic():
        print(f"  {row['topic']}: {row['paper_count']} papers")

    graph.close()


if __name__ == "__main__":
    main()
