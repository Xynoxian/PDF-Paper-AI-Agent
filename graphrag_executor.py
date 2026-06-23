"""
GraphRAG Executor (D3 -- Retrieval-to-Generation Pipeline)
============================================================
Graph-filtered retrieval with LLM answer generation and provenance safety.

Architecture (Method A -- Pre-Filter):

    Query
      |
      +-> LLM: generate Cypher from user question
      |         |
      |         v
      |     Neo4j -> paper_ids (subgraph)
      |         |
      v         v
    HybridSearcher.search(paper_ids filter)
      |
      +-> BM25 (filtered)  --+
      +-> Dense (filtered) --+
                              v
                          RRF fusion -> top-K chunks
                              |
                              v
                    LLM: generate answer with page-range citations
                              |
                              v
                    Provenance filter: verify citations exist in metadata

If Cypher generation fails or returns no results, the pipeline
falls back to unfiltered hybrid search -- recall is preserved.

Usage:
    from graphrag_executor import GraphRAGExecutor

    executor = GraphRAGExecutor()
    response = executor.query("How does attention work in transformers?")
    print(response.answer)
    print(response.citations)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from graph_build import KnowledgeGraph
from hybrid_search import HybridSearcher, SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Optional local LLM via Ollama — no API key required (runs on your machine).
FALLBACK_LLM_ENABLED: bool = os.getenv("FALLBACK_LLM_ENABLED", "true").lower() in ("1", "true", "yes")
FALLBACK_LLM_BASE_URL: str = os.getenv("FALLBACK_LLM_BASE_URL", "http://localhost:11434/v1")
FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "llama3.2")
# Placeholder only — Ollama ignores this; kept internal so you never configure a key.
_LOCAL_LLM_DUMMY_KEY = "local-no-key-required"

# ---------------------------------------------------------------------------
# Graph schema (injected into the Cypher-generation prompt)
# ---------------------------------------------------------------------------

GRAPH_SCHEMA = """
Node labels and properties:
  - Paper: paper_id (string, unique), title (string), abstract (string), year (int), venue (string)
  - Author: name (string, unique)
  - Topic: name (string, unique)

Relationship types:
  - (Author)-[:WROTE]->(Paper)
  - (Paper)-[:HAS_TOPIC]->(Topic)
  - (Paper)-[:CITES]->(Paper)

Known topics stored in the database:
  Machine Learning, Natural Language Processing, Computer Vision,
  Reinforcement Learning, Information Retrieval, Knowledge Graphs,
  Generative AI, Optimization, Robotics, Ethics & Fairness
""".strip()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

CYPHER_SYSTEM_PROMPT = (
    "You are a Cypher query generator for a Neo4j academic-paper knowledge graph.\n\n"
    f"Schema:\n{GRAPH_SCHEMA}\n\n"
    "Task: given the user's question, produce a Cypher query that returns "
    "paper_id values of relevant papers.\n\n"
    "Rules:\n"
    "1. ONLY use node labels, relationship types, and properties from the schema above.\n"
    "2. The query MUST return a column named `paper_id`.\n"
    "3. Use toLower() and CONTAINS for flexible string matching.\n"
    "4. LIMIT results to 20.\n"
    "5. Prefer simple single-MATCH patterns.\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    '{"cypher": "MATCH ... RETURN ... LIMIT 20", "intent": "brief description"}\n\n'
    "If the question has no clear graph-structural angle "
    "(no author, topic, or citation reference), respond:\n"
    '{"cypher": null, "intent": "no_match"}'
)

ANSWER_SYSTEM_PROMPT = (
    "You are an academic research assistant. Answer the user's question "
    "using ONLY the provided text chunks from research papers.\n\n"
    "Strict rules:\n"
    "1. Use ONLY information from the chunks below. Never use external knowledge.\n"
    "2. After every claim, cite the source using EXACTLY this format: "
    '[Author1, Author2 et al.] "Paper Title" (pp. X-Y)\n'
    "3. The paper title and page numbers in your citation MUST match one of the "
    "provided chunks exactly. Do not fabricate titles or page numbers.\n"
    "4. If the provided chunks do not contain enough information, state that explicitly.\n"
    "5. Be concise and direct."
)

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class GraphRAGResponse:
    """Structured output from the GraphRAG pipeline."""

    answer: str
    citations: list[str]
    verified_citations: list[str]
    dropped_citations: list[str]
    chunks_used: int
    graph_filter_applied: bool
    graph_papers_found: int
    cypher_generated: str | None
    intent: str
    fallback: bool
    provenance_score: float
    bm25_top_score: float = 0.0
    dense_top_score: float = 0.0


# ---------------------------------------------------------------------------
# Provenance Filtering (Requirement 3 -- Safety)
# ---------------------------------------------------------------------------


def provenance_filter(
    answer: str,
    chunks: list[SearchResult],
) -> tuple[str, list[str], list[str], float]:
    """Verify that citations in the LLM answer exist in retrieved metadata.

    Checks each citation pattern [Author...] "Title" (pp. X-Y) against the
    actual titles and page ranges from the retrieved chunks.

    Args:
        answer: The raw LLM-generated answer with citations.
        chunks: The SearchResult objects that were provided as context.

    Returns:
        (filtered_answer, verified_citations, dropped_citations, score)
        where score = len(verified) / len(all_found) or 1.0 if no citations.
    """
    title_pages: dict[str, set[tuple[int, int]]] = {}
    for chunk in chunks:
        title_lower = chunk.title.strip().lower()
        if title_lower not in title_pages:
            title_pages[title_lower] = set()
        title_pages[title_lower].add((chunk.page_start, chunk.page_end))

    citation_pattern = re.compile(
        r'\[([^\]]*)\]\s*"([^"]+)"\s*\(pp\.\s*(\d+)\s*-\s*(\d+)\)'
    )

    found_citations = citation_pattern.findall(answer)

    if not found_citations:
        return answer, [], [], 1.0

    verified: list[str] = []
    dropped: list[str] = []

    for authors, title, page_start, page_end in found_citations:
        full_cite = f'[{authors}] "{title}" (pp. {page_start}-{page_end})'
        title_lower = title.strip().lower()
        ps, pe = int(page_start), int(page_end)

        is_valid = False
        if title_lower in title_pages:
            for chunk_ps, chunk_pe in title_pages[title_lower]:
                if ps >= chunk_ps and pe <= chunk_pe:
                    is_valid = True
                    break
                if ps == chunk_ps and pe == chunk_pe:
                    is_valid = True
                    break
                if abs(ps - chunk_ps) <= 1 and abs(pe - chunk_pe) <= 1:
                    is_valid = True
                    break

        if is_valid:
            verified.append(full_cite)
        else:
            dropped.append(full_cite)

    filtered_answer = answer
    for cite in dropped:
        filtered_answer = filtered_answer.replace(cite, "[CITATION REMOVED - unverified]")

    score = len(verified) / len(found_citations) if found_citations else 1.0

    if dropped:
        logger.warning(
            "Provenance filter dropped %d/%d citations.",
            len(dropped), len(found_citations),
        )

    return filtered_answer, verified, dropped, score


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class GraphRAGExecutor:
    """D3 retrieval-to-generation pipeline with graph pre-filtering and provenance safety.

    Pipeline steps:
      1. LLM generates a Cypher query from the user question.
      2. Cypher runs against Neo4j -> set of paper_ids.
      3. BM25 + Dense search run filtered to those paper_ids, then RRF-fused.
         If the filter yields nothing, falls back to unfiltered search.
      4. LLM generates a grounded answer with page-range citations.
      5. Provenance filter verifies all citations against retrieved metadata.
    """

    def __init__(
        self,
        llm_api_key: str = LLM_API_KEY,
        llm_base_url: str = LLM_BASE_URL,
        llm_model: str = LLM_MODEL,
        llm_temperature: float = LLM_TEMPERATURE,
        fallback_enabled: bool = FALLBACK_LLM_ENABLED,
        fallback_base_url: str = FALLBACK_LLM_BASE_URL,
        fallback_model: str = FALLBACK_LLM_MODEL,
        searcher: HybridSearcher | None = None,
        graph: KnowledgeGraph | None = None,
        use_tuned_slm: bool = False,
    ) -> None:
        """Initialize the GraphRAG executor.

        Args:
            llm_api_key: API key for the LLM provider.
            llm_base_url: Base URL for the LLM API (OpenAI-compatible).
            llm_model: Model identifier to use for generation.
            llm_temperature: Sampling temperature for LLM calls.
            fallback_enabled: Whether to try local Ollama when the primary LLM fails.
            fallback_base_url: Base URL for Ollama (default: http://localhost:11434/v1).
            fallback_model: Model name pulled in Ollama (e.g. llama3.2).
            searcher: Pre-configured HybridSearcher instance (or creates one).
            graph: Pre-configured KnowledgeGraph instance (or creates one).
            use_tuned_slm: If True, use the QLoRA-tuned SLM for answer generation.
        """
        self._use_tuned_slm = use_tuned_slm
        self._primary_llm = OpenAI(api_key=llm_api_key or "not-set", base_url=llm_base_url)
        self._model = llm_model
        self._temperature = llm_temperature
        self._fallback_enabled = fallback_enabled
        self._fallback_base_url = fallback_base_url
        self._fallback_llm: OpenAI | None = None
        self._fallback_model = fallback_model
        self._local_llm_available: bool | None = None
        if fallback_enabled:
            self._fallback_llm = OpenAI(
                api_key=_LOCAL_LLM_DUMMY_KEY,
                base_url=fallback_base_url,
            )
        self._last_llm_source = "primary"
        self._searcher = searcher or HybridSearcher()
        self._graph = graph or KnowledgeGraph()

    def _is_local_llm_running(self) -> bool:
        """Check whether Ollama (or similar) is reachable — no API key involved."""
        if self._local_llm_available is not None:
            return self._local_llm_available

        import requests

        root = self._fallback_base_url.rstrip("/").removesuffix("/v1")
        try:
            resp = requests.get(f"{root}/api/tags", timeout=1.5)
            self._local_llm_available = resp.status_code == 200
        except Exception:
            self._local_llm_available = False

        if not self._local_llm_available:
            logger.info("Local Ollama not detected at %s — offline mode will be used.", root)
        return self._local_llm_available

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        purpose: str,
    ) -> str:
        """Call the primary LLM, then optional local Ollama (no key), else raise."""
        try:
            response = self._primary_llm.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=messages,
            )
            self._last_llm_source = "primary"
            content = response.choices[0].message.content
            if content:
                return content.strip()
            raise ValueError("Primary LLM returned empty content")
        except Exception as primary_exc:
            logger.warning("Primary LLM failed (%s): %s", purpose, primary_exc)

        if (
            self._fallback_enabled
            and self._fallback_llm is not None
            and self._is_local_llm_running()
        ):
            try:
                response = self._fallback_llm.chat.completions.create(
                    model=self._fallback_model,
                    temperature=self._temperature,
                    messages=messages,
                )
                self._last_llm_source = "local"
                content = response.choices[0].message.content
                if content:
                    logger.info(
                        "Local Ollama (%s) succeeded for %s.",
                        self._fallback_model,
                        purpose,
                    )
                    return content.strip()
            except Exception as fallback_exc:
                logger.warning("Local Ollama failed (%s): %s", purpose, fallback_exc)

        self._last_llm_source = "offline"
        raise RuntimeError(f"All LLM providers failed for {purpose}")

    @staticmethod
    def _offline_answer(query: str, chunks: list[SearchResult]) -> str:
        """Key-free answer built from retrieved chunks — always available."""
        query_terms = {
            w.lower()
            for w in re.findall(r"\w+", query)
            if len(w) > 2
        }

        parts = [
            "Offline mode (no API key needed): answer assembled from your ingested papers.\n",
        ]

        for chunk in chunks[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text.strip())
            ranked: list[tuple[int, str]] = []
            for sentence in sentences:
                cleaned = sentence.strip()
                if len(cleaned) < 25:
                    continue
                words = {w.lower() for w in re.findall(r"\w+", cleaned)}
                overlap = len(query_terms & words) if query_terms else 0
                ranked.append((overlap, cleaned))
            ranked.sort(key=lambda item: item[0], reverse=True)

            picks = [s for score, s in ranked[:2] if score > 0]
            if not picks:
                excerpt = chunk.text.strip().replace("\n", " ")
                picks = [excerpt[:350] + ("..." if len(excerpt) > 350 else "")]

            parts.append(f"\n{chunk.citation()}")
            for pick in picks:
                parts.append(f"  • {pick}")

        parts.append(
            "\nTip: install Ollama for fuller offline AI answers (ollama pull llama3.2), "
            "or fix LLM_API_KEY for cloud Gemini."
        )
        return "\n".join(parts)

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._graph.close()

    # ---- Step 1: Cypher generation ----------------------------------------

    def _generate_cypher(self, query: str) -> tuple[str | None, str]:
        """Use the LLM to produce a Cypher query for the user's question.

        The LLM receives the full graph schema and is instructed to return
        a JSON object with 'cypher' and 'intent' fields. If the question
        has no graph-structural angle, cypher is null.

        Args:
            query: The user's natural-language question.

        Returns:
            Tuple of (cypher_string_or_None, intent_description).
        """
        try:
            raw = self._chat_completion(
                messages=[
                    {"role": "system", "content": CYPHER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                purpose="cypher_generation",
            )

            if raw.startswith("```"):
                raw = raw.strip("`").removeprefix("json").strip()

            parsed = json.loads(raw)
            cypher = parsed.get("cypher")
            intent = parsed.get("intent", "unknown")
            logger.info("Cypher generated | intent: %s", intent)
            if cypher:
                logger.info("Cypher: %s", cypher)
            return cypher, intent

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Cypher generation parse error: %s", exc)
            return None, "parse_error"
        except Exception as exc:
            logger.warning("Cypher generation failed: %s", exc)
            return None, "llm_error"

    # ---- Step 2: Graph query -> paper IDs ---------------------------------

    def _execute_cypher(self, cypher: str) -> list[str]:
        """Run a Cypher query against Neo4j and extract paper_id values.

        Args:
            cypher: A valid Cypher query string that returns paper_id.

        Returns:
            Deduplicated list of paper_id strings. Empty list on failure.
        """
        try:
            with self._graph._driver.session() as session:
                result = session.run(cypher)
                paper_ids: list[str] = []
                for record in result:
                    pid = record.get("paper_id")
                    if pid:
                        paper_ids.append(pid)
                return list(set(paper_ids))
        except Exception as exc:
            logger.warning("Cypher execution failed: %s", exc)
            return []

    # ---- Step 3: Filtered retrieval ---------------------------------------

    def _filtered_search(
        self,
        query: str,
        paper_ids: list[str] | None,
        top_k: int,
    ) -> tuple[list[SearchResult], bool]:
        """Run hybrid search, optionally filtered to specific paper IDs.

        If paper_ids are provided but the filtered search returns no results,
        falls back to unfiltered search to preserve recall.

        Args:
            query: The search query string.
            paper_ids: Paper IDs from graph traversal, or None.
            top_k: Number of top results to return.

        Returns:
            Tuple of (results, fallback_used).
        """
        if paper_ids:
            results = self._searcher.search(
                query, top_k=top_k, paper_ids=paper_ids,
            )
            if results:
                return results, False
            logger.info(
                "Filtered search returned 0 results -- falling back to unfiltered."
            )

        results = self._searcher.search(query, top_k=top_k)
        fallback = bool(paper_ids)
        return results, fallback

    # ---- Step 4: Answer generation ----------------------------------------

    @staticmethod
    def _build_context(chunks: list[SearchResult]) -> str:
        """Format retrieved chunks into a numbered context block for the LLM.

        Each chunk includes its citation metadata (authors, title, pages)
        so the LLM can produce accurate citations.

        Args:
            chunks: List of SearchResult objects from hybrid search.

        Returns:
            Formatted string with numbered chunks separated by dividers.
        """
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Chunk {i}] {chunk.citation()}\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate_answer(self, query: str, chunks: list[SearchResult]) -> str:
        """Ask the LLM (or tuned SLM) to answer using the retrieved chunks.

        When use_tuned_slm is enabled, routes to the QLoRA fine-tuned model
        with disk caching. Otherwise uses the cloud/Ollama LLM chain.

        Args:
            query: The user's original question.
            chunks: Retrieved text chunks with citation metadata.

        Returns:
            The generated answer string.
        """
        # Route to tuned SLM if enabled and available
        if self._use_tuned_slm:
            try:
                from slm_tuner import cached_generate
                context = self._build_context(chunks)
                augmented = f"{query}\n\nContext from papers:\n{context[:1500]}"
                answer, cache_hit = cached_generate(augmented)
                if cache_hit:
                    logger.info("SLM answer served from cache.")
                self._last_llm_source = "tuned_slm"
                return answer
            except (ImportError, FileNotFoundError) as exc:
                logger.warning("Tuned SLM unavailable, falling back to LLM: %s", exc)

        context = self._build_context(chunks)

        user_prompt = (
            f"Question: {query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer the question using ONLY the chunks above. "
            "Cite every claim with the exact format shown in the rules."
        )

        try:
            return self._chat_completion(
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                purpose="answer_generation",
            )
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            return self._offline_answer(query, chunks)

    @staticmethod
    def _extract_unique_citations(chunks: list[SearchResult]) -> list[str]:
        """Build a deduplicated citation list from retrieved chunks.

        Args:
            chunks: The SearchResult objects used for context.

        Returns:
            List of unique citation strings.
        """
        seen: set[str] = set()
        citations: list[str] = []
        for chunk in chunks:
            c = chunk.citation()
            if c not in seen:
                seen.add(c)
                citations.append(c)
        return citations

    # ---- Public API -------------------------------------------------------

    def query(self, user_query: str, top_k: int = 5) -> GraphRAGResponse:
        """Execute the full GraphRAG pipeline end-to-end.

        Steps:
          1. LLM generates Cypher from the user query.
          2. Cypher runs against Neo4j -> paper_ids.
          3. Filtered BM25 + Dense + RRF retrieval (fallback if empty).
          4. LLM generates a grounded answer with page-range citations.
          5. Provenance filter verifies citations against retrieved metadata.

        Args:
            user_query: The natural-language question from the user.
            top_k: Number of chunks to retrieve.

        Returns:
            GraphRAGResponse with answer, citations, and provenance metadata.
        """
        # Step 1: Generate Cypher
        cypher, intent = self._generate_cypher(user_query)

        # Step 2: Execute Cypher -> paper IDs
        paper_ids: list[str] = []
        if cypher:
            paper_ids = self._execute_cypher(cypher)
            logger.info("Graph returned %d paper(s).", len(paper_ids))

        # Step 3: Filtered hybrid search
        chunks, fallback = self._filtered_search(
            user_query,
            paper_ids if paper_ids else None,
            top_k,
        )

        # Extract top scores for feedback loop
        bm25_top = 0.0
        dense_top = 0.0
        for c in chunks:
            if c.source == "bm25" and c.score > bm25_top:
                bm25_top = c.score
            elif c.source == "dense" and c.score > dense_top:
                dense_top = c.score

        # No chunks at all
        if not chunks:
            return GraphRAGResponse(
                answer="No relevant documents found for your query.",
                citations=[],
                verified_citations=[],
                dropped_citations=[],
                chunks_used=0,
                graph_filter_applied=False,
                graph_papers_found=len(paper_ids),
                cypher_generated=cypher,
                intent=intent,
                fallback=fallback,
                provenance_score=1.0,
                bm25_top_score=bm25_top,
                dense_top_score=dense_top,
            )

        # Step 4: Generate answer
        raw_answer = self._generate_answer(user_query, chunks)

        # Step 5: Provenance filtering (safety)
        filtered_answer, verified, dropped, prov_score = provenance_filter(
            raw_answer, chunks,
        )

        citations = self._extract_unique_citations(chunks)

        return GraphRAGResponse(
            answer=filtered_answer,
            citations=citations,
            verified_citations=verified,
            dropped_citations=dropped,
            chunks_used=len(chunks),
            graph_filter_applied=bool(paper_ids) and not fallback,
            graph_papers_found=len(paper_ids),
            cypher_generated=cypher,
            intent=intent,
            fallback=fallback,
            provenance_score=round(prov_score, 4),
            bm25_top_score=round(bm25_top, 4),
            dense_top_score=round(dense_top, 4),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Quick smoke test from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG Executor")
    parser.add_argument("query", type=str, help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    executor = GraphRAGExecutor()
    resp = executor.query(args.query, top_k=args.top_k)

    print(f"\n{'='*72}")
    print(f"Intent           : {resp.intent}")
    print(f"Cypher           : {resp.cypher_generated}")
    print(f"Graph hits       : {resp.graph_papers_found}")
    print(f"Filter applied   : {resp.graph_filter_applied}")
    print(f"Fallback         : {resp.fallback}")
    print(f"Chunks used      : {resp.chunks_used}")
    print(f"Provenance score : {resp.provenance_score}")
    print(f"Verified cites   : {len(resp.verified_citations)}")
    print(f"Dropped cites    : {len(resp.dropped_citations)}")
    print(f"{'='*72}")
    print(f"\n{resp.answer}\n")

    if resp.dropped_citations:
        print("DROPPED citations (failed provenance):")
        for c in resp.dropped_citations:
            print(f"  X  {c}")

    print("\nVerified citations:")
    for c in resp.verified_citations:
        print(f"  +  {c}")

    executor.close()


if __name__ == "__main__":
    main()
