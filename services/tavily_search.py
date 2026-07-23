"""
services/tavily_search.py
Tavily web search client for the Research Agent.

Supports both single-query and multi-query (planned) search modes.
Converts Tavily API results into ScoredChunk objects so the Pipeline and
Generator can handle web evidence with the same interface as local chunks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from data.chunker import Chunk
from config import settings
from services.retriever import ScoredChunk
from utils.log import get_logger
from utils.events import research_log

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tavily_search(
    query: str,
    max_results: int | None = None,
    search_depth: str | None = None,
    search_topic: str | None = None,
    citation_offset: int = 0,
    query_index: int = 0,
) -> list[ScoredChunk]:
    """
    Search the web via Tavily and return results as ScoredChunk objects.

    Args:
        query:           The search query.
        max_results:     Override TAVILY_MAX_RESULTS if provided.
        search_depth:    "basic" or "advanced". Defaults to TAVILY_SEARCH_DEPTH.
        search_topic:    "general" or "news". Defaults to TAVILY_SEARCH_TOPIC.
        citation_offset: Starting citation_id to avoid ID collisions.
        query_index:     Index of this query in a multi-query plan (for unique chunk IDs).

    Returns:
        List of ScoredChunk objects with source = "[WEB] title — url".
    """
    _require_api_key()

    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "tavily-python is not installed. "
            "Run: pip install tavily-python"
        )

    max_results  = max_results  or settings.TAVILY_MAX_RESULTS
    search_depth = search_depth or settings.TAVILY_SEARCH_DEPTH
    search_topic = search_topic or settings.TAVILY_SEARCH_TOPIC

    log.info(
        f"[WEB] Tavily search: \"{query[:60]}{'...' if len(query) > 60 else ''}\" "
        f"(depth={search_depth}, topic={search_topic}, max={max_results})"
    )

    import time as _time
    _t0 = _time.time()
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=search_topic,
    )
    _elapsed = round((_time.time() - _t0) * 1000, 1)

    results: list[ScoredChunk] = []
    source_urls: list[str] = []
    for i, item in enumerate(response.get("results", [])):
        title   = item.get("title", "Untitled")
        url     = item.get("url", "")
        content = item.get("content", "").strip()
        score   = float(item.get("score", 0.5))

        if not content:
            continue

        source_urls.append(f"{title} — {url}")
        chunk = Chunk(
            chunk_id    = f"web::q{query_index}_r{i}",
            content     = content,
            source      = f"[WEB] {title} — {url}",
            chunk_index = i,
        )
        results.append(ScoredChunk(
            chunk       = chunk,
            score       = round(score, 4),
            citation_id = citation_offset + i + 1,
        ))

    research_log.emit(
        "search",
        f"Fetched {len(results)} page(s) for \"{query[:50]}{'...' if len(query) > 50 else ''}\"",
        duration_ms=_elapsed,
        details={"query": query, "results": len(results), "sources": source_urls},
    )

    log.info(f"[WEB] Tavily returned {len(results)} result(s)")
    return results


def tavily_search_multi(
    planned_queries: list,
    citation_offset: int = 0,
) -> list[ScoredChunk]:
    """
    Execute multiple planned searches concurrently and merge results.

    Args:
        planned_queries: List of PlannedQuery objects (each with .query and .purpose).
        citation_offset: Starting citation_id for the first result.

    Returns:
        Flat list of ScoredChunk objects from all queries (unsorted, not deduped —
        the pipeline handles global ranking and deduplication).
    """
    _require_api_key()

    all_results: list[ScoredChunk] = []
    result_count = 0  # Running offset for citation IDs

    def _search_one(idx: int, pq) -> list[ScoredChunk]:
        log.info(f"[WEB] Query {idx + 1}/{len(planned_queries)}: \"{pq.query}\" — {pq.purpose}")
        return tavily_search(
            query=pq.query,
            citation_offset=citation_offset + (idx * settings.TAVILY_MAX_RESULTS),
            query_index=idx,
        )

    # Run searches concurrently
    with ThreadPoolExecutor(max_workers=min(len(planned_queries), 5)) as pool:
        futures = {
            pool.submit(_search_one, i, pq): i
            for i, pq in enumerate(planned_queries)
        }
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                idx = futures[future]
                log.warning(f"[WEB] Query {idx + 1} failed: {e}")
                research_log.emit(
                    "search", f"Query {idx + 1} failed: {e}",
                    level="error",
                )

    log.info(f"[WEB] Multi-search: {len(planned_queries)} queries → {len(all_results)} total results")
    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_api_key() -> None:
    """Raise a clear error if the Tavily API key is not configured."""
    if not settings.TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY is not set. "
            "Add it to your .env file to use web or hybrid retrieval modes. "
            "Get a key at https://tavily.com"
        )
