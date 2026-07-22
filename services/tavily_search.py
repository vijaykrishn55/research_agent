"""
services/tavily_search.py
Tavily web search client for the Research Agent.

Converts Tavily API results into ScoredChunk objects so the Pipeline and
Generator can handle web evidence with the same interface as local chunks.

Optionally rewrites the user query into an optimised search string before
calling the API (controlled by TAVILY_OPTIMIZE_QUERY setting).
"""

from __future__ import annotations

from data.chunker import Chunk
from config import settings
from services.retriever import ScoredChunk
from utils.log import get_logger

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
) -> list[ScoredChunk]:
    """
    Search the web via Tavily and return results as ScoredChunk objects.

    Args:
        query:           The search query (already optimised if applicable).
        max_results:     Override TAVILY_MAX_RESULTS if provided.
        search_depth:    "basic" or "advanced". Defaults to TAVILY_SEARCH_DEPTH.
        search_topic:    "general" or "news". Defaults to TAVILY_SEARCH_TOPIC.
        citation_offset: Starting citation_id; use len(local_chunks) in hybrid
                         mode to avoid ID collisions with local chunks.

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

    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=search_topic,
    )

    results: list[ScoredChunk] = []
    for i, item in enumerate(response.get("results", [])):
        title   = item.get("title", "Untitled")
        url     = item.get("url", "")
        content = item.get("content", "").strip()
        score   = float(item.get("score", 0.5))

        if not content:
            continue

        chunk = Chunk(
            chunk_id    = f"web::result_{i}",
            content     = content,
            source      = f"[WEB] {title} — {url}",
            chunk_index = i,
        )
        results.append(ScoredChunk(
            chunk       = chunk,
            score       = round(score, 4),
            citation_id = citation_offset + i + 1,
        ))

    log.info(f"[WEB] Tavily returned {len(results)} result(s)")
    return results


def optimize_query(question: str, provider) -> str:
    """
    Rewrite a user question into a concise web search query.

    Only called when TAVILY_OPTIMIZE_QUERY is True. Uses a single lightweight
    LLM call. Falls back to the original question on any error.

    Args:
        question: The raw user question.
        provider: An LLMProvider instance to call.

    Returns:
        A short optimised search query string.
    """
    prompt = (
        f"Convert the following research question into an effective web search query. "
        f"Output only the search query — no explanation, no punctuation at the end, "
        f"maximum 12 words.\n\nQuestion: {question}"
    )
    try:
        response = provider.generate(
            prompt=prompt,
            system_prompt="You are a search query optimizer.",
            max_tokens=40,
            temperature=0.0,
        )
        optimised = response.text.strip().strip('"').strip("'")
        log.info(f"[OPT] Query optimised: \"{optimised}\"")
        return optimised if optimised else question
    except Exception as e:
        log.warning(f"[OPT] Query optimisation failed ({e}), using original")
        return question


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
