"""
services/retriever.py
Semantic search over the indexed document chunks.

Takes a natural-language query, embeds it, and returns the
most relevant chunks from the FAISS index.
"""

from dataclasses import dataclass

from data.chunker import Chunk
from data.index_store import IndexStore
from providers.embedder import Embedder
from utils.log import get_logger

log = get_logger(__name__)


@dataclass
class ScoredChunk:
    """A chunk with its retrieval similarity score."""
    chunk: Chunk
    score: float
    citation_id: int = 0   # Assigned during generation


class Retriever:
    """Embeds a query and retrieves the most relevant chunks."""

    def __init__(self, embedder: Embedder, index_store: IndexStore):
        self.embedder = embedder
        self.index_store = index_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Returns an empty list if no documents have been ingested.
        """
        if self.index_store.total_vectors == 0:
            log.warning("[WARN]  No documents indexed — cannot retrieve")
            return []

        query_embedding = self.embedder.embed_query(query)
        raw_results = self.index_store.search(query_embedding, top_k)

        results = []
        for i, (chunk, score) in enumerate(raw_results):
            results.append(ScoredChunk(
                chunk=chunk,
                score=round(score, 4),
                citation_id=i + 1,
            ))

        log.info(
            f"[?] Retrieved {len(results)} chunks for: "
            f"\"{query[:60]}{'...' if len(query) > 60 else ''}\""
        )

        return results
