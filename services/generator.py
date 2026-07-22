"""
services/generator.py
Grounded answer generation with citation tracking.

Builds a minimal prompt from retrieved chunks, calls the LLM,
and parses citations from the response.
"""

import re
from dataclasses import dataclass, field

from providers.base import LLMProvider, ProviderResponse
from services.retriever import ScoredChunk
from utils.log import get_logger

log = get_logger(__name__)


# ── System prompt — kept minimal to save tokens ──────────────────────────────

SYSTEM_PROMPT = """You are a research assistant that answers questions strictly from provided evidence.

Rules:
1. Read ALL numbered evidence chunks before answering.
2. Cite every claim with [N] where N is the chunk number (e.g. [1], [2][3]).
3. If multiple chunks support a claim, cite all of them: [1][2].
4. NEVER say information is missing if it appears anywhere in the provided chunks.
5. NEVER fabricate information or citations not grounded in the chunks.
6. Only state "The available evidence does not address this question" when truly NO chunk is relevant.
7. Be concise and direct."""


@dataclass
class Citation:
    """A single citation linking a claim to a source chunk."""
    citation_id: int
    source: str
    chunk_preview: str    # First 150 chars of the chunk


@dataclass
class ResearchAnswer:
    """The complete research response with citations and metrics."""
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks_retrieved: int = 0
    chunks_cited: int = 0
    confidence: str = "low"    # low, medium, high
    metrics: dict = field(default_factory=dict)


class Generator:
    """Generates grounded answers from retrieved chunks using an LLM."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, question: str, scored_chunks: list[ScoredChunk]) -> ResearchAnswer:
        """
        Generate a grounded answer from retrieved chunks.

        If no chunks are provided, returns an honest "insufficient evidence" answer
        without calling the LLM (saves tokens).
        """
        if not scored_chunks:
            return ResearchAnswer(
                question=question,
                answer="No relevant documents were found. Please ingest documents first, "
                       "or rephrase your question.",
                confidence="low",
            )

        # Build the evidence block
        prompt = self._build_prompt(question, scored_chunks)

        # Call the LLM
        response = self.provider.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # Parse citations from the response
        cited_ids = self._extract_citation_ids(response.text)
        citations = self._build_citations(cited_ids, scored_chunks)
        confidence = self._assess_confidence(scored_chunks, cited_ids)

        answer = ResearchAnswer(
            question=question,
            answer=response.text,
            citations=citations,
            chunks_retrieved=len(scored_chunks),
            chunks_cited=len(citations),
            confidence=confidence,
            metrics={
                "tokens_used": response.tokens_used,
                "latency_ms": response.latency_ms,
                "model": response.model,
                "provider": response.provider,
            },
        )

        log.info(
            f"[LOG] Generated answer: {len(citations)} citations, "
            f"confidence={confidence}, {response.tokens_used} tokens"
        )

        return answer

    def _build_prompt(self, question: str, chunks: list[ScoredChunk]) -> str:
        """Build the user prompt with numbered evidence chunks."""
        n = len(chunks)
        header = f"You have {n} evidence chunk(s) below. Use ALL of them when answering.\n"

        evidence_lines = []
        for sc in chunks:
            evidence_lines.append(
                f"--- Chunk [{sc.citation_id}] | Source: {sc.chunk.source} ---\n"
                f"{sc.chunk.content}"
            )

        evidence_block = "\n\n".join(evidence_lines)
        return f"{header}\n{evidence_block}\n\n--- Question ---\n{question}"

    def _extract_citation_ids(self, text: str) -> set[int]:
        """Extract all [N] citation references from the LLM output."""
        matches = re.findall(r"\[(\d+)\]", text)
        return {int(m) for m in matches}

    def _build_citations(
        self, cited_ids: set[int], chunks: list[ScoredChunk]
    ) -> list[Citation]:
        """Map cited IDs back to their source chunks."""
        citations = []
        for sc in chunks:
            if sc.citation_id in cited_ids:
                preview = sc.chunk.content[:150]
                if len(sc.chunk.content) > 150:
                    preview += "..."
                citations.append(Citation(
                    citation_id=sc.citation_id,
                    source=sc.chunk.source,
                    chunk_preview=preview,
                ))
        return citations

    def _assess_confidence(
        self, chunks: list[ScoredChunk], cited_ids: set[int]
    ) -> str:
        """
        Heuristic confidence assessment based on retrieval scores.

        high:   Top chunk score >= 0.6 and multiple citations
        medium: Top chunk score >= 0.4 or at least one citation
        low:    Otherwise
        """
        if not chunks:
            return "low"

        top_score = chunks[0].score
        num_cited = len(cited_ids)

        if top_score >= 0.6 and num_cited >= 2:
            return "high"
        elif top_score >= 0.4 or num_cited >= 1:
            return "medium"
        else:
            return "low"
