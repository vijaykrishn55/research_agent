"""
services/generator.py
Grounded answer generation with citation tracking.

Produces structured research briefs from retrieved evidence using
Markdown formatting with section headings and cited claims.
"""

import re
from dataclasses import dataclass, field

from providers.base import LLMProvider, ProviderResponse
from services.retriever import ScoredChunk
from utils.log import get_logger

log = get_logger(__name__)


# ── System prompt — structured research brief ─────────────────────────────────

SYSTEM_PROMPT = """You are a research analyst producing professional research briefs from provided evidence.

Structure your response using Markdown. Use these sections when evidence supports them — omit any section that lacks evidence:

## Overview
2–4 sentence summary describing the subject. Cite every factual claim [N].

## Key Facts
- Bullet list of factual details (founded, headquarters, industry, size, mission, status, etc.)
- Only include facts directly stated in the evidence.

## Products / Services
Summarize the products, technologies, platforms, or services mentioned.

## Notable Information
Funding, milestones, partnerships, launches, acquisitions, or other significant findings.

## Evidence Summary
Short synthesis of what the evidence collectively shows. If sources agree, note that. If sources disagree, state both positions explicitly.

Rules:
1. Cite every factual claim with [N] where N is the chunk number (e.g. [1], [2][3]).
2. If multiple chunks support a claim, cite all of them: [1][2].
3. NEVER fabricate information or citations not grounded in the chunks.
4. Omit any section that has no supporting evidence.
5. If sources conflict, state both positions rather than choosing one.
6. Be concise and information-dense.
7. Only state "The available evidence does not address this question" when truly NO chunk is relevant."""


@dataclass
class Citation:
    """A single citation linking a claim to a source chunk."""
    citation_id: int
    source: str
    chunk_preview: str    # First 150 chars of the chunk
    source_type: str = "doc"   # "doc" for local documents, "web" for Tavily results


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
    """Generates grounded research briefs from retrieved chunks using an LLM."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, question: str, scored_chunks: list[ScoredChunk]) -> ResearchAnswer:
        """
        Generate a grounded research brief from retrieved chunks.

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
            f"[LOG] Generated brief: {len(citations)} citations, "
            f"confidence={confidence}, {response.tokens_used} tokens"
        )

        return answer

    def _build_prompt(self, question: str, chunks: list[ScoredChunk]) -> str:
        """Build the user prompt with numbered evidence chunks."""
        n = len(chunks)
        header = (
            f"You have {n} evidence chunk(s) below. "
            f"Produce a research brief answering the question using ALL relevant evidence.\n"
        )

        evidence_lines = []
        for sc in chunks:
            evidence_lines.append(
                f"--- Chunk [{sc.citation_id}] | Source: {sc.chunk.source} ---\n"
                f"{sc.chunk.content}"
            )

        evidence_block = "\n\n".join(evidence_lines)
        return f"{header}\n{evidence_block}\n\n--- Research Question ---\n{question}"

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
                source_type = "web" if sc.chunk.source.startswith("[WEB]") else "doc"
                citations.append(Citation(
                    citation_id=sc.citation_id,
                    source=sc.chunk.source,
                    chunk_preview=preview,
                    source_type=source_type,
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
