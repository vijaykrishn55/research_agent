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
from utils.events import research_log

log = get_logger(__name__)


# ── System prompt — structured research brief ─────────────────────────────────

SYSTEM_PROMPT = """You are a research analyst producing professional, structured research briefs from provided evidence.

FORMAT:
- Use Markdown with section headings (##) and bullet lists.
- Choose section headings that fit the SUBJECT and EVIDENCE naturally. Do NOT use a fixed template.
  For a company you might use: Overview, Products, Team, Recent Developments.
  For a technology: Overview, How It Works, Applications, Current State.
  For a person: Background, Career, Notable Work.
  Adapt freely — use whatever structure best organises the available evidence.
- Only create a section if the evidence supports it. Omit sections with no evidence.

EVIDENCE GROUNDING (CRITICAL):
- Every factual claim MUST include a citation: [1], [2][3].
- NEVER state a fact (dates, names, numbers, locations) unless it appears explicitly in the evidence chunks.
- If a commonly expected detail (e.g., founding year, headquarters) is NOT in the evidence, do NOT guess. Either omit it or state it is not available in the retrieved evidence.
- If sources disagree, present both positions with their citations.

STYLE:
- Be concise and information-dense.
- Write like a research analyst, not a chatbot.
- Do NOT add a concluding summary or synthesis section — the section headings above are sufficient.

If NO chunk is relevant, state: "The available evidence does not address this question.\""""


@dataclass
class Citation:
    """A single citation linking a claim to a source chunk."""
    citation_id: int
    source: str
    chunk_preview: str    # First 150 chars of the chunk
    source_type: str = "doc"   # "doc" for local documents, "web" for Tavily results
    full_text: str | None = None  # Complete chunk content for evidence display


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

        research_log.emit(
            "generation",
            f"LLM call complete — {response.provider}/{response.model}",
            duration_ms=response.latency_ms,
            details={
                "model": response.model,
                "provider": response.provider,
                "tokens_used": response.tokens_used,
            },
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
        """Map cited IDs back to their source chunks.

        Every ID referenced in the answer gets a Citation entry so that
        no citation is silently dropped from the display.  IDs that don't
        match any retrieved chunk get a placeholder.
        """
        # Build a lookup from citation_id -> ScoredChunk
        chunk_map: dict[int, ScoredChunk] = {
            sc.citation_id: sc for sc in chunks
        }

        citations: list[Citation] = []
        for cid in sorted(cited_ids):
            sc = chunk_map.get(cid)
            if sc is not None:
                preview = sc.chunk.content[:150]
                if len(sc.chunk.content) > 150:
                    preview += "..."
                source_type = "web" if sc.chunk.source.startswith("[WEB]") else "doc"
                citations.append(Citation(
                    citation_id=cid,
                    source=sc.chunk.source,
                    chunk_preview=preview,
                    source_type=source_type,
                    full_text=sc.chunk.content,
                ))
            else:
                # Placeholder for IDs the LLM cited but weren't in our chunks
                citations.append(Citation(
                    citation_id=cid,
                    source="[Source not retrieved]",
                    chunk_preview="This citation was referenced in the answer but "
                                  "its source chunk was not found in the retrieved results.",
                    source_type="unknown",
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


# ── Persistence — save / load last answer ─────────────────────────────────────

import json as _json
import os as _os


def _cache_path() -> str:
    """Return the path for the last-answer cache file."""
    from config import settings
    return _os.path.join(settings.INDEX_DIR, ".last_answer.json")


def save_last_answer(answer: "ResearchAnswer") -> None:
    """Persist a ResearchAnswer to disk as JSON (for the explain command)."""
    data = {
        "question": answer.question,
        "answer": answer.answer,
        "citations": [
            {
                "citation_id": c.citation_id,
                "source": c.source,
                "chunk_preview": c.chunk_preview,
                "source_type": c.source_type,
                "full_text": c.full_text,
            }
            for c in answer.citations
        ],
        "chunks_retrieved": answer.chunks_retrieved,
        "chunks_cited": answer.chunks_cited,
        "confidence": answer.confidence,
    }
    path = _cache_path()
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
    log.debug(f"[CACHE] Saved last answer to {path}")


def load_last_answer() -> "ResearchAnswer | None":
    """Load the most recently saved ResearchAnswer from disk."""
    path = _cache_path()
    if not _os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    citations = [
        Citation(
            citation_id=c["citation_id"],
            source=c["source"],
            chunk_preview=c["chunk_preview"],
            source_type=c["source_type"],
            full_text=c.get("full_text"),
        )
        for c in data.get("citations", [])
    ]
    return ResearchAnswer(
        question=data["question"],
        answer=data["answer"],
        citations=citations,
        chunks_retrieved=data.get("chunks_retrieved", 0),
        chunks_cited=data.get("chunks_cited", 0),
        confidence=data.get("confidence", "low"),
    )
