"""
routes/research.py
Research question endpoint.

POST /api/research   — ask a research question
GET  /api/research/status — system status
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.log import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/research", tags=["research"])

# Pipeline is injected at startup via set_pipeline()
_pipeline = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


# -- Request / Response Models -------------------------------------------------

class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The research question to answer")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    mode: str = Field(
        "local",
        pattern="^(local|web|hybrid|auto)$",
        description="Retrieval mode: local | web | hybrid | auto",
    )


class CitationModel(BaseModel):
    citation_id: int
    source: str
    chunk_preview: str
    source_type: str = "doc"   # "doc" | "web"
    full_text: str | None = None


class ResearchResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]
    chunks_retrieved: int
    chunks_cited: int
    confidence: str
    metrics: dict


class EvidenceResponse(BaseModel):
    citation_id: int
    source: str
    source_type: str
    full_text: str | None = None
    chunk_preview: str


class StatusResponse(BaseModel):
    total_chunks: int
    total_vectors: int
    ingested_files: list[str]
    embedding_model: str
    llm_provider: str
    llm_model: str


# -- Endpoints -----------------------------------------------------------------

# POST /api/research
@router.post("/", response_model=ResearchResponse)
async def ask_question(request: ResearchRequest):
    """Submit a research question and receive a grounded answer with citations."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        answer = _pipeline.ask(request.question, request.top_k, mode=request.mode)
    except Exception as e:
        log.error(f"[ERR] Research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return ResearchResponse(
        question=answer.question,
        answer=answer.answer,
        citations=[
            CitationModel(
                citation_id=c.citation_id,
                source=c.source,
                chunk_preview=c.chunk_preview,
                source_type=c.source_type,
                full_text=c.full_text,
            )
            for c in answer.citations
        ],
        chunks_retrieved=answer.chunks_retrieved,
        chunks_cited=answer.chunks_cited,
        confidence=answer.confidence,
        metrics=answer.metrics,
    )


# GET /api/research/status
@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    status = _pipeline.status()
    return StatusResponse(
        total_chunks=status["total_chunks"],
        total_vectors=status["total_vectors"],
        ingested_files=status["ingested_files"],
        embedding_model=status["embedding_model"],
        llm_provider=status["llm_provider"],
        llm_model=status["llm_model"],
    )


# GET /api/research/evidence/{citation_id}
@router.get("/evidence/{citation_id}", response_model=EvidenceResponse)
async def get_evidence(citation_id: int):
    """Return the full evidence chunk for a specific citation from the last research answer."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    last = _pipeline.last_answer
    if last is None:
        raise HTTPException(
            status_code=404,
            detail="No research has been performed yet. Submit a question first.",
        )

    # Find the matching citation
    for c in last.citations:
        if c.citation_id == citation_id:
            return EvidenceResponse(
                citation_id=c.citation_id,
                source=c.source,
                source_type=c.source_type,
                full_text=c.full_text,
                chunk_preview=c.chunk_preview,
            )

    available = sorted(c.citation_id for c in last.citations)
    raise HTTPException(
        status_code=404,
        detail=f"Citation [{citation_id}] not found. Available IDs: {available}",
    )
