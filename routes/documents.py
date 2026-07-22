"""
routes/documents.py
Document upload and index management endpoints.

POST /api/documents/upload   — upload and ingest files
GET  /api/documents          — list ingested files
DELETE /api/documents        — clear the entire index
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from utils.log import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

# Pipeline is injected at startup via set_pipeline()
_pipeline = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


class IngestResponse(BaseModel):
    file: str
    status: str
    chunks: int = 0
    elapsed_seconds: float = 0
    reason: str = ""


class IndexStatus(BaseModel):
    total_chunks: int
    total_vectors: int
    ingested_files: list[str]


# POST /api/documents/upload
@router.post("/upload", response_model=list[IngestResponse])
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload one or more files for ingestion into the research index."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    results = []
    for file in files:
        try:
            content = await file.read()
            result = _pipeline.ingest_bytes(file.filename, content)
            results.append(result)
        except Exception as e:
            log.error(f"[ERR] Failed to ingest {file.filename}: {e}")
            results.append({
                "file": file.filename,
                "status": "error",
                "reason": str(e),
            })

    return results


# GET /api/documents
@router.get("/", response_model=IndexStatus)
async def list_documents():
    """List all ingested files and index statistics."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    status = _pipeline.status()
    return IndexStatus(
        total_chunks=status["total_chunks"],
        total_vectors=status["total_vectors"],
        ingested_files=status["ingested_files"],
    )


# DELETE /api/documents
@router.delete("/")
async def clear_index():
    """Clear the entire document index."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    _pipeline.clear()
    return {"success": True, "message": "Index cleared"}
