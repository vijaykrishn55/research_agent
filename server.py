"""
server.py
FastAPI application entry point with WebSocket streaming and frontend serving.

Initializes the RAG pipeline, mounts all API routes, serves the Claymorphism
frontend, and provides a WebSocket endpoint for real-time research events.

Run:
    uvicorn server:app --reload
    python server.py
"""

import asyncio
import os
import tempfile

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from services.pipeline import Pipeline
from routes import documents as documents_routes
from routes import research as research_routes
from utils.log import get_logger, setup_logging
from utils.events import research_log

setup_logging()
log = get_logger(__name__)

# -- App -----------------------------------------------------------------------

app = FastAPI(
    title="Research Agent",
    description="AI Research Agent with RAG-powered grounded answers and citations",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Pipeline singleton --------------------------------------------------------

pipeline = Pipeline()

# Inject pipeline into route modules
documents_routes.set_pipeline(pipeline)
research_routes.set_pipeline(pipeline)

# -- Mount API routes ----------------------------------------------------------

app.include_router(documents_routes.router)
app.include_router(research_routes.router)


# -- Static file serving (frontend) -------------------------------------------

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the Claymorphism frontend."""
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Research Agent</h1><p>Frontend not found. Place index.html in the static/ directory.</p>",
        status_code=200,
    )


# -- WebSocket: real-time research streaming -----------------------------------

@app.websocket("/ws/research")
async def ws_research(websocket: WebSocket):
    """Stream real-time pipeline events while running a research query.

    Client sends:
        {"question": "...", "mode": "local|web|hybrid|auto", "top_k": 8}

    Server streams:
        {"type": "event",   "data": {phase, message, duration_ms, ...}}
        {"type": "result",  "data": {full ResearchResponse}}
        {"type": "error",   "message": "..."}
        {"type": "done"}
    """
    await websocket.accept()
    try:
        raw = await websocket.receive_json()
    except Exception:
        await websocket.close(code=1008, reason="Invalid JSON")
        return

    question = raw.get("question", "").strip()
    mode = raw.get("mode", "local")
    top_k = raw.get("top_k")

    if not question or len(question) < 3:
        await websocket.send_json({"type": "error", "message": "Question must be at least 3 characters."})
        await websocket.close()
        return

    # Event bridge: pipeline thread -> asyncio queue -> WebSocket
    loop = asyncio.get_event_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    def on_event(event_dict):
        """Called from pipeline thread on every research_log.emit()."""
        loop.call_soon_threadsafe(event_queue.put_nowait, event_dict)

    research_log.subscribe(on_event)

    answer = None
    error_msg = None

    async def run_pipeline():
        nonlocal answer, error_msg
        try:
            answer = await loop.run_in_executor(
                None, pipeline.ask, question, top_k, mode,
            )
        except Exception as e:
            error_msg = str(e)

    task = asyncio.create_task(run_pipeline())

    # Stream events until pipeline finishes
    try:
        while not task.done():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.25)
                await websocket.send_json({"type": "event", "data": event})
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                research_log.unsubscribe(on_event)
                task.cancel()
                return

        # Drain any remaining events
        while True:
            try:
                event = event_queue.get_nowait()
                await websocket.send_json({"type": "event", "data": event})
            except asyncio.QueueEmpty:
                break

    except WebSocketDisconnect:
        research_log.unsubscribe(on_event)
        return

    research_log.unsubscribe(on_event)

    # Send final result
    if error_msg:
        await websocket.send_json({"type": "error", "message": error_msg})
    elif answer:
        result = {
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
            "metrics": answer.metrics,
        }
        await websocket.send_json({"type": "result", "data": result})

    await websocket.send_json({"type": "done"})
    await websocket.close()


# -- Settings API ---------------------------------------------------------------

_SENSITIVE_KEYS = {"groq_api_key", "openai_api_key", "tavily_api_key"}


class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    groq_model: str | None = None
    openai_model: str | None = None
    top_k: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    similarity_threshold: float | None = None
    max_generation_tokens: int | None = None
    generation_temperature: float | None = None
    tavily_api_key: str | None = None
    tavily_max_results: int | None = None
    tavily_search_depth: str | None = None


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "••••••••"
    return value[:4] + "••••••••" + value[-4:]


@app.get("/api/settings")
async def get_settings():
    """Return current configuration. Sensitive keys are masked."""
    return {
        "llm_provider": settings.LLM_PROVIDER,
        "groq_api_key": _mask(settings.GROQ_API_KEY),
        "groq_api_key_set": bool(settings.GROQ_API_KEY),
        "openai_api_key": _mask(settings.OPENAI_API_KEY),
        "openai_api_key_set": bool(settings.OPENAI_API_KEY),
        "groq_model": settings.GROQ_MODEL,
        "openai_model": settings.OPENAI_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_device": settings.EMBEDDING_DEVICE,
        "top_k": settings.TOP_K,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "max_generation_tokens": settings.MAX_GENERATION_TOKENS,
        "generation_temperature": settings.GENERATION_TEMPERATURE,
        "tavily_api_key": _mask(settings.TAVILY_API_KEY),
        "tavily_api_key_set": bool(settings.TAVILY_API_KEY),
        "tavily_max_results": settings.TAVILY_MAX_RESULTS,
        "tavily_search_depth": settings.TAVILY_SEARCH_DEPTH,
    }


@app.put("/api/settings")
async def update_settings(updates: SettingsUpdate):
    """Update configuration in memory (applies immediately) and persist to .env."""
    changed = {}

    _mapping = {
        "llm_provider": ("LLM_PROVIDER", str),
        "groq_model": ("GROQ_MODEL", str),
        "openai_model": ("OPENAI_MODEL", str),
        "top_k": ("TOP_K", int),
        "chunk_size": ("CHUNK_SIZE", int),
        "chunk_overlap": ("CHUNK_OVERLAP", int),
        "similarity_threshold": ("SIMILARITY_THRESHOLD", float),
        "max_generation_tokens": ("MAX_GENERATION_TOKENS", int),
        "generation_temperature": ("GENERATION_TEMPERATURE", float),
        "tavily_max_results": ("TAVILY_MAX_RESULTS", int),
        "tavily_search_depth": ("TAVILY_SEARCH_DEPTH", str),
    }

    for field_name, (attr, cast) in _mapping.items():
        val = getattr(updates, field_name, None)
        if val is not None:
            setattr(settings, attr, cast(val))
            changed[attr] = str(val)

    # Handle sensitive keys — only update if non-empty and not a masked echo
    _key_mapping = {
        "groq_api_key": "GROQ_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "tavily_api_key": "TAVILY_API_KEY",
    }
    for field_name, attr in _key_mapping.items():
        val = getattr(updates, field_name, None)
        if val and "•" not in val:  # Ignore masked values echoed back
            setattr(settings, attr, val)
            os.environ[attr] = val
            changed[attr] = "***updated***"

    # Persist to .env file
    if changed:
        _persist_env(changed)

    return {"success": True, "updated": list(changed.keys())}


def _persist_env(changed: dict):
    """Merge changed values into the .env file, creating it if needed."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    existing: dict[str, str] = {}
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
    existing.update(changed)
    with open(env_path, "w", encoding="utf-8") as f:
        for k, v in sorted(existing.items()):
            f.write(f"{k}={v}\n")


# -- Export download endpoint --------------------------------------------------

class ExportRequest(BaseModel):
    format: str = Field(..., pattern="^(md|pdf)$", description="Export format: md or pdf")


@app.post("/api/research/export")
async def export_research(req: ExportRequest):
    """Export the last research answer as a downloadable Markdown or PDF file."""
    from services.generator import Citation
    from services.exporter import export_markdown, export_pdf

    last = pipeline.last_answer
    if last is None:
        raise HTTPException(status_code=404, detail="No research to export. Submit a question first.")

    try:
        if req.format == "md":
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as tmp:
                tmp_path = tmp.name
            export_markdown(last, list(last.citations), tmp_path)
            return FileResponse(
                path=tmp_path,
                media_type="text/markdown",
                filename=f"research_{last.question[:30].strip().replace(' ', '_')}.md",
                background=None,
            )
        elif req.format == "pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            export_pdf(last, list(last.citations), tmp_path)
            return FileResponse(
                path=tmp_path,
                media_type="application/pdf",
                filename=f"research_{last.question[:30].strip().replace(' ', '_')}.pdf",
                background=None,
            )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


# -- Direct execution ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print()
    print("   Research Agent v2.0")
    print(f"   http://{settings.HOST}:{settings.PORT}")
    print(f"   WebSocket: ws://{settings.HOST}:{settings.PORT}/ws/research")
    print()
    uvicorn.run("server:app", host=settings.HOST, port=settings.PORT, reload=True)
