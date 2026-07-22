"""
server.py
FastAPI application entry point.

Initializes the RAG pipeline and mounts all API routes.

Run:
    uvicorn server:app --reload
    python server.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from services.pipeline import Pipeline
from routes import documents as documents_routes
from routes import research as research_routes
from utils.log import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

# -- App -----------------------------------------------------------------------

app = FastAPI(
    title="Research Agent",
    description="AI Research Agent with RAG-powered grounded answers and citations",
    version="1.0.0",
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

# -- Mount routes --------------------------------------------------------------

app.include_router(documents_routes.router)
app.include_router(research_routes.router)


@app.get("/")
async def root():
    return {
        "service": "Research Agent",
        "version": "1.0.0",
        "endpoints": [
            "/api/documents/upload",
            "/api/documents",
            "/api/research",
            "/api/research/status",
        ],
    }


# -- Direct execution ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    log.info(f"[START] Starting Research Agent on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
