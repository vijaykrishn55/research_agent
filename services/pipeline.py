"""
services/pipeline.py
RAG pipeline orchestrator.

Single entry point that coordinates:
  1. Document ingestion (load -> chunk -> embed -> index)
  2. Research queries (retrieve -> generate)
"""

import os
import time

from data.loader import load_file, load_directory, LoadedDocument
from data.chunker import chunk_document
from data.index_store import IndexStore
from providers.embedder import Embedder
from providers.base import LLMProvider, create_provider
from services.retriever import Retriever
from services.generator import Generator, ResearchAnswer
from config import settings
from utils.log import get_logger

log = get_logger(__name__)


class Pipeline:
    """
    Orchestrates the full RAG pipeline.

    Lazily initializes all components on first use.
    Persists the index to disk after every ingestion.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        index_dir: str | None = None,
    ):
        self._index_store = IndexStore(index_dir)
        self._embedder = Embedder()
        self._retriever = Retriever(self._embedder, self._index_store)
        self._provider = provider
        self._generator: Generator | None = None

    @property
    def generator(self) -> Generator:
        """Lazy-init the generator (defers provider creation until needed)."""
        if self._generator is None:
            if self._provider is None:
                self._provider = create_provider()
            self._generator = Generator(self._provider)
        return self._generator

    # -- Ingestion -------------------------------------------------------------

    def ingest_file(self, path: str) -> dict:
        """
        Ingest a single file: load -> chunk -> embed -> index -> save.

        Returns a summary dict with ingestion stats.
        """
        filename = os.path.basename(path)

        if self._index_store.is_file_ingested(filename):
            log.info(f"[SKIP]  Skipping {filename} (already ingested)")
            return {"file": filename, "status": "skipped", "reason": "already ingested"}

        start = time.time()
        doc = load_file(path)
        chunks = chunk_document(doc.content, doc.source)

        if not chunks:
            return {"file": filename, "status": "skipped", "reason": "no content"}

        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed(texts)

        self._index_store.add(chunks, embeddings)
        self._index_store.save()

        elapsed = round(time.time() - start, 1)
        log.info(f"[OK] Ingested {filename}: {len(chunks)} chunks in {elapsed}s")

        return {
            "file": filename,
            "status": "ingested",
            "chunks": len(chunks),
            "elapsed_seconds": elapsed,
        }

    def ingest_directory(self, path: str) -> list[dict]:
        """Ingest all supported files from a directory."""
        documents = load_directory(path)
        results = []

        for doc in documents:
            # Build a temporary path for load_file compatibility
            # Since we already have LoadedDocument, chunk directly
            if self._index_store.is_file_ingested(doc.source):
                log.info(f"[SKIP]  Skipping {doc.source} (already ingested)")
                results.append({"file": doc.source, "status": "skipped", "reason": "already ingested"})
                continue

            start = time.time()
            chunks = chunk_document(doc.content, doc.source)

            if not chunks:
                results.append({"file": doc.source, "status": "skipped", "reason": "no content"})
                continue

            texts = [c.content for c in chunks]
            embeddings = self._embedder.embed(texts)
            self._index_store.add(chunks, embeddings)

            elapsed = round(time.time() - start, 1)
            results.append({
                "file": doc.source,
                "status": "ingested",
                "chunks": len(chunks),
                "elapsed_seconds": elapsed,
            })

        # Save once after all files
        self._index_store.save()

        total_chunks = sum(r.get("chunks", 0) for r in results)
        ingested = sum(1 for r in results if r["status"] == "ingested")
        log.info(f"[+] Ingestion complete: {ingested} files, {total_chunks} chunks")

        return results

    def ingest_bytes(self, filename: str, content: bytes) -> dict:
        """Ingest a file from raw bytes (used by the API upload endpoint)."""
        import tempfile

        ext = os.path.splitext(filename)[1].lower()
        if ext not in settings.SUPPORTED_EXTENSIONS:
            return {
                "file": filename,
                "status": "error",
                "reason": f"Unsupported file type: {ext}",
            }

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            return self.ingest_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    # -- Research --------------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> ResearchAnswer:
        """
        Answer a research question using RAG.

        Steps:
        1. Retrieve the most relevant chunks
        2. Generate a grounded answer with citations
        """
        start = time.time()

        scored_chunks = self._retriever.retrieve(question, top_k)
        answer = self.generator.generate(question, scored_chunks)

        total_ms = round((time.time() - start) * 1000, 1)
        answer.metrics["total_latency_ms"] = total_ms

        return answer

    # -- Index Management ------------------------------------------------------

    def clear(self) -> None:
        """Clear the entire index."""
        self._index_store.clear()

    def status(self) -> dict:
        """Return current pipeline status."""
        return {
            "total_chunks": self._index_store.total_chunks,
            "total_vectors": self._index_store.total_vectors,
            "ingested_files": sorted(self._index_store.ingested_files),
            "embedding_model": self._embedder.model_name,
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": getattr(self._provider, "model", settings.GROQ_MODEL),
            "index_dir": self._index_store.index_dir,
        }
