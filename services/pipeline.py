"""
services/pipeline.py
RAG pipeline orchestrator.

Single entry point that coordinates:
  1. Document ingestion (load -> chunk -> embed -> index)
  2. Research queries (retrieve -> generate)

Supports four retrieval modes:
  local   — FAISS index only (default, unchanged behaviour)
  web     — Tavily web search only
  hybrid  — local + web, merged, ranked, deduplicated
  auto    — local first; falls back to hybrid if evidence is insufficient
"""

import os
import time

from data.loader import load_file, load_directory, LoadedDocument
from data.chunker import chunk_document
from data.index_store import IndexStore
from providers.embedder import Embedder
from providers.base import LLMProvider, create_provider
from services.retriever import Retriever, ScoredChunk
from services.generator import Generator, ResearchAnswer, save_last_answer
from config import settings
from utils.log import get_logger
from utils.events import research_log

log = get_logger(__name__)

# Valid retrieval modes
VALID_MODES = {"local", "web", "hybrid", "auto"}


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
        self._last_answer: ResearchAnswer | None = None

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

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        mode: str = "local",
    ) -> ResearchAnswer:
        """
        Answer a research question.

        mode="local"   — FAISS retrieval only (default, backward-compatible)
        mode="web"     — Tavily web search only
        mode="hybrid"  — local + web, merged, ranked, deduplicated
        mode="auto"    — local first; upgrades to hybrid if evidence is thin
        """
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(VALID_MODES))}"
            )

        start = time.time()
        k = top_k or settings.TOP_K
        research_plan = None  # Will be set if planner runs

        research_log.reset()
        research_log.emit(
            "pipeline", f"Starting research ({mode} mode)",
            details={"question": question, "mode": mode, "top_k": k},
        )

        if mode == "local":
            t0 = time.time()
            research_log.emit("retrieval", "Searching local index…")
            scored_chunks = self._retriever.retrieve(question, k)
            research_log.emit(
                "retrieval", f"Local search complete — {len(scored_chunks)} chunk(s)",
                duration_ms=round((time.time() - t0) * 1000, 1),
                details={"chunks": len(scored_chunks)},
            )

        elif mode == "web":
            t0 = time.time()
            research_log.emit("search", "Starting web retrieval…")
            research_plan, scored_chunks = self._web_retrieve(question, k)
            research_log.emit(
                "search", f"Web retrieval done — {len(scored_chunks)} result(s)",
                duration_ms=round((time.time() - t0) * 1000, 1),
                details={"chunks": len(scored_chunks)},
            )

        elif mode == "hybrid":
            t0 = time.time()
            research_log.emit("retrieval", "Searching local index…")
            local_chunks = self._retriever.retrieve(question, k)
            research_log.emit(
                "retrieval", f"Local search complete — {len(local_chunks)} chunk(s)",
                duration_ms=round((time.time() - t0) * 1000, 1),
            )

            t0 = time.time()
            research_log.emit("search", "Starting web retrieval…")
            research_plan, web_chunks = self._web_retrieve(question, k, citation_offset=len(local_chunks))
            research_log.emit(
                "search", f"Web retrieval done — {len(web_chunks)} result(s)",
                duration_ms=round((time.time() - t0) * 1000, 1),
            )

            scored_chunks = self._merge(local_chunks, web_chunks, k)

        else:  # auto
            t0 = time.time()
            research_log.emit("retrieval", "Searching local index…")
            local_chunks = self._retriever.retrieve(question, k)
            research_log.emit(
                "retrieval", f"Local search complete — {len(local_chunks)} chunk(s)",
                duration_ms=round((time.time() - t0) * 1000, 1),
            )

            if self._is_sufficient(local_chunks):
                log.info("[AUTO] Local evidence sufficient — skipping web search")
                research_log.emit(
                    "pipeline", "Local evidence sufficient — skipping web search",
                    details={"top_score": local_chunks[0].score if local_chunks else 0},
                )
                scored_chunks = local_chunks
            else:
                if settings.TAVILY_API_KEY:
                    log.info("[AUTO] Insufficient local evidence — falling back to hybrid")
                    research_log.emit(
                        "pipeline", "Insufficient local evidence — falling back to hybrid",
                        details={
                            "top_score": local_chunks[0].score if local_chunks else 0,
                            "chunks": len(local_chunks),
                        },
                    )
                    t0 = time.time()
                    research_log.emit("search", "Starting web retrieval…")
                    research_plan, web_chunks = self._web_retrieve(question, k, citation_offset=len(local_chunks))
                    research_log.emit(
                        "search", f"Web retrieval done — {len(web_chunks)} result(s)",
                        duration_ms=round((time.time() - t0) * 1000, 1),
                    )
                    scored_chunks = self._merge(local_chunks, web_chunks, k)
                else:
                    log.warning(
                        "[AUTO] Evidence thin but TAVILY_API_KEY not set — "
                        "answering with local evidence only"
                    )
                    research_log.emit(
                        "pipeline",
                        "Evidence thin but TAVILY_API_KEY not set — using local only",
                        level="warn",
                    )
                    scored_chunks = local_chunks

        t0 = time.time()
        research_log.emit(
            "generation",
            f"Generating answer from {len(scored_chunks)} chunk(s)…",
        )
        answer = self.generator.generate(question, scored_chunks)
        research_log.emit(
            "generation",
            f"Answer generated — {answer.chunks_cited} citation(s), "
            f"confidence: {answer.confidence}",
            duration_ms=round((time.time() - t0) * 1000, 1),
            details={
                "citations": answer.chunks_cited,
                "confidence": answer.confidence,
                "tokens": answer.metrics.get("tokens_used", "N/A"),
            },
        )

        total_ms = round((time.time() - start) * 1000, 1)
        answer.metrics["total_latency_ms"] = total_ms
        answer.metrics["mode"] = mode

        # Attach research plan for display
        if research_plan:
            answer.metrics["research_plan"] = [
                {"query": pq.query, "purpose": pq.purpose}
                for pq in research_plan
            ]

        research_log.emit(
            "pipeline", "Research complete",
            duration_ms=total_ms,
            details={"mode": mode, "total_chunks": len(scored_chunks)},
        )

        # Attach the event log for verbose display / API consumers
        answer.metrics["event_log"] = research_log.to_dicts()

        # Cache for the evidence API endpoint
        self._last_answer = answer

        # Persist to disk so `python cli.py explain <N>` works across sessions
        save_last_answer(answer)

        return answer

    @property
    def last_answer(self) -> ResearchAnswer | None:
        """The most recent ResearchAnswer (for the evidence endpoint)."""
        return self._last_answer

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
            "tavily_enabled": bool(settings.TAVILY_API_KEY),
        }

    # -- Private helpers -------------------------------------------------------

    def _web_retrieve(
        self,
        question: str,
        top_k: int,
        citation_offset: int = 0,
    ) -> tuple[list | None, list[ScoredChunk]]:
        """Run web search — single query for simple, multi-query for broad.

        Returns (research_plan, scored_chunks) where research_plan is a list
        of PlannedQuery objects (or None for simple queries).
        """
        from services.research_planner import classify_complexity, plan_research
        from services.tavily_search import tavily_search, tavily_search_multi

        t0 = time.time()
        research_log.emit("planner", "Classifying query complexity…")
        complexity = classify_complexity(question, self.generator.provider)
        research_log.emit(
            "planner", f"Query classified as: {complexity}",
            duration_ms=round((time.time() - t0) * 1000, 1),
            details={"complexity": complexity},
        )

        if complexity == "simple":
            research_log.emit("search", "Simple query — single Tavily search")
            t1 = time.time()
            chunks = tavily_search(query=question, citation_offset=citation_offset)
            research_log.emit(
                "search", f"Tavily returned {len(chunks)} result(s)",
                duration_ms=round((time.time() - t1) * 1000, 1),
            )
            return None, chunks
        else:
            t1 = time.time()
            research_log.emit("planner", "Broad query — planning sub-queries…")
            queries = plan_research(question, self.generator.provider)
            research_log.emit(
                "planner", f"Research plan created — {len(queries)} sub-quer(ies)",
                duration_ms=round((time.time() - t1) * 1000, 1),
                details={
                    "queries": [{"query": pq.query, "purpose": pq.purpose} for pq in queries],
                },
            )

            t1 = time.time()
            research_log.emit(
                "search", f"Executing {len(queries)} concurrent web searches…",
            )
            chunks = tavily_search_multi(
                planned_queries=queries,
                citation_offset=citation_offset,
            )
            research_log.emit(
                "search", f"All web searches complete — {len(chunks)} total result(s)",
                duration_ms=round((time.time() - t1) * 1000, 1),
            )
            return queries, chunks

    def _merge(
        self,
        local_chunks: list[ScoredChunk],
        web_chunks: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """
        Merge local and web chunks, sort by score, deduplicate, slice to top_k.
        Re-assigns citation_ids sequentially after sorting.
        """
        combined = local_chunks + web_chunks
        pre_dedup = len(combined)

        research_log.emit(
            "pipeline",
            f"Merging {len(local_chunks)} local + {len(web_chunks)} web chunk(s)…",
        )

        # Sort by relevance score descending
        combined.sort(key=lambda sc: sc.score, reverse=True)

        # Deduplicate by Jaccard token overlap
        combined = _dedup(combined, settings.DEDUP_SIMILARITY_THRESHOLD)
        dropped = pre_dedup - len(combined)

        # Take top-k
        combined = combined[:top_k]

        # Re-assign citation IDs so they are 1..N after ranking
        for idx, sc in enumerate(combined):
            sc.citation_id = idx + 1

        research_log.emit(
            "pipeline",
            f"Merge complete — {pre_dedup} → {len(combined)} chunk(s) "
            f"({dropped} duplicate(s) dropped, top-{top_k} kept)",
            details={
                "local": len(local_chunks),
                "web": len(web_chunks),
                "dedup_dropped": dropped,
                "final": len(combined),
            },
        )

        log.info(
            f"[MERGE] {len(local_chunks)} local + {len(web_chunks)} web "
            f"→ {len(combined)} after dedup/rank"
        )
        return combined

    def _is_sufficient(self, chunks: list[ScoredChunk]) -> bool:
        """
        Heuristic: local evidence is 'sufficient' when we have at least
        AUTO_MIN_CHUNKS results whose top score meets AUTO_MIN_SCORE.
        """
        if len(chunks) < settings.AUTO_MIN_CHUNKS:
            return False
        return chunks[0].score >= settings.AUTO_MIN_SCORE


# -- Module-level helpers ------------------------------------------------------

def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedup(chunks: list[ScoredChunk], threshold: float) -> list[ScoredChunk]:
    """
    Remove chunks whose content overlaps heavily with a higher-ranked chunk.

    Operates on a pre-sorted list (highest score first). Chunks with
    Jaccard similarity > threshold against any already-kept chunk are dropped.
    """
    seen_tokens: list[set[str]] = []
    result: list[ScoredChunk] = []

    for sc in chunks:
        tokens = set(sc.chunk.content.lower().split())
        if any(_jaccard(tokens, s) > threshold for s in seen_tokens):
            log.debug(f"[DEDUP] Dropped near-duplicate chunk: {sc.chunk.chunk_id}")
            continue
        seen_tokens.append(tokens)
        result.append(sc)

    return result
