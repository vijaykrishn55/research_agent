"""
data/index_store.py
FAISS-backed vector index with disk persistence.

Stores embeddings alongside chunk metadata so the index survives restarts.
"""

import json
import os

import faiss
import numpy as np

from data.chunker import Chunk
from config import settings
from utils.log import get_logger

log = get_logger(__name__)


class IndexStore:
    """
    Manages a FAISS index and its associated chunk metadata.

    Persists to disk as:
        {index_dir}/index.faiss   — the FAISS binary index
        {index_dir}/meta.json     — chunk metadata + ingested file list
    """

    def __init__(self, index_dir: str | None = None):
        self.index_dir = index_dir or settings.INDEX_DIR
        self._index_path = os.path.join(self.index_dir, "index.faiss")
        self._meta_path = os.path.join(self.index_dir, "meta.json")

        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []
        self.ingested_files: set[str] = set()

        self._load()

    # -- Public API ------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the index."""
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings shape must be (num_chunks, dim)")

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            log.info(f"[IDX]  Created new FAISS index (dim={dim})")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        # Track which files have been ingested
        for chunk in chunks:
            self.ingested_files.add(chunk.source)

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """
        Search the index for the most similar chunks.

        Returns list of (Chunk, similarity_score) tuples, sorted by relevance.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = min(top_k or settings.TOP_K, self.index.ntotal)
        query = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if score < settings.SIMILARITY_THRESHOLD:
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        os.makedirs(self.index_dir, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, self._index_path)

        meta = {
            "chunks": [c.to_dict() for c in self.chunks],
            "ingested_files": sorted(self.ingested_files),
        }
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        log.info(f"[SAVE] Index saved ({self.index.ntotal} vectors, {len(self.ingested_files)} files)")

    def clear(self) -> None:
        """Remove all data and delete persisted files."""
        self.index = None
        self.chunks = []
        self.ingested_files = set()

        for path in (self._index_path, self._meta_path):
            if os.path.exists(path):
                os.remove(path)

        log.info("[DEL]  Index cleared")

    def is_file_ingested(self, filename: str) -> bool:
        """Check if a file has already been ingested."""
        return filename in self.ingested_files

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    # -- Private ---------------------------------------------------------------

    def _load(self) -> None:
        """Load existing index and metadata from disk if available."""
        if not os.path.exists(self._index_path) or not os.path.exists(self._meta_path):
            return

        try:
            self.index = faiss.read_index(self._index_path)

            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.chunks = [Chunk.from_dict(d) for d in meta.get("chunks", [])]
            self.ingested_files = set(meta.get("ingested_files", []))

            log.info(
                f"[LOAD] Loaded existing index: {self.index.ntotal} vectors, "
                f"{len(self.chunks)} chunks, {len(self.ingested_files)} files"
            )
        except Exception as e:
            log.error(f"[ERR] Failed to load index: {e}. Starting fresh.")
            self.index = None
            self.chunks = []
            self.ingested_files = set()
