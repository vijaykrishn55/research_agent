"""
providers/embedder.py
Sentence-transformer embedding model wrapper.

Loads the model once, reuses it for all embed calls.
Embeddings are returned as numpy arrays ready for FAISS.
"""

import time

import numpy as np

from config import settings
from utils.log import get_logger

log = get_logger(__name__)


class Embedder:
    """
    Wraps sentence-transformers for local embedding generation.

    The model is loaded lazily on first use and cached for the process lifetime.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self._model = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            log.info(f"[INIT] Loading embedding model: {self.model_name} (device={self.device})")
            start = time.time()

            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

            elapsed = round(time.time() - start, 1)
            log.info(f"[OK] Embedding model ready ({elapsed}s)")

        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into dense vectors.

        Returns numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([], dtype="float32").reshape(0, 0)

        start = time.time()
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # IndexStore handles normalization
        )
        elapsed = round((time.time() - start) * 1000, 1)
        log.info(f"[EMB] Embedded {len(texts)} texts ({elapsed}ms)")

        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, dim)."""
        return self.embed([text])

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self.model.get_sentence_embedding_dimension()
