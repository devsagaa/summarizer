"""Sentence-transformer embedding wrapper (singleton-safe)."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model '%s' …", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded.")
    return model


class Embedder:
    """Thin wrapper around SentenceTransformer with L2-normalised outputs."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIM

    @property
    def model(self) -> SentenceTransformer:
        return _get_model(self.model_name)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts.
        Returns float32 array of shape (N, dim), L2-normalised for cosine sim.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # normalise for cosine via inner-product
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]
