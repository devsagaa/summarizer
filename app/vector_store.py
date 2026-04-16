"""FAISS-backed vector store with on-disk persistence."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP (inner-product = cosine for L2-normalised vectors).

    Metadata is stored in a parallel list keyed by the FAISS row index.
    The index and metadata are persisted to disk after every mutation.
    """

    _INDEX_FILE = "index.faiss"
    _META_FILE = "metadata.pkl"

    def __init__(self, dimension: int = 384, persist_dir: str | Path = "vector_db"):
        self.dimension = dimension
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]) -> None:
        """Add vectors + associated metadata to the store."""
        if embeddings.shape[0] == 0:
            return
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata_list)
        self._save()
        logger.debug("VectorStore: added %d vectors (total %d)", len(metadata_list), self.index.ntotal)

    def search(self, query: np.ndarray, k: int = 5) -> list[dict]:
        """
        Return the top-k most similar chunks.
        Each result dict contains all metadata keys + 'score'.
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        q = query.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS fills with -1 when ntotal < k
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results

    def search_by_doc(self, query: np.ndarray, doc_id: str, k: int = 5) -> list[dict]:
        """Retrieve top-k chunks scoped to a single document."""
        # Retrieve more candidates then filter — simple and accurate for small stores
        candidates = self.search(query, k=min(self.index.ntotal, k * 10))
        filtered = [c for c in candidates if c.get("doc_id") == doc_id]
        return filtered[:k]

    def delete_document(self, doc_id: str) -> int:
        """
        Remove all vectors belonging to `doc_id`.
        Returns the number of vectors removed.
        FAISS IndexFlatIP supports reconstruct(), so we rebuild the index.
        """
        keep = [(i, m) for i, m in enumerate(self.metadata) if m.get("doc_id") != doc_id]
        removed = len(self.metadata) - len(keep)

        if removed == 0:
            return 0

        new_index = faiss.IndexFlatIP(self.dimension)
        if keep:
            vectors = np.vstack([
                self.index.reconstruct(i).reshape(1, -1) for i, _ in keep
            ])
            new_index.add(vectors)

        self.index = new_index
        self.metadata = [m for _, m in keep]
        self._save()
        logger.info("VectorStore: removed %d vectors for doc_id=%s", removed, doc_id)
        return removed

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def doc_ids(self) -> list[str]:
        return list({m["doc_id"] for m in self.metadata})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.persist_dir / self._INDEX_FILE))
        with open(self.persist_dir / self._META_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load(self) -> None:
        idx_path = self.persist_dir / self._INDEX_FILE
        meta_path = self.persist_dir / self._META_FILE
        if idx_path.exists() and meta_path.exists():
            try:
                self.index = faiss.read_index(str(idx_path))
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(
                    "VectorStore: loaded %d vectors from '%s'",
                    self.index.ntotal,
                    self.persist_dir,
                )
            except Exception as exc:
                logger.warning("VectorStore: failed to load persisted data (%s). Starting fresh.", exc)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []
