import json
import logging
from pathlib import Path
from typing import List

import faiss
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FAISSRetriever:
    """FAISS-based retriever using IndexFlatIP for cosine similarity search."""

    def __init__(self, embedding_dim: int = 128) -> None:
        """
        Initialize empty FAISS index.

        Args:
            embedding_dim: Dimensionality of embedding vectors.
        """
        self.embedding_dim = embedding_dim
        self.index: faiss.IndexFlatIP = None
        self.metadata: List[dict] = []

    def __enter__(self) -> "FAISSRetriever":
        return self

    def __exit__(self, *args) -> None:
        pass

    def build_index(
        self, embeddings: np.ndarray, metadata: List[dict]
    ) -> None:
        """
        Build FAISS index from L2-normalized embeddings.

        Args:
            embeddings: Array of shape (N, embedding_dim), must be L2-normalized.
            metadata: List of dicts, each containing at minimum image_path and report.
        """
        assert embeddings.shape[0] == len(metadata), \
            "Number of embeddings must match number of metadata entries"
        dim = embeddings.shape[1]
        self.embedding_dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata
        logger.info("Built FAISS IndexFlatIP with %d vectors (dim=%d)", len(metadata), dim)

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[dict]:
        """
        Retrieve top-k most similar entries for a query embedding.

        Args:
            query_embedding: 1D or 2D array of shape (embedding_dim,) or (1, embedding_dim).
            top_k: Number of results to return.

        Returns:
            List of dicts sorted by score descending, each with score, image_path, report.
        """
        assert self.index is not None, "Index not built. Call build_index() first."
        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Persist FAISS index and metadata to disk.

        Args:
            index_path: Path for the .faiss index file.
            metadata_path: Path for the metadata JSON file.
        """
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        logger.info("Saved FAISS index to %s and metadata to %s", index_path, metadata_path)

    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load FAISS index and metadata from disk.

        Args:
            index_path: Path to saved .faiss index.
            metadata_path: Path to saved metadata JSON.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.embedding_dim = self.index.d
        logger.info(
            "Loaded FAISS index from %s (%d vectors, dim=%d)",
            index_path, self.index.ntotal, self.embedding_dim
        )
