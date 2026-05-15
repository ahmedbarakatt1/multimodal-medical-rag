import gc
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ColPaliEmbedder:
    """Embeds chest X-ray images and text queries using ColPali."""

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = "auto",
        cache_dir: str = ".cache/colpali",
    ) -> None:
        """
        Load ColPali model and processor.

        Args:
            model_name: HuggingFace model ID.
            device: 'auto', 'cuda', or 'cpu'.
            cache_dir: Directory for caching model weights.
        """
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading ColPali model '%s' on device '%s'", model_name, self.device)
        self.processor = ColPaliProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        ).to(self.device).eval()
        logger.info("ColPali model loaded.")

    def __enter__(self) -> "ColPaliEmbedder":
        return self

    def __exit__(self, *args) -> None:
        self.clear_cache()

    def clear_cache(self) -> None:
        """Free GPU memory."""
        torch.cuda.empty_cache()
        gc.collect()

    def _l2_normalize(self, arr: np.ndarray) -> np.ndarray:
        """L2-normalize rows of a 2D array."""
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        return arr / norms

    def embed_images(
        self, images: List[Image.Image], batch_size: int = 8
    ) -> np.ndarray:
        """
        Embed a list of PIL images into L2-normalized vectors.

        Args:
            images: List of PIL images.
            batch_size: Images processed per forward pass.

        Returns:
            np.ndarray of shape (N, embedding_dim), L2-normalized.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch = images[i: i + batch_size]
            inputs = self.processor.process_images(batch).to(self.device)
            with torch.inference_mode(), torch.cuda.amp.autocast():
                embeddings = self.model(**inputs)
            # ColPali returns per-patch embeddings; mean-pool across patches
            emb = embeddings.mean(dim=1).float().cpu().numpy()
            all_embeddings.append(emb)
            self.clear_cache()

        result = np.vstack(all_embeddings)
        return self._l2_normalize(result)

    def embed_texts(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed a list of text strings into L2-normalized vectors.

        Args:
            texts: List of query strings.
            batch_size: Texts processed per forward pass.

        Returns:
            np.ndarray of shape (N, embedding_dim), L2-normalized.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i: i + batch_size]
            inputs = self.processor.process_queries(batch).to(self.device)
            with torch.inference_mode(), torch.cuda.amp.autocast():
                embeddings = self.model(**inputs)
            emb = embeddings.mean(dim=1).float().cpu().numpy()
            all_embeddings.append(emb)
            self.clear_cache()

        result = np.vstack(all_embeddings)
        return self._l2_normalize(result)

    def save_embeddings(self, embeddings: np.ndarray, path: str) -> None:
        """
        Save embeddings to a .npy file.

        Args:
            embeddings: Array to save.
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        logger.info("Saved embeddings to %s (shape=%s)", path, embeddings.shape)

    def load_embeddings(self, path: str) -> np.ndarray:
        """
        Load embeddings from a .npy file.

        Args:
            path: Path to .npy file.

        Returns:
            np.ndarray of embeddings.
        """
        embeddings = np.load(path)
        logger.info("Loaded embeddings from %s (shape=%s)", path, embeddings.shape)
        return embeddings
