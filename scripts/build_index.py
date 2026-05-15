#!/usr/bin/env python3
"""Build and save the FAISS retrieval index from image embeddings.

Usage:
    python scripts/build_index.py --csv data/raw/dataset.csv \\
                                   --image-root data/raw/ \\
                                   --output-dir data/processed/
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.loader import MIMICCXRDataset
from src.retrieval.colpali_embedder import ColPaliEmbedder
from src.retrieval.faiss_index import FAISSRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_index(csv: str, image_root: str, output_dir: str, force_rebuild: bool = False) -> None:
    """
    Embed all images and build a FAISS index.

    Args:
        csv: Path to dataset CSV.
        image_root: Root directory for images.
        output_dir: Directory to save index, metadata, and embeddings.
        force_rebuild: If True, recompute even if cache files exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    emb_path = str(out / "image_embeddings.npy")
    index_path = str(out / "faiss.index")
    meta_path = str(out / "metadata.json")

    dataset = MIMICCXRDataset(csv, image_root, split="train")
    images = [dataset[i]["image"] for i in range(len(dataset))]
    metadata = [
        {"image_path": dataset[i]["image_path"], "report": dataset[i]["report"]}
        for i in range(len(dataset))
    ]

    with ColPaliEmbedder() as embedder:
        if Path(emb_path).exists() and not force_rebuild:
            logger.info("Embedding cache found at %s, skipping computation", emb_path)
            embeddings = embedder.load_embeddings(emb_path)
        else:
            logger.info("Embedding %d images...", len(images))
            embeddings = embedder.embed_images(images)
            embedder.save_embeddings(embeddings, emb_path)

    dim = embeddings.shape[1]
    retriever = FAISSRetriever(embedding_dim=dim)
    retriever.build_index(embeddings, metadata)
    retriever.save(index_path, meta_path)

    logger.info("Index built and saved to %s", output_dir)
    print(f"FAISS index : {index_path}")
    print(f"Metadata    : {meta_path}")
    print(f"Embeddings  : {emb_path}")


def main() -> None:
    """Entrypoint for index building script."""
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--image-root", required=True, help="Root directory for images")
    parser.add_argument("--output-dir", default="data/processed/", help="Output directory")
    parser.add_argument("--force-rebuild", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()
    build_index(args.csv, args.image_root, args.output_dir, args.force_rebuild)


if __name__ == "__main__":
    main()
