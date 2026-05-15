#!/usr/bin/env python3
"""Launch the Gradio demo with full model stack.

Usage:
    python scripts/run_demo.py [--share] [--port 7860]
    python scripts/run_demo.py --index-dir data/processed/ --share
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.gradio_app import build_demo
from src.models.medgemma import MedGemmaModel
from src.report_generation.generator import ReportGenerator
from src.retrieval.colpali_embedder import ColPaliEmbedder
from src.retrieval.faiss_index import FAISSRetriever
from src.vqa.pipeline import VQAPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load all models and launch the Gradio interface."""
    parser = argparse.ArgumentParser(description="Launch Medical Multimodal RAG Gradio demo")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Local port")
    parser.add_argument("--index-dir", default="data/processed/", help="FAISS index directory")
    parser.add_argument("--no-rag", action="store_true", help="Launch without RAG (models only)")
    args = parser.parse_args()

    logger.info("Loading MedGemma model...")
    medgemma = MedGemmaModel()

    retriever = None
    embedder = None

    if not args.no_rag:
        idx_dir = Path(args.index_dir)
        index_path = str(idx_dir / "faiss.index")
        meta_path = str(idx_dir / "metadata.json")

        if idx_dir.exists() and Path(index_path).exists():
            logger.info("Loading ColPali embedder...")
            embedder = ColPaliEmbedder()
            logger.info("Loading FAISS retriever from %s...", args.index_dir)
            retriever = FAISSRetriever()
            retriever.load(index_path, meta_path)
        else:
            logger.warning(
                "FAISS index not found at %s. Running without RAG. "
                "Run build_index.py first to enable retrieval.", index_path
            )

    report_generator = ReportGenerator(medgemma, retriever=retriever, embedder=embedder)
    vqa_pipeline = VQAPipeline(medgemma, retriever=retriever, embedder=embedder)

    demo = build_demo(report_generator=report_generator, vqa_pipeline=vqa_pipeline)
    logger.info("Launching Gradio demo on port %d (share=%s)...", args.port, args.share)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
