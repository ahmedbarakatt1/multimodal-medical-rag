#!/usr/bin/env python3
"""Run full evaluation of baseline and RAG report generation + VQA.

Usage:
    python scripts/run_evaluation.py --csv data/raw/dataset.csv \\
                                      --index-dir data/processed/ \\
                                      --n-samples 100
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.loader import MIMICCXRDataset
from src.evaluation.metrics import EvaluationSuite
from src.models.medgemma import MedGemmaModel
from src.qa_generation.generator import QADatasetGenerator
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


def run_evaluation(
    csv: str,
    index_dir: str,
    n_samples: int,
    qa_path: str,
    results_dir: str,
) -> None:
    """
    Evaluate baseline and RAG pipelines and save all metrics.

    Args:
        csv: Path to dataset CSV.
        index_dir: Directory containing faiss.index and metadata.json.
        n_samples: Number of samples to evaluate.
        qa_path: Path to qa_dataset.json.
        results_dir: Directory to save evaluation results.
    """
    idx_dir = Path(index_dir)
    index_path = str(idx_dir / "faiss.index")
    meta_path = str(idx_dir / "metadata.json")
    emb_path = str(idx_dir / "image_embeddings.npy")

    dataset = MIMICCXRDataset(csv, str(Path(csv).parent), split="test")
    n_samples = min(n_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)
    samples = [dataset[i] for i in indices]

    logger.info("Loading models...")
    medgemma = MedGemmaModel()
    embedder = ColPaliEmbedder()
    retriever = FAISSRetriever()
    retriever.load(index_path, meta_path)

    report_gen = ReportGenerator(medgemma, retriever=retriever, embedder=embedder)
    vqa = VQAPipeline(medgemma, retriever=retriever, embedder=embedder)
    evaluator = EvaluationSuite(results_dir=results_dir)

    images = [s["image"] for s in samples]
    references = [s["report"] for s in samples]

    logger.info("Generating baseline reports...")
    baseline_preds = [report_gen.generate_baseline(img) for img in images]

    logger.info("Generating RAG reports...")
    rag_results = [report_gen.generate_with_rag(img) for img in images]
    rag_preds = [
        f"{r['findings']} {r['impression']}" for r in rag_results
    ]

    # VQA evaluation
    vqa_pairs = []
    vqa_refs = []
    if Path(qa_path).exists():
        qa_data = QADatasetGenerator().load(qa_path)
        sample_qa = random.sample(qa_data, min(n_samples, len(qa_data)))
        for qa in sample_qa:
            from PIL import Image
            try:
                img = Image.open(qa["image_path"]).convert("RGB")
                vqa_pairs.append({"image": img, "question": qa["question"]})
                vqa_refs.append(qa["answer"])
            except Exception:
                pass
    vqa_preds = [r["answer"] for r in vqa.batch_answer(vqa_pairs)] if vqa_pairs else []

    logger.info("Computing metrics...")
    baseline_scores = evaluator.run_full_evaluation(
        baseline_preds, references,
        vqa_preds if vqa_preds else [""], vqa_refs if vqa_refs else [""],
    )
    rag_scores = evaluator.run_full_evaluation(
        rag_preds, references,
        vqa_preds if vqa_preds else [""], vqa_refs if vqa_refs else [""],
    )

    evaluator.plot_comparison(baseline_scores, rag_scores)
    logger.info("Evaluation complete. Results saved to %s", results_dir)


def main() -> None:
    """Entrypoint for evaluation script."""
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--index-dir", default="data/processed/", help="FAISS index directory")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--qa-path", default="data/qa/qa_dataset.json", help="QA dataset path")
    parser.add_argument("--results-dir", default="results/", help="Results output directory")
    args = parser.parse_args()
    run_evaluation(args.csv, args.index_dir, args.n_samples, args.qa_path, args.results_dir)


if __name__ == "__main__":
    main()
