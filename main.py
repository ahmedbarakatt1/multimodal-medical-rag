#!/usr/bin/env python3
"""Main CLI entrypoint for the Medical Multimodal RAG project.

Usage:
    python main.py build-index   --csv data/raw/dataset.csv --image-root data/raw/
    python main.py generate-qa   --csv data/raw/dataset.csv --image-root data/raw/
    python main.py evaluate      --csv data/raw/dataset.csv --index-dir data/processed/ --n-samples 100
    python main.py demo          [--share] [--port 7860]
    python main.py generate      --image path/to/image.jpg --style structured [--rag]
    python main.py vqa           --image path/to/image.jpg --question "Is there cardiomegaly?"
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_build_index(args: argparse.Namespace) -> None:
    """Build and save the FAISS retrieval index."""
    from scripts.build_index import build_index
    build_index(args.csv, args.image_root, args.output_dir, args.force_rebuild)


def cmd_generate_qa(args: argparse.Namespace) -> None:
    """Generate QA dataset from radiology reports."""
    from scripts.generate_qa import generate_qa
    generate_qa(args.csv, args.image_root, args.output_path)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run full evaluation pipeline."""
    from scripts.run_evaluation import run_evaluation
    run_evaluation(args.csv, args.index_dir, args.n_samples, args.qa_path, args.results_dir)


def cmd_demo(args: argparse.Namespace) -> None:
    """Launch Gradio demo."""
    from scripts.run_demo import main as run_demo_main
    sys.argv = ["run_demo.py"]
    if args.share:
        sys.argv.append("--share")
    if args.port:
        sys.argv += ["--port", str(args.port)]
    if args.index_dir:
        sys.argv += ["--index-dir", args.index_dir]
    run_demo_main()


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a radiology report for a single image."""
    from PIL import Image
    from src.models.medgemma import MedGemmaModel
    from src.report_generation.generator import ReportGenerator

    medgemma = MedGemmaModel()
    retriever = None
    embedder = None

    if args.rag:
        from src.retrieval.colpali_embedder import ColPaliEmbedder
        from src.retrieval.faiss_index import FAISSRetriever
        idx_dir = Path(args.index_dir)
        embedder = ColPaliEmbedder()
        retriever = FAISSRetriever()
        retriever.load(str(idx_dir / "faiss.index"), str(idx_dir / "metadata.json"))

    gen = ReportGenerator(medgemma, retriever=retriever, embedder=embedder)
    image = Image.open(args.image).convert("RGB")

    if args.rag:
        result = gen.generate_with_rag(image, style=args.style)
        print("\n=== FINDINGS ===")
        print(result["findings"])
        print("\n=== IMPRESSION ===")
        print(result["impression"])
        print("\n=== ABNORMALITIES ===")
        for a in result["abnormalities"]:
            print(f"  • {a}")
    else:
        report = gen.generate_baseline(image, style=args.style)
        print(report)


def cmd_vqa(args: argparse.Namespace) -> None:
    """Answer a medical question about a chest X-ray."""
    from PIL import Image
    from src.models.medgemma import MedGemmaModel
    from src.vqa.pipeline import VQAPipeline

    medgemma = MedGemmaModel()
    retriever = None
    embedder = None

    if not args.no_rag:
        from src.retrieval.colpali_embedder import ColPaliEmbedder
        from src.retrieval.faiss_index import FAISSRetriever
        idx_dir = Path(args.index_dir)
        if (idx_dir / "faiss.index").exists():
            embedder = ColPaliEmbedder()
            retriever = FAISSRetriever()
            retriever.load(str(idx_dir / "faiss.index"), str(idx_dir / "metadata.json"))

    pipeline = VQAPipeline(medgemma, retriever=retriever, embedder=embedder)
    image = Image.open(args.image).convert("RGB")
    result = pipeline.answer(image, args.question, use_rag=(retriever is not None))

    print(f"\nQuestion type : {result['question_type']}")
    print(f"Answer        : {result['answer']}")


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        description="Medical Multimodal RAG — Chest X-Ray Analysis CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-index
    p_idx = sub.add_parser("build-index", help="Build FAISS retrieval index")
    p_idx.add_argument("--csv", required=True)
    p_idx.add_argument("--image-root", required=True)
    p_idx.add_argument("--output-dir", default="data/processed/")
    p_idx.add_argument("--force-rebuild", action="store_true")

    # generate-qa
    p_qa = sub.add_parser("generate-qa", help="Generate QA dataset")
    p_qa.add_argument("--csv", required=True)
    p_qa.add_argument("--image-root", required=True)
    p_qa.add_argument("--output-path", default="data/qa/qa_dataset.json")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run evaluation pipeline")
    p_eval.add_argument("--csv", required=True)
    p_eval.add_argument("--index-dir", default="data/processed/")
    p_eval.add_argument("--n-samples", type=int, default=100)
    p_eval.add_argument("--qa-path", default="data/qa/qa_dataset.json")
    p_eval.add_argument("--results-dir", default="results/")

    # demo
    p_demo = sub.add_parser("demo", help="Launch Gradio demo")
    p_demo.add_argument("--share", action="store_true")
    p_demo.add_argument("--port", type=int, default=7860)
    p_demo.add_argument("--index-dir", default="data/processed/")

    # generate
    p_gen = sub.add_parser("generate", help="Generate report for a single image")
    p_gen.add_argument("--image", required=True)
    p_gen.add_argument("--style", default="structured",
                       choices=["concise", "detailed", "structured"])
    p_gen.add_argument("--rag", action="store_true")
    p_gen.add_argument("--index-dir", default="data/processed/")

    # vqa
    p_vqa = sub.add_parser("vqa", help="Answer a question about a chest X-ray")
    p_vqa.add_argument("--image", required=True)
    p_vqa.add_argument("--question", required=True)
    p_vqa.add_argument("--no-rag", action="store_true")
    p_vqa.add_argument("--index-dir", default="data/processed/")

    args = parser.parse_args()

    dispatch = {
        "build-index": cmd_build_index,
        "generate-qa": cmd_generate_qa,
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
        "generate": cmd_generate,
        "vqa": cmd_vqa,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
