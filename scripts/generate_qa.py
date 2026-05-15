#!/usr/bin/env python3
"""Generate QA dataset from MIMIC-CXR reports.

Usage:
    python scripts/generate_qa.py --csv data/raw/dataset.csv \\
                                    --image-root data/raw/
"""
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.loader import MIMICCXRDataset
from src.qa_generation.generator import QADatasetGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_qa(csv: str, image_root: str, output_path: str) -> None:
    """
    Generate QA pairs from the full training split and save to JSON.

    Args:
        csv: Path to dataset CSV.
        image_root: Root directory for images.
        output_path: Output path for qa_dataset.json.
    """
    dataset = MIMICCXRDataset(csv, image_root, split="train")
    generator = QADatasetGenerator(output_path=output_path)
    qa_pairs = generator.generate_from_dataset(dataset)
    generator.save(qa_pairs)

    # Print per-condition and per-type counts
    condition_counts = Counter(qa["condition"] for qa in qa_pairs)
    type_counts = Counter(qa["question_type"] for qa in qa_pairs)

    print(f"\nGenerated {len(qa_pairs)} QA pairs → {output_path}")
    print("\nPer-condition counts:")
    for condition, count in sorted(condition_counts.items()):
        print(f"  {condition:<25} {count}")
    print("\nPer-type counts:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype:<20} {count}")


def main() -> None:
    """Entrypoint for QA generation script."""
    parser = argparse.ArgumentParser(description="Generate QA dataset from radiology reports")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--image-root", required=True, help="Root directory for images")
    parser.add_argument(
        "--output-path", default="data/qa/qa_dataset.json", help="Output JSON path"
    )
    args = parser.parse_args()
    generate_qa(args.csv, args.image_root, args.output_path)


if __name__ == "__main__":
    main()
