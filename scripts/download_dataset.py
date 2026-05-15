#!/usr/bin/env python3
"""Download the MIMIC-CXR dataset from Kaggle.

Usage:
    python scripts/download_dataset.py --output-dir data/raw
"""
import argparse
import logging
import os
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_dataset(output_dir: str) -> None:
    """
    Authenticate with Kaggle and download MIMIC-CXR dataset.

    Args:
        output_dir: Directory where the dataset will be extracted.
    """
    import kaggle

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_id = "simhadrisadaram/mimic-cxr-dataset"
    logger.info("Downloading Kaggle dataset: %s → %s", dataset_id, out)

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_id, path=str(out), unzip=False)

    zip_files = list(out.glob("*.zip"))
    for zf in zip_files:
        logger.info("Extracting %s...", zf)
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(out)
        zf.unlink()

    all_files = list(out.rglob("*"))
    image_files = [f for f in all_files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    total_size_mb = sum(f.stat().st_size for f in all_files if f.is_file()) / 1e6

    logger.info("Download complete.")
    logger.info("Total files: %d | Images: %d | Size: %.1f MB",
                len(all_files), len(image_files), total_size_mb)
    print(f"\nDataset extracted to: {out.resolve()}")
    print(f"Total files : {len(all_files)}")
    print(f"Image files : {len(image_files)}")
    print(f"Total size  : {total_size_mb:.1f} MB")


def main() -> None:
    """Entrypoint for dataset download script."""
    parser = argparse.ArgumentParser(description="Download MIMIC-CXR dataset from Kaggle")
    parser.add_argument(
        "--output-dir", default="data/raw", help="Directory to extract dataset into"
    )
    args = parser.parse_args()
    download_dataset(args.output_dir)


if __name__ == "__main__":
    main()
