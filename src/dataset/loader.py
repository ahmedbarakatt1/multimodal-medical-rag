import logging
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MIMICCXRDataset(Dataset):
    """PyTorch Dataset for MIMIC-CXR chest X-ray images and radiology reports."""

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        split: str = "train",
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize dataset with CSV of image paths and reports.

        Args:
            csv_path: Path to CSV with columns image_path and text.
            image_root: Root directory prepended to image_path values.
            split: One of 'train', 'val', or 'test'.
            val_size: Fraction of data for validation.
            test_size: Fraction of data for test.
            random_state: Random seed for reproducibility.
            transform: Optional torchvision transform applied to PIL images.
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.image_root = Path(image_root)
        self.transform = transform

        df = pd.read_csv(csv_path)
        assert "image_path" in df.columns and "text" in df.columns, \
            "CSV must contain 'image_path' and 'text' columns"

        # First carve out test set
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )
        # Then split train_val into train and val
        adjusted_val = val_size / (1.0 - test_size)
        train, val = train_test_split(
            train_val, test_size=adjusted_val, random_state=random_state, shuffle=True
        )

        splits = {"train": train, "val": val, "test": test}
        self._split_sizes = {k: len(v) for k, v in splits.items()}
        self.data = splits[split].reset_index(drop=True)

        logger.info(
            "MIMICCXRDataset loaded — train: %d, val: %d, test: %d (using split='%s')",
            self._split_sizes["train"],
            self._split_sizes["val"],
            self._split_sizes["test"],
            split,
        )

    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one sample as a dict.

        Returns:
            dict with keys: image (PIL.Image), report (str), image_path (str).
        """
        row = self.data.iloc[idx]
        img_path = self.image_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "report": str(row["text"]),
            "image_path": str(row["image_path"]),
        }

    def get_split_stats(self) -> dict:
        """Return sample counts per split."""
        return dict(self._split_sizes)
