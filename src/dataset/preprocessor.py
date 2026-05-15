import re
import logging
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Common boilerplate headers found in radiology reports
_BOILERPLATE_PATTERNS = [
    r"(?i)^radiology report\s*[:.]?\s*",
    r"(?i)^report\s*[:.]?\s*",
    r"(?i)^final report\s*[:.]?\s*",
    r"(?i)^clinical indication\s*[:.]?\s*.*",
    r"(?i)^technique\s*[:.]?\s*.*",
    r"(?i)^comparison\s*[:.]?\s*.*",
]


class ImagePreprocessor:
    """Preprocesses chest X-ray images for model input."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initialize with target resize dimensions.

        Args:
            target_size: (height, width) to resize images to.
        """
        self.target_size = target_size

    def get_transform(self) -> transforms.Compose:
        """Return a torchvision transform pipeline for preprocessing."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess a list of PIL images into a batched tensor.

        Args:
            images: List of PIL images.

        Returns:
            Tensor of shape (N, 3, H, W).
        """
        transform = self.get_transform()
        tensors = [transform(img) for img in images]
        return torch.stack(tensors)


class ReportCleaner:
    """Cleans and parses radiology report text."""

    def clean(self, report: str) -> str:
        """
        Lowercase, strip boilerplate headers, and collapse whitespace.

        Args:
            report: Raw radiology report string.

        Returns:
            Cleaned report string.
        """
        text = report.lower()
        for pattern in _BOILERPLATE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_sections(self, report: str) -> dict:
        """
        Extract findings and impression sections from a report.

        Args:
            report: Raw or cleaned report string.

        Returns:
            dict with keys 'findings' and 'impression' (empty string if not found).
        """
        text = report.lower()

        findings_match = re.search(
            r"findings\s*[:.]?\s*(.*?)(?=impression|$)", text, re.DOTALL
        )
        impression_match = re.search(
            r"impression\s*[:.]?\s*(.*?)$", text, re.DOTALL
        )

        findings = findings_match.group(1).strip() if findings_match else ""
        impression = impression_match.group(1).strip() if impression_match else ""

        if not findings and not impression:
            findings = self.clean(report)

        return {"findings": findings, "impression": impression}

    def batch_clean(self, reports: List[str]) -> List[str]:
        """
        Clean a list of reports.

        Args:
            reports: List of raw report strings.

        Returns:
            List of cleaned report strings.
        """
        return [self.clean(r) for r in reports]
