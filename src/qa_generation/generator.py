import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_CONDITIONS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Lung Opacity",
    "No Finding",
    "Support Devices",
]

QUESTION_TEMPLATES = {
    "binary": "Is there evidence of {condition} in this chest X-ray?",
    "descriptive": "Describe the {condition} findings visible in this X-ray.",
    "severity": "How severe is the {condition} shown in this image?",
    "comparison": "Compared to a normal chest X-ray, what abnormalities are present?",
    "clinical": "What are the clinical implications of the findings in this X-ray?",
}

# Keywords per condition used to extract answers from report text
_CONDITION_KEYWORDS = {
    "Atelectasis": ["atelectasis", "atelectatic", "collapse", "subsegmental"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement", "increased cardiac"],
    "Consolidation": ["consolidation", "consolidative", "airspace"],
    "Edema": ["edema", "oedema", "pulmonary edema", "interstitial"],
    "Pleural Effusion": ["effusion", "pleural fluid", "pleural effusion"],
    "Pneumonia": ["pneumonia", "pneumonic", "infection", "infiltrate"],
    "Pneumothorax": ["pneumothorax", "pneumothoraces", "air in pleural"],
    "Lung Opacity": ["opacity", "opacification", "haziness"],
    "No Finding": ["no acute", "normal", "unremarkable", "no findings"],
    "Support Devices": ["tube", "line", "catheter", "pacemaker", "device", "support"],
}


class QADatasetGenerator:
    """Generates question-answer pairs from radiology reports for VQA evaluation."""

    def __init__(self, output_path: str = "data/qa/qa_dataset.json") -> None:
        """
        Initialize the generator.

        Args:
            output_path: Path where qa_dataset.json will be saved.
        """
        self.output_path = output_path

    def generate_from_dataset(
        self,
        dataset,
        conditions: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Generate QA pairs from a MIMICCXRDataset instance.

        Args:
            dataset: MIMICCXRDataset instance.
            conditions: Subset of conditions to generate for; None = all 10.

        Returns:
            List of QA dicts conforming to the output schema.
        """
        if conditions is None:
            conditions = SUPPORTED_CONDITIONS

        qa_pairs: List[dict] = []
        for idx in tqdm(range(len(dataset)), desc="Generating QA pairs"):
            sample = dataset[idx]
            report = sample["report"]
            image_path = sample["image_path"]

            for condition in conditions:
                for q_type, template in QUESTION_TEMPLATES.items():
                    question = template.format(condition=condition)
                    answer = self._generate_answer_from_report(report, condition, q_type)
                    qa_pairs.append({
                        "id": str(uuid.uuid4()),
                        "image_path": image_path,
                        "question": question,
                        "answer": answer,
                        "question_type": q_type,
                        "condition": condition,
                        "report": report,
                    })

        logger.info("Generated %d QA pairs from %d samples.", len(qa_pairs), len(dataset))
        return qa_pairs

    def _generate_answer_from_report(
        self, report: str, condition: str, question_type: str
    ) -> str:
        """
        Extract or construct an answer from report text using regex/heuristics.

        Args:
            report: Radiology report text.
            condition: Medical condition being queried.
            question_type: One of the five question types.

        Returns:
            Answer string derived from the report.
        """
        report_lower = report.lower()
        keywords = _CONDITION_KEYWORDS.get(condition, [condition.lower()])
        found = any(kw in report_lower for kw in keywords)

        if question_type == "binary":
            if found:
                return f"Yes, there is evidence of {condition} in this chest X-ray."
            return f"No, there is no evidence of {condition} in this chest X-ray."

        if question_type == "comparison":
            sentences = [s.strip() for s in re.split(r"[.!?]", report) if s.strip()]
            abnormal = [s for s in sentences if any(
                kw in s.lower() for kw in ["opacity", "effusion", "consolidation",
                                            "cardiomegaly", "atelectasis", "edema"]
            )]
            if abnormal:
                return "Compared to a normal chest X-ray, the following abnormalities are present: " + ". ".join(abnormal[:2]) + "."
            return "Compared to a normal chest X-ray, no significant abnormalities are identified."

        if question_type == "clinical":
            sentences = [s.strip() for s in re.split(r"[.!?]", report) if s.strip()]
            impression = next(
                (s for s in sentences if "impression" in s.lower()), None
            )
            if impression:
                return f"The clinical implications are: {impression}."
            return "Clinical correlation is recommended based on the radiological findings."

        # For descriptive and severity: extract relevant sentences
        sentences = [s.strip() for s in re.split(r"[.!?]", report) if s.strip()]
        relevant = [s for s in sentences if any(kw in s.lower() for kw in keywords)]

        if question_type == "descriptive":
            if relevant:
                return " ".join(relevant[:2]) + "."
            return f"No specific {condition} findings are described in this report."

        if question_type == "severity":
            severity_words = ["mild", "moderate", "severe", "large", "small", "bilateral", "unilateral"]
            for sent in relevant:
                for sev in severity_words:
                    if sev in sent.lower():
                        return f"The {condition} appears {sev} based on the radiological findings: {sent}."
            if relevant:
                return f"The extent of {condition} is noted: {relevant[0]}."
            return f"Severity of {condition} cannot be determined from the available report."

        return "No relevant information found in the report."

    def save(self, qa_pairs: List[dict]) -> None:
        """
        Save QA pairs to output_path as JSON.

        Args:
            qa_pairs: List of QA dicts.
        """
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d QA pairs to %s", len(qa_pairs), self.output_path)

    def load(self, path: str) -> List[dict]:
        """
        Load QA pairs from a JSON file.

        Args:
            path: Path to a qa_dataset.json file.

        Returns:
            List of QA dicts.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded %d QA pairs from %s", len(data), path)
        return data
