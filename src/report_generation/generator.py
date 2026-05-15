import logging
from typing import List, Optional

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STRUCTURED_RAG_PROMPT = """You are a board-certified radiologist. Based on the chest \
X-ray image and the following similar cases retrieved from our database, generate a \
structured radiology report.

Retrieved similar reports for context:
{context}

Generate a structured report with:
1. FINDINGS: Describe all visible abnormalities or normal findings
2. IMPRESSION: Provide the clinical interpretation
3. ABNORMALITIES: List each abnormality as a bullet point

Be precise, avoid speculation, and only describe what is directly visible."""

CONCISE_PROMPT = """You are a radiologist. Provide a concise, one-paragraph radiology \
report for this chest X-ray. Focus on key findings and clinical impression only."""

DETAILED_PROMPT = """You are a senior radiologist. Provide a comprehensive radiology \
report for this chest X-ray covering: lung fields, cardiac silhouette, mediastinum, \
pleural spaces, osseous structures, and any incidental findings."""


class ReportGenerator:
    """Generates radiology reports using MedGemma, with optional RAG augmentation."""

    def __init__(
        self,
        medgemma,
        retriever=None,
        embedder=None,
    ) -> None:
        """
        Initialize the report generator.

        Args:
            medgemma: MedGemmaModel instance.
            retriever: FAISSRetriever instance (required for RAG mode).
            embedder: ColPaliEmbedder instance (required for RAG mode).
        """
        self.medgemma = medgemma
        self.retriever = retriever
        self.embedder = embedder

    def __enter__(self) -> "ReportGenerator":
        return self

    def __exit__(self, *args) -> None:
        self.medgemma.clear_cache()

    def generate_baseline(
        self, image: Image.Image, style: str = "concise"
    ) -> str:
        """
        Generate a report from image only, no retrieval context.

        Args:
            image: PIL chest X-ray image.
            style: 'concise', 'detailed', or 'structured'.

        Returns:
            Generated report string.
        """
        prompts = {
            "concise": CONCISE_PROMPT,
            "detailed": DETAILED_PROMPT,
            "structured": STRUCTURED_RAG_PROMPT.format(context="(no context provided)"),
        }
        prompt = prompts.get(style, CONCISE_PROMPT)
        logger.info("Generating baseline report (style='%s')", style)
        return self.medgemma.generate(image, prompt)

    def generate_with_rag(
        self,
        image: Image.Image,
        top_k: int = 3,
        style: str = "structured",
    ) -> dict:
        """
        Generate a report augmented with retrieved similar cases.

        Args:
            image: PIL chest X-ray image.
            top_k: Number of similar reports to retrieve as context.
            style: Prompt style for generation.

        Returns:
            Dict with keys: findings, impression, abnormalities,
            retrieved_reports, retrieved_image_paths.
        """
        assert self.retriever is not None and self.embedder is not None, \
            "retriever and embedder must be provided for RAG mode"

        query_emb = self.embedder.embed_images([image])[0]
        hits = self.retriever.retrieve(query_emb, top_k=top_k)

        retrieved_reports = [h.get("report", "") for h in hits]
        retrieved_image_paths = [h.get("image_path", "") for h in hits]

        prompt = self._build_rag_prompt(retrieved_reports, style)
        logger.info("Generating RAG report (top_k=%d, style='%s')", top_k, style)
        raw_output = self.medgemma.generate(image, prompt)

        findings = self._extract_section(raw_output, "FINDINGS")
        impression = self._extract_section(raw_output, "IMPRESSION")
        abnormalities = self._extract_abnormalities(raw_output)

        if not findings:
            findings = raw_output
        if not impression:
            impression = raw_output

        return {
            "findings": findings,
            "impression": impression,
            "abnormalities": abnormalities,
            "retrieved_reports": retrieved_reports,
            "retrieved_image_paths": retrieved_image_paths,
        }

    def _build_rag_prompt(self, context_reports: List[str], style: str) -> str:
        """
        Build a prompt string with retrieved report context inserted.

        Args:
            context_reports: List of retrieved radiology report strings.
            style: One of 'concise', 'detailed', 'structured'.

        Returns:
            Formatted prompt string.
        """
        context = "\n\n".join(
            f"Case {i+1}:\n{r}" for i, r in enumerate(context_reports)
        )
        if style == "structured":
            return STRUCTURED_RAG_PROMPT.format(context=context)
        if style == "detailed":
            return DETAILED_PROMPT + f"\n\nRetrieved similar cases for context:\n{context}"
        return CONCISE_PROMPT + f"\n\nRetrieved similar cases for context:\n{context}"

    def _extract_section(self, text: str, section: str) -> str:
        """Extract a named section from structured report output."""
        import re
        pattern = rf"{section}\s*[:.]?\s*(.*?)(?=\n[A-Z]{{3,}}|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_abnormalities(self, text: str) -> List[str]:
        """Extract bullet-point abnormalities from report text."""
        import re
        section = self._extract_section(text, "ABNORMALITIES")
        if not section:
            return []
        bullets = re.findall(r"[-•*]\s*(.+)", section)
        return [b.strip() for b in bullets if b.strip()]
