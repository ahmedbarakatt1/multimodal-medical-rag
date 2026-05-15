import logging
from typing import List, Optional

from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_QUESTION_TYPE_KEYWORDS = {
    "binary": ["is there", "are there", "does this", "do you see", "can you see",
               "is it", "is this"],
    "severity": ["how severe", "what severity", "mild", "moderate", "severe",
                 "how bad", "extent"],
    "clinical": ["clinical", "implication", "treatment", "management", "recommend",
                 "significance", "prognosis"],
    "comparison": ["compared", "difference", "normal", "abnormal", "versus"],
    "descriptive": ["describe", "what", "explain", "findings", "visible", "show"],
}

_VQA_PROMPT_TEMPLATES = {
    "binary": (
        "You are a radiologist. Answer with 'Yes' or 'No' and a brief justification.\n"
        "Question: {question}\n"
        "{context_block}"
        "Answer:"
    ),
    "severity": (
        "You are a radiologist. Classify severity as mild, moderate, or severe and explain.\n"
        "Question: {question}\n"
        "{context_block}"
        "Answer:"
    ),
    "clinical": (
        "You are a radiologist. Describe the clinical implications concisely.\n"
        "Question: {question}\n"
        "{context_block}"
        "Answer:"
    ),
    "comparison": (
        "You are a radiologist. Compare findings to a normal chest X-ray.\n"
        "Question: {question}\n"
        "{context_block}"
        "Answer:"
    ),
    "descriptive": (
        "You are a radiologist. Provide a detailed description of the findings.\n"
        "Question: {question}\n"
        "{context_block}"
        "Answer:"
    ),
}


class VQAPipeline:
    """Visual Question Answering pipeline over chest X-ray images."""

    def __init__(
        self,
        medgemma,
        retriever=None,
        embedder=None,
    ) -> None:
        """
        Initialize the VQA pipeline.

        Args:
            medgemma: MedGemmaModel instance.
            retriever: FAISSRetriever instance (used when use_rag=True).
            embedder: ColPaliEmbedder instance (used when use_rag=True).
        """
        self.medgemma = medgemma
        self.retriever = retriever
        self.embedder = embedder

    def __enter__(self) -> "VQAPipeline":
        return self

    def __exit__(self, *args) -> None:
        self.medgemma.clear_cache()

    def answer(
        self,
        image: Image.Image,
        question: str,
        use_rag: bool = True,
        top_k: int = 3,
    ) -> dict:
        """
        Answer a medical question about a chest X-ray image.

        Args:
            image: PIL chest X-ray image.
            question: Free-text clinical question.
            use_rag: Whether to retrieve similar cases for context.
            top_k: Number of similar cases to retrieve.

        Returns:
            Dict with keys: answer (str), question_type (str), context_used (list[str]).
        """
        question_type = self._classify_question(question)
        context_reports: List[str] = []

        if use_rag and self.retriever is not None and self.embedder is not None:
            query_emb = self.embedder.embed_images([image])[0]
            hits = self.retriever.retrieve(query_emb, top_k=top_k)
            context_reports = [h.get("report", "") for h in hits]

        prompt = self._build_vqa_prompt(question, question_type, context_reports)
        logger.info("VQA — type='%s', use_rag=%s", question_type, use_rag)
        answer_text = self.medgemma.generate(image, prompt)

        return {
            "answer": answer_text,
            "question_type": question_type,
            "context_used": context_reports,
        }

    def _classify_question(self, question: str) -> str:
        """
        Classify question type using keyword matching.

        Args:
            question: Free-text question string.

        Returns:
            One of: 'binary', 'severity', 'clinical', 'comparison', 'descriptive'.
        """
        q_lower = question.lower()
        scores = {qtype: 0 for qtype in _QUESTION_TYPE_KEYWORDS}
        for qtype, keywords in _QUESTION_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    scores[qtype] += 1
        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return "descriptive"
        return best

    def _build_vqa_prompt(
        self,
        question: str,
        question_type: str,
        context: List[str],
    ) -> str:
        """
        Build the final prompt string for VQA.

        Args:
            question: The clinical question.
            question_type: Classified question type.
            context: List of retrieved report strings.

        Returns:
            Formatted prompt string.
        """
        if context:
            context_block = "Retrieved similar cases:\n" + "\n".join(
                f"- {r[:300]}" for r in context
            ) + "\n\n"
        else:
            context_block = ""

        template = _VQA_PROMPT_TEMPLATES.get(question_type, _VQA_PROMPT_TEMPLATES["descriptive"])
        return template.format(question=question, context_block=context_block)

    def batch_answer(self, items: List[dict]) -> List[dict]:
        """
        Answer multiple image-question pairs.

        Args:
            items: List of dicts with keys 'image' (PIL.Image) and 'question' (str).

        Returns:
            List of answer dicts.
        """
        results = []
        for item in tqdm(items, desc="Batch VQA"):
            result = self.answer(
                image=item["image"],
                question=item["question"],
                use_rag=item.get("use_rag", True),
                top_k=item.get("top_k", 3),
            )
            results.append(result)
        return results
