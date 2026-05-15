import json
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EvaluationSuite:
    """Computes NLP and retrieval metrics for report generation and VQA."""

    def __init__(self, results_dir: str = "results/") -> None:
        """
        Initialize evaluation suite.

        Args:
            results_dir: Directory where JSON results and plots will be saved.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def bleu(
        self, predictions: List[str], references: List[str]
    ) -> dict:
        """
        Compute BLEU-1, BLEU-2, and BLEU-4 scores.

        Args:
            predictions: List of generated report strings.
            references: List of ground-truth report strings.

        Returns:
            Dict with keys bleu1, bleu2, bleu4 (float values 0–1).
        """
        import evaluate
        bleu_metric = evaluate.load("bleu")
        refs_wrapped = [[r] for r in references]
        result1 = bleu_metric.compute(predictions=predictions, references=refs_wrapped, max_order=1)
        result2 = bleu_metric.compute(predictions=predictions, references=refs_wrapped, max_order=2)
        result4 = bleu_metric.compute(predictions=predictions, references=refs_wrapped, max_order=4)
        return {
            "bleu1": result1["bleu"],
            "bleu2": result2["bleu"],
            "bleu4": result4["bleu"],
        }

    def rouge(
        self, predictions: List[str], references: List[str]
    ) -> dict:
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

        Args:
            predictions: List of generated report strings.
            references: List of ground-truth report strings.

        Returns:
            Dict with keys rouge1, rouge2, rougeL (float values 0–1).
        """
        import evaluate
        rouge_metric = evaluate.load("rouge")
        result = rouge_metric.compute(predictions=predictions, references=references)
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }

    def bert_score(
        self, predictions: List[str], references: List[str]
    ) -> dict:
        """
        Compute BERTScore precision, recall, and F1.

        Args:
            predictions: List of generated report strings.
            references: List of ground-truth report strings.

        Returns:
            Dict with keys precision, recall, f1 (float values 0–1).
        """
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }

    def recall_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
        k: int = 5,
    ) -> float:
        """
        Compute Recall@K for retrieval evaluation.

        Args:
            retrieved_ids: List of retrieved ID lists (one per query).
            relevant_ids: List of ground-truth relevant ID lists.
            k: Cutoff rank.

        Returns:
            Mean Recall@K across all queries.
        """
        recalls = []
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            top_k = set(retrieved[:k])
            rel_set = set(relevant)
            if not rel_set:
                recalls.append(0.0)
            else:
                recalls.append(len(top_k & rel_set) / len(rel_set))
        return float(np.mean(recalls))

    def mean_similarity(self, similarity_scores: List[float]) -> float:
        """
        Compute the mean of a list of similarity scores.

        Args:
            similarity_scores: List of float similarity values.

        Returns:
            Mean similarity as a float.
        """
        return float(np.mean(similarity_scores)) if similarity_scores else 0.0

    def exact_match(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """
        Compute exact match accuracy (case-insensitive strip comparison).

        Args:
            predictions: List of predicted answer strings.
            references: List of ground-truth answer strings.

        Returns:
            Fraction of exact matches.
        """
        matches = sum(
            p.strip().lower() == r.strip().lower()
            for p, r in zip(predictions, references)
        )
        return matches / len(predictions) if predictions else 0.0

    def run_full_evaluation(
        self,
        report_preds: List[str],
        report_refs: List[str],
        vqa_preds: List[str],
        vqa_refs: List[str],
    ) -> dict:
        """
        Run all metrics and save results to results_dir.

        Args:
            report_preds: Predicted radiology reports.
            report_refs: Reference radiology reports.
            vqa_preds: Predicted VQA answers.
            vqa_refs: Ground-truth VQA answers.

        Returns:
            Dict of all computed metric values.
        """
        logger.info("Running full evaluation...")
        results = {}

        logger.info("Computing BLEU...")
        results["bleu"] = self.bleu(report_preds, report_refs)

        logger.info("Computing ROUGE...")
        results["rouge"] = self.rouge(report_preds, report_refs)

        logger.info("Computing BERTScore...")
        results["bert_score"] = self.bert_score(report_preds, report_refs)

        logger.info("Computing VQA exact match...")
        results["vqa_exact_match"] = self.exact_match(vqa_preds, vqa_refs)

        output_path = self.results_dir / "evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results saved to %s", output_path)

        return results

    def plot_comparison(
        self,
        baseline_scores: dict,
        rag_scores: dict,
        output_path: str = "results/comparison.png",
    ) -> None:
        """
        Bar chart comparing baseline vs RAG across key metrics.

        Args:
            baseline_scores: Dict of metric name → score for baseline.
            rag_scores: Dict of metric name → score for RAG.
            output_path: Path to save the PNG figure.
        """
        metrics = ["BLEU-4", "ROUGE-L", "BERTScore-F1"]
        baseline_vals = [
            baseline_scores.get("bleu", {}).get("bleu4", 0),
            baseline_scores.get("rouge", {}).get("rougeL", 0),
            baseline_scores.get("bert_score", {}).get("f1", 0),
        ]
        rag_vals = [
            rag_scores.get("bleu", {}).get("bleu4", 0),
            rag_scores.get("rouge", {}).get("rougeL", 0),
            rag_scores.get("bert_score", {}).get("f1", 0),
        ]

        x = np.arange(len(metrics))
        width = 0.35

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#4C72B0")
        ax.bar(x + width / 2, rag_vals, width, label="RAG", color="#DD8452")

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("Baseline vs RAG — Radiology Report Quality")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.0)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Comparison plot saved to %s", output_path)
