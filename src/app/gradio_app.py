import logging

import gradio as gr
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _gpu_memory_status() -> str:
    """Return current GPU memory usage as a formatted string."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return (
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"Allocated: {allocated:.2f} GB / Reserved: {reserved:.2f} GB / "
            f"Total: {total:.2f} GB"
        )
    return "GPU: Not available (running on CPU)"


def build_demo(report_generator=None, vqa_pipeline=None) -> gr.Blocks:
    """
    Build and return a two-tab Gradio Blocks demo.

    Args:
        report_generator: ReportGenerator instance (optional; None shows placeholder).
        vqa_pipeline: VQAPipeline instance (optional; None shows placeholder).

    Returns:
        gr.Blocks instance ready for .launch().
    """

    def generate_report(image, style, use_rag):
        if image is None:
            return "Please upload a chest X-ray image.", "", _gpu_memory_status()
        try:
            if report_generator is None:
                return ("(Demo mode) Report generator not loaded. "
                        "Load models via run_demo.py to generate real reports."), "", _gpu_memory_status()
            if use_rag:
                result = report_generator.generate_with_rag(image, style=style)
                report_text = (
                    f"FINDINGS:\n{result['findings']}\n\n"
                    f"IMPRESSION:\n{result['impression']}\n\n"
                    f"ABNORMALITIES:\n" + "\n".join(f"• {a}" for a in result["abnormalities"])
                )
                context_text = "\n\n---\n\n".join(result["retrieved_reports"])
            else:
                report_text = report_generator.generate_baseline(image, style=style)
                context_text = ""
            return report_text, context_text, _gpu_memory_status()
        except Exception as e:
            logger.error("Report generation error: %s", e)
            return f"Error: {e}", "", _gpu_memory_status()

    def answer_question(image, question, use_rag):
        if image is None or not question.strip():
            return "Please upload an image and enter a question.", "", _gpu_memory_status()
        try:
            if vqa_pipeline is None:
                return ("(Demo mode) VQA pipeline not loaded. "
                        "Load models via run_demo.py to get real answers."), "unknown", _gpu_memory_status()
            result = vqa_pipeline.answer(image, question, use_rag=use_rag)
            return result["answer"], result["question_type"], _gpu_memory_status()
        except Exception as e:
            logger.error("VQA error: %s", e)
            return f"Error: {e}", "unknown", _gpu_memory_status()

    with gr.Blocks(title="Medical Multimodal RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🩺 Medical Multimodal RAG — Chest X-Ray Analysis
            **Models**: ColPali (vidore/colpali-v1.2) + MedGemma (google/medgemma-4b-it)
            **Pipeline**: Retrieval-Augmented Generation for radiology report synthesis and VQA
            """
        )

        gpu_status = gr.Textbox(
            label="GPU Status",
            value=_gpu_memory_status(),
            interactive=False,
            lines=1,
        )

        with gr.Tabs():
            # ── Tab 1: Report Generation ──────────────────────────────────────
            with gr.Tab("Report Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        report_image = gr.Image(type="pil", label="Upload Chest X-Ray")
                    with gr.Column(scale=1):
                        report_style = gr.Radio(
                            choices=["concise", "detailed", "structured"],
                            value="structured",
                            label="Report Style",
                        )
                        report_use_rag = gr.Checkbox(
                            value=True, label="Use RAG Retrieval"
                        )
                        report_btn = gr.Button("Generate Report", variant="primary")
                        report_output = gr.Textbox(
                            label="Generated Report", lines=12, interactive=False
                        )

                with gr.Accordion("Retrieved Context Reports", open=False):
                    context_output = gr.Textbox(
                        label="Retrieved Similar Reports",
                        lines=8,
                        interactive=False,
                    )

                report_btn.click(
                    fn=generate_report,
                    inputs=[report_image, report_style, report_use_rag],
                    outputs=[report_output, context_output, gpu_status],
                )

            # ── Tab 2: Medical VQA ────────────────────────────────────────────
            with gr.Tab("Medical VQA"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vqa_image = gr.Image(type="pil", label="Upload Chest X-Ray")
                    with gr.Column(scale=1):
                        vqa_question = gr.Textbox(
                            label="Enter your question",
                            placeholder="e.g. Is there evidence of cardiomegaly?",
                            lines=3,
                        )
                        vqa_use_rag = gr.Checkbox(value=True, label="Use RAG Retrieval")
                        vqa_btn = gr.Button("Ask Question", variant="primary")
                        vqa_answer = gr.Textbox(
                            label="Answer", lines=6, interactive=False
                        )
                        vqa_type = gr.Label(label="Detected Question Type")

                vqa_btn.click(
                    fn=answer_question,
                    inputs=[vqa_image, vqa_question, vqa_use_rag],
                    outputs=[vqa_answer, vqa_type, gpu_status],
                )

        gr.Markdown(
            """
            ---
            ⚠️ **Disclaimer**: For research purposes only. Not for clinical use.
            This tool is not a substitute for professional medical diagnosis.
            """
        )

    return demo
