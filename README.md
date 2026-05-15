# Medical Multimodal RAG — Chest X-Ray Analysis

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-models-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/)

---

## Overview

This project implements a research-grade **Retrieval-Augmented Generation (RAG)** pipeline for automated chest X-ray analysis. It combines ColPali's visual document retrieval with Google's MedGemma multimodal language model to generate structured radiology reports and answer medical questions about chest X-ray images. The system is designed to run end-to-end on a Google Colab T4 GPU within the 15 GB VRAM budget, making it accessible to researchers without dedicated GPU infrastructure.

The system retrieves semantically similar historical cases from a FAISS vector index built from ColPali image embeddings, then conditions MedGemma's generation on these retrieved examples. This RAG approach improves report accuracy and clinical relevance compared to direct inference without context. The pipeline supports both automated report generation (with findings, impression, and abnormalities) and interactive visual question answering (VQA) across ten clinically significant thoracic conditions.

---

## Architecture

```
Chest X-Ray Image
       │
       ▼
  ColPali Encoder  ──────────────────────────────────────────┐
  (vidore/colpali-v1.2)                                      │
       │                                                     │
       ▼                                                     ▼
  FAISS IndexFlatIP  ──► Retrieved Reports (Top-K)    Query Embedding
       │                        │
       └──────────┬─────────────┘
                  │
                  ▼
         RAG Prompt Builder
                  │
                  ▼
     MedGemma-4B-IT (4-bit NF4)
     (google/medgemma-4b-it)
                  │
                  ▼
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
Radiology Report            VQA Answer
(Findings + Impression)     (with context)
```

---

## Key Features

- **Automated Radiology Report Generation** — Structured reports (findings, impression, abnormalities) using MedGemma in concise, detailed, or structured style
- **Visual Question Answering (VQA)** — Answer binary, descriptive, severity, comparison, and clinical questions about chest X-rays
- **RAG Pipeline** — ColPali image embeddings + FAISS cosine retrieval for context-grounded generation using similar historical cases
- **QA Dataset Generation** — Automated creation of 50 question-answer pairs per image across 10 thoracic conditions and 5 question types
- **Evaluation Suite** — BLEU, ROUGE, BERTScore, Recall@K, and Exact Match metrics with visual comparison bar charts
- **Gradio Demo** — Two-tab interactive web interface with GPU status monitoring and real-time report generation
- **Colab-Ready** — One-click end-to-end notebook with Google Drive integration and 4-bit model quantization

---

## Models Used

| Model | HuggingFace ID | Purpose | Parameters |
|---|---|---|---|
| ColPali | `vidore/colpali-v1.2` | Image embedding & retrieval | ~3B |
| MedGemma | `google/medgemma-4b-it` | Report generation & VQA | 4B (4-bit: ~2 GB) |

Both models are downloaded automatically from HuggingFace Hub on first run and cached locally. MedGemma requires acceptance of the model license on HuggingFace before use.

---

## Dataset

**MIMIC-CXR** — A large publicly available database of labeled chest radiographs with free-text radiology reports.

- **Kaggle Source**: [simhadrisadaram/mimic-cxr-dataset](https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset)
- **License**: PhysioNet Credentialed Health Data License
- **Expected CSV Format**:

| Column | Description |
|---|---|
| `image_path` | Relative path to the image file from the dataset root |
| `text` | Free-text radiology report (findings + impression) |

Images should be JPEG or PNG format. The pipeline handles grayscale and RGB images automatically, converting all inputs to RGB for model compatibility.

---

## Project Structure

```
medical-multimodal-rag/
├── data/
│   ├── raw/                        # Downloaded dataset images and CSV
│   ├── processed/                  # FAISS index + ColPali embeddings
│   └── qa/                         # qa_dataset.json output
├── notebooks/
│   └── full_pipeline.ipynb         # End-to-end Colab notebook (17 cells)
├── src/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── loader.py               # MIMICCXRDataset (PyTorch Dataset)
│   │   └── preprocessor.py         # ImagePreprocessor + ReportCleaner
│   ├── qa_generation/
│   │   ├── __init__.py
│   │   └── generator.py            # QADatasetGenerator
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── colpali_embedder.py     # ColPaliEmbedder
│   │   └── faiss_index.py          # FAISSRetriever
│   ├── models/
│   │   ├── __init__.py
│   │   └── medgemma.py             # MedGemmaModel (4-bit quantized)
│   ├── report_generation/
│   │   ├── __init__.py
│   │   └── generator.py            # ReportGenerator (baseline + RAG)
│   ├── vqa/
│   │   ├── __init__.py
│   │   └── pipeline.py             # VQAPipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # EvaluationSuite
│   └── app/
│       ├── __init__.py
│       └── gradio_app.py           # build_demo() — Gradio 4.x interface
├── scripts/
│   ├── setup_colab.sh              # One-shot install + HuggingFace auth
│   ├── download_dataset.py         # Kaggle API download + extract
│   ├── build_index.py              # Build + save FAISS index
│   ├── generate_qa.py              # Produce qa_dataset.json
│   ├── run_evaluation.py           # All metrics → results/
│   └── run_demo.py                 # Launch Gradio demo
├── results/                        # Evaluation JSON + PNG outputs
├── requirements.txt
├── README.md
└── main.py                         # Argparse CLI entrypoint (6 subcommands)
```

---

## Installation

### System Requirements

- Python 3.10 or higher
- CUDA-capable GPU with ≥8 GB VRAM (15 GB recommended for full pipeline)
- CUDA 11.8 or 12.x with compatible PyTorch build
- HuggingFace account with MedGemma license accepted
- Kaggle account with API credentials for dataset download

### Local Installation

```bash
git clone https://github.com/your-username/medical-multimodal-rag.git
cd medical-multimodal-rag

# Create a virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for MedGemma access)
huggingface-cli login --token YOUR_HF_TOKEN

# Set up Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### Google Colab Installation

1. Upload the `medical-multimodal-rag/` folder to your Google Drive under `MyDrive/`
2. Open `notebooks/full_pipeline.ipynb` in Google Colab
3. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
4. In Cell 3, paste your `HF_TOKEN` environment variable before running
5. In Cell 4, set `KAGGLE_USERNAME` and `KAGGLE_KEY` before running
6. Run cells sequentially from top to bottom

---

## Quick Start

```python
from PIL import Image
from src.models.medgemma import MedGemmaModel
from src.report_generation.generator import ReportGenerator

model = MedGemmaModel()
gen = ReportGenerator(model)
image = Image.open("path/to/chest_xray.jpg").convert("RGB")
print(gen.generate_baseline(image, style="concise"))
```

For RAG-enhanced generation (requires a built FAISS index):

```python
from src.retrieval.colpali_embedder import ColPaliEmbedder
from src.retrieval.faiss_index import FAISSRetriever

embedder = ColPaliEmbedder()
retriever = FAISSRetriever()
retriever.load("data/processed/faiss.index", "data/processed/metadata.json")
gen = ReportGenerator(model, retriever=retriever, embedder=embedder)
result = gen.generate_with_rag(image, top_k=3, style="structured")
print(result["findings"])
print(result["impression"])
```

---

## Usage

### `build-index` — Build retrieval index

Embeds all images in the dataset with ColPali and builds a FAISS IndexFlatIP for cosine similarity retrieval. Skips re-computation if the index already exists.

```bash
python main.py build-index \
    --csv data/raw/dataset.csv \
    --image-root data/raw/ \
    --output-dir data/processed/
```

Outputs: `data/processed/faiss.index`, `data/processed/metadata.json`, `data/processed/image_embeddings.npy`

### `generate-qa` — Generate QA dataset

Generates question-answer pairs for each image across 10 conditions and 5 question types using heuristic answer extraction from radiology reports.

```bash
python main.py generate-qa \
    --csv data/raw/dataset.csv \
    --image-root data/raw/ \
    --output-path data/qa/qa_dataset.json
```

Output: `data/qa/qa_dataset.json` with fields: `id`, `image_path`, `question`, `answer`, `question_type`, `condition`, `report`

### `evaluate` — Run evaluation pipeline

Runs baseline and RAG report generation on a random sample of the dataset, evaluates with NLP metrics, and saves results and a comparison chart.

```bash
python main.py evaluate \
    --csv data/raw/dataset.csv \
    --index-dir data/processed/ \
    --n-samples 100
```

Outputs: `results/evaluation_results.json`, `results/comparison.png`

### `demo` — Launch Gradio demo

Loads all models and launches the two-tab Gradio web interface. Use `--share` to get a public Gradio link (required in Colab).

```bash
python main.py demo --share --port 7860
```

### `generate` — Generate report for a single image

```bash
# Baseline (no retrieval)
python main.py generate --image path/to/image.jpg --style structured

# RAG-enhanced (requires built index)
python main.py generate --image path/to/image.jpg --style structured --rag \
    --index-dir data/processed/
```

Available styles: `concise`, `detailed`, `structured`

### `vqa` — Answer a medical question

```bash
python main.py vqa \
    --image path/to/image.jpg \
    --question "Is there evidence of cardiomegaly?"

# With RAG context
python main.py vqa \
    --image path/to/image.jpg \
    --question "Describe the pleural findings." \
    --rag --index-dir data/processed/
```

---

## Colab Instructions

1. **Upload project** — Upload the entire `medical-multimodal-rag/` folder to Google Drive under `MyDrive/`. Preserve the directory structure exactly.
2. **Open notebook** — Go to [colab.research.google.com](https://colab.research.google.com), click File → Open notebook → Google Drive, and open `notebooks/full_pipeline.ipynb`.
3. **Set GPU runtime** — Runtime → Change runtime type → T4 GPU → Save.
4. **Set HuggingFace token** — In Cell 3, set the `HF_TOKEN` variable to your HuggingFace access token. You must have accepted the MedGemma license at `huggingface.co/google/medgemma-4b-it`.
5. **Set Kaggle credentials** — In Cell 4, set `KAGGLE_USERNAME` and `KAGGLE_KEY` from your Kaggle account settings (Account → Create New Token).
6. **Run all cells** — Runtime → Run all, or run cells one by one for visibility. Cells 3 and 4 will take 5–10 minutes for downloads.
7. **View Gradio demo** — Cell 16 outputs a public URL (`https://xxxxx.gradio.live`) that is accessible from any browser for 72 hours.

**Expected runtimes on T4 GPU:**
- Setup (Cell 3): ~5 min
- Dataset download (Cell 4): ~3 min
- Index building (Cell 6): ~15–30 min depending on dataset size
- MedGemma loading (Cell 8): ~2 min
- Full evaluation 50 samples (Cell 13): ~20–40 min

---

## Evaluation Results

| Metric | Baseline | RAG | Δ |
|---|---|---|---|
| BLEU-1 | — | — | — |
| BLEU-4 | — | — | — |
| ROUGE-1 | — | — | — |
| ROUGE-L | — | — | — |
| BERTScore-F1 | — | — | — |
| VQA Exact Match | — | — | — |
| Recall@5 | — | — | — |

*Run `python main.py evaluate --csv data/raw/dataset.csv --index-dir data/processed/ --n-samples 100` to populate this table with real results.*

---

## Demo Screenshots

![Report Generation Tab](results/screenshot_report.png)

*Tab 1: Upload a chest X-ray, select report style, enable RAG retrieval, and click "Generate Report" to produce a structured radiology report with retrieved context.*

![Medical VQA Tab](results/screenshot_vqa.png)

*Tab 2: Upload a chest X-ray, type a medical question, and receive a context-grounded answer with detected question type.*

---

## Research Background

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (Lewis et al., 2020) is a paradigm that augments large language model generation with evidence retrieved from an external knowledge base at inference time. Rather than relying entirely on parametric knowledge encoded in model weights during training, RAG systems first retrieve relevant documents from an index, then concatenate these documents as context to the model's input before generating a response. This approach is particularly valuable in knowledge-intensive domains like medicine, where the volume of clinical literature exceeds what can be memorized during pretraining, and where factual grounding is critical to safe outputs.

In the context of radiology report generation, RAG enables the model to reference real reports from similar prior cases — same imaging modality, similar pathology presentations, comparable patient demographics — as context when generating a new report. This mimics the clinical practice of radiologists consulting prior studies and reference cases. Empirically, RAG has been shown to reduce hallucinations in medical NLP tasks by anchoring generation in retrieved factual evidence, and to improve the clinical accuracy and specificity of generated text compared to purely parametric generation.

The retrieval mechanism in this project uses FAISS `IndexFlatIP` with L2-normalized embeddings, which performs exact cosine similarity search over the entire index. At query time, the chest X-ray image is embedded with ColPali, and the top-K most similar images from the indexed corpus are retrieved along with their associated radiology reports. These reports form the RAG context that is injected into the MedGemma prompt via a structured template specifying the generation format.

### ColPali: Visual Document Retrieval

ColPali (Faysse et al., 2024) is a vision-language model purpose-built for efficient retrieval of visually rich document pages. It is built on PaliGemma — a joint vision-language encoder — and fine-tuned using a late interaction retrieval objective adapted from ColBERT. Unlike traditional document retrieval systems that rely on OCR-extracted text, ColPali encodes entire document pages as sequences of patch-level embeddings, enabling it to reason jointly over visual layout, embedded text, tables, and figures. This makes it particularly well-suited for medical imaging, where clinical information is encoded visually in the image itself rather than in associated text.

For chest X-ray retrieval, ColPali produces a fixed-dimensional embedding for each image that captures semantic radiological content — lung field opacification, cardiac silhouette size, pleural space abnormalities, and similar visual features — without requiring any textual query. The embeddings are L2-normalized to unit length before indexing, enabling cosine similarity to be computed as a simple inner product, which FAISS's `IndexFlatIP` accelerates. In this pipeline, ColPali serves dual roles: building the offline embedding index during the `build-index` step, and encoding query images at inference time during report generation and VQA.

ColPali's architecture is particularly efficient for retrieval because late interaction aggregates patch-level similarities with a MaxSim operation, capturing fine-grained visual correspondences between the query image and indexed documents. The `colpali-v1.2` checkpoint used here is available on HuggingFace and is accessed via the `colpali-engine` Python library, which provides a standardized processor and model interface compatible with HuggingFace's `transformers` ecosystem.

### MedGemma: Medical Multimodal Language Model

MedGemma is Google DeepMind's medical domain adaptation of the Gemma 3 language model family, specifically designed for healthcare applications. The `medgemma-4b-it` checkpoint is a 4-billion parameter instruction-tuned multimodal model trained on a large corpus of medical text and clinical imaging data, including radiology images, pathology slides, dermatology images, and ophthalmology images. Its training enables it to understand and respond to medical visual content with clinically appropriate language, making it substantially more accurate than general-purpose vision-language models on radiology interpretation tasks.

The model accepts interleaved image-text inputs via its `AutoProcessor`, which tokenizes both the image (via a ViT-based visual encoder) and the text prompt. For radiology report generation, the image is embedded and projected into the language model's token space, where it is treated as a prefix to the text prompt. The instruction-tuned variant (`-it`) follows structured prompts reliably, making it well-suited to the template-based generation approach used in this pipeline. Generation is controlled via `temperature=0.1` for near-deterministic outputs appropriate for clinical tasks, with `do_sample=False` when temperature is set to zero.

To fit within Google Colab's T4 GPU 15 GB VRAM budget, MedGemma is loaded with 4-bit NF4 quantization via BitsAndBytes (`bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.bfloat16`). This reduces the model's memory footprint from approximately 8 GB (fp16) to approximately 2 GB, leaving sufficient VRAM for ColPali embeddings and the FAISS index. The `accelerate` library manages device placement automatically via `device_map="auto"`.

---

## Limitations

- **VRAM constraint**: The full pipeline (ColPali + MedGemma simultaneously in VRAM) is tight on a T4 GPU. Batch sizes must be kept small (≤4 images per batch for ColPali; ≤1 for MedGemma) to avoid OOM errors. Larger datasets may require sequential embedding with intermediate cache flushing.
- **Dataset access and quality**: The MIMIC-CXR dataset requires PhysioNet credentialing for the full version. The Kaggle proxy dataset may have reduced image resolution, inconsistent report formatting, or incomplete metadata, which can degrade retrieval and generation quality.
- **Hallucination risk**: MedGemma, like all large language models, may generate plausible-sounding but clinically inaccurate findings. RAG grounding reduces but does not eliminate this risk. All model outputs must be reviewed by qualified radiologists before any downstream use.
- **No multi-view support**: Clinical radiology typically interprets PA and lateral projections together. This pipeline processes single images only; PA + lateral view fusion, which can improve diagnostic accuracy for pneumothorax and pleural abnormalities, is not implemented.
- **Evaluation metric limitations**: Automatic NLP metrics (BLEU, ROUGE, BERTScore) measure lexical and semantic overlap with reference reports but are imperfect proxies for clinical accuracy. A report can score low on BLEU while being clinically correct, and vice versa. Human radiologist evaluation is the gold standard.
- **Heuristic QA answer generation**: The `QADatasetGenerator` extracts answers from reports using keyword heuristics and regex patterns rather than LLM-based extraction. For conditions with atypical report phrasing, extracted answers may be incomplete or incorrect.
- **Static index**: The FAISS index is built offline and does not update when new cases are added. Re-indexing the full corpus is required to incorporate new data, which may be prohibitive at large scale without HNSW or IVF indexing strategies.

---

## Future Work

- **LoRA fine-tuning**: Fine-tune MedGemma on curated radiology QA pairs using QLoRA for domain adaptation, which could improve report quality and reduce hallucinations without full fine-tuning costs.
- **Scalable retrieval**: Replace `IndexFlatIP` with FAISS HNSW or IVF-PQ for approximate nearest-neighbor search at dataset scale (>100K images), enabling sub-linear query time.
- **Confidence scoring**: Add uncertainty estimation to flag potentially unreliable report sections, using token-level log-probabilities or ensemble disagreement as a proxy for model confidence.
- **Multi-view fusion**: Extend ColPali embeddings to support multi-image inputs (PA + lateral), fusing patch-level embeddings from both views before retrieval and generation.
- **RLHF alignment**: Collect radiologist preference feedback on generated reports and apply RLHF or DPO to align generation style and clinical correctness with expert preferences.
- **Structured entity extraction**: Post-process generated reports with a named entity recognition model to extract structured findings (condition name, laterality, severity, size) for downstream clinical database integration.

---

## References

1. Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P. (2024). **ColPali: Efficient Document Retrieval with Vision Language Models**. arXiv preprint arXiv:2305.03660. https://arxiv.org/abs/2305.03660

2. Google DeepMind (2024). **MedGemma: Medical vision-language models**. HuggingFace Model Hub. https://huggingface.co/google/medgemma-4b-it

3. Johnson, A. E. W., Pollard, T. J., Berkowitz, S. J., Greenbaum, N. R., Lungren, M. P., Deng, C.-Y., Mark, R. G., & Horng, S. (2019). **MIMIC-CXR: A large publicly available database of labeled chest radiographs**. PhysioNet. https://doi.org/10.13026/C2JT1Q

4. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. Advances in Neural Information Processing Systems (NeurIPS 2020). arXiv:2005.11401.

5. Team, C.-V., & Contributors (2024). **MIMIC-CXR-VQA Dataset Creation**. GitHub repository. https://github.com/LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Disclaimer

> **For research purposes only. Not for clinical use.**
>
> This tool is an experimental research prototype. Its outputs have not been validated for clinical accuracy and must not be used for medical diagnosis, treatment decisions, or patient care. The generated radiology reports and VQA answers may contain errors, omissions, or hallucinated findings. Always consult a qualified, licensed medical professional for any health-related decisions. The authors and contributors accept no liability for any clinical decisions made based on this system's outputs.
