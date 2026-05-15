#!/usr/bin/env bash
set -e

# --- Mount Google Drive ---
python - <<'EOF'
from google.colab import drive
drive.mount('/content/drive')
EOF

PROJECT_ROOT="/content/drive/MyDrive/medical-multimodal-rag"

# --- Install dependencies ---
pip install -r "$PROJECT_ROOT/requirements.txt" -q

# --- HuggingFace login ---
if [ -z "$HF_TOKEN" ]; then
  echo "WARNING: HF_TOKEN not set. Set it before running this script."
else
  huggingface-cli login --token "$HF_TOKEN"
fi

# --- Create runtime directories if missing ---
mkdir -p "$PROJECT_ROOT/results"
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/data/qa"
mkdir -p "$PROJECT_ROOT/.cache/colpali"
mkdir -p "$PROJECT_ROOT/.cache/medgemma"

echo "Setup complete. Project root: $PROJECT_ROOT"
