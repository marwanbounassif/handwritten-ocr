# Handwritten Notes OCR

A local-first OCR pipeline for converting handwritten notes to text using vision-language models. Runs entirely on your machine (Apple Silicon MPS, NVIDIA CUDA, or CPU).

## Setup

Requires Python 3.10+ and [Poetry](https://python-poetry.org/docs/#installation).

```bash
# Install dependencies
poetry install

# (Optional) Include Jupyter/notebook support
poetry install --extras notebooks
```

Copy `.env.example` to `.env` and add your [HuggingFace token](https://huggingface.co/settings/tokens) — required for downloading gated models:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=your_token_here
```

## CLI Commands

```bash
# Single image
ocr path/to/image.jpg

# With ground truth comparison
ocr path/to/image.jpg --ground-truth path/to/gt.md

# Custom thresholds
ocr path/to/image.jpg --max-iterations 15 --accept-threshold 90

# Batch (all images in a folder)
ocr path/to/folder/ --output-dir results/

# Standalone evaluation
python -m ocr_agent.eval_final path/to/transcription.txt --ground-truth path/to/gt.md
```

If you prefer not to use the script entry point, `python -m ocr_agent` works identically to `ocr`.
