# Copilot Instructions — Handwritten OCR

## Project Overview

Local-first OCR pipeline converting handwritten note images to text using HuggingFace vision-language models. Runs on Apple Silicon (MPS), NVIDIA (CUDA), or CPU — no cloud APIs for inference. The project is packaged with Poetry (`pyproject.toml`).

## Architecture

- **`create_notes.py`** — Core module. Contains the `HandwrittenOCR` class using Microsoft TrOCR (`microsoft/trocr-large-handwritten`) for single-image and batch-directory processing. This is the programmatic/CLI entry point.
- **`notebooks/ocr_playground.ipynb`** — Interactive experimentation notebook. Tests additional models (GOT-OCR-2.0-hf, olmOCR-2-7B-FP8) beyond what `create_notes.py` supports. This is where new models are prototyped before being added to the main module.
- **`data/input/`** — User places handwritten images here (PNG, JPG, JPEG, BMP, TIFF).
- **`data/output/`** — OCR results saved as `.txt` (plain text) and `.md` (markdown with metadata: source filename, timestamp, model name).

### Key discrepancy to know

`create_notes.py` uses **TrOCR** (single-line OCR), while the notebook experiments with **GOT-OCR-2.0-hf** and **olmOCR-2** (full-page OCR). The notebook represents the more current direction. When adding features, prefer the full-page models (GOT-OCR, olmOCR) over TrOCR for handwritten note use cases.

## Environment & Setup

```bash
poetry install                        # core dependencies
poetry install --extras notebooks     # include Jupyter/matplotlib
```

- Python 3.10+, PyTorch with MPS/CUDA support
- HuggingFace token required in `.env` (see `.env.example`) — needed for gated model downloads
- Token loaded via `python-dotenv`; the notebook loads from `Path.cwd().parent / ".env"`
- CLI entry point: `ocr` (or `python -m ocr_agent`)

## Device Selection Pattern

Always use this pattern for device selection (do not hardcode a device):
```python
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

## Model Loading Pattern

Models are loaded via `transformers` (`AutoProcessor` + `AutoModelForImageTextToText` or `VisionEncoderDecoderModel`). Use `torch.bfloat16` for GPU/MPS, `torch.float32` for CPU. Pass `device_map=device` to `from_pretrained()`.

## Output Format Convention

When saving OCR results, always produce **both** `.txt` and `.md` files:
- `.txt` — raw transcribed text only
- `.md` — includes metadata header (source filename, timestamp, model name used) followed by the transcribed text. See `save_ocr_result()` in the notebook for the template.

## Testing

Tests live in `tests/test_ocr.py` and are **end-to-end** — they load the actual model, generate synthetic test images with PIL, run OCR, and verify output. They are not unit tests with mocks. Run with:
```bash
python tests/test_ocr.py
```
Tests use `sys.path.insert` to import from the project root. Test artifacts (`test_data/`, `test_output/`) are created and cleaned up automatically.

## Conventions

- Use `pathlib.Path` for all file paths, not `os.path`
- Images are always converted to RGB before processing: `Image.open(path).convert("RGB")`
- Print progress with emoji checkmarks (`✓`/`✗`) for user-facing status
- `data/output/` has a `.gitkeep` to preserve the empty directory in git
- Never commit `.env`, model caches, or `data/input/` images (see `.gitignore`)
