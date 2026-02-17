# Copilot Instructions — Handwritten OCR

## Project Overview

Local-first OCR pipeline converting handwritten note images to text using a multi-agent LangGraph workflow. Uses **olmOCR** (`allenai/olmOCR-2-7B-1025`) for OCR and **Ollama** (`qwen3:32b`) for LLM-based quality improvement. Runs on Apple Silicon (MPS), NVIDIA (CUDA), or CPU — no cloud APIs. Packaged with Poetry (`pyproject.toml`).

## Architecture

### Core Modules (`ocr_agent/`)

| Module | Purpose |
|---|---|
| `config.py` | All tuneable parameters (model names, thresholds, strategies) |
| `state.py` | `OCRState` TypedDict shared across the LangGraph pipeline |
| `graph.py` | LangGraph definition — nodes, edges, conditional routing |
| `nodes.py` | LangGraph node implementations (initial_ocr, critic, editor, reocr, accept, plateau, max_iterations) |
| `agents.py` | Pydantic-validated LLM agents: Critic, Editor, Arbitrator |
| `tools.py` | Deterministic functions: OCR inference, image preprocessing, metrics (CER/WER), LLM calls via Ollama |
| `transcribe.py` | CLI entry point — arg parsing, pipeline invocation, output saving |
| `trace.py` | `Trace` class for timestamped event logging and observability |
| `eval_final.py` | Standalone evaluation script (CER/WER against ground truth) |
| `__main__.py` | `python -m ocr_agent` entry point, routes to `transcribe.main()` |

### Pipeline Flow (LangGraph)

```
START → initial_ocr (2-pass + agreement check)
      → critic (LLM evaluates quality)
      → routing:
          ├─ accept (confidence >= threshold) → END
          ├─ plateau (no improvement for N iters) → END
          ├─ max_iterations → END
          ├─ needs_editing → editor → critic (loop)
          └─ needs_reocr → reocr (next preprocessing strategy + arbitrator merge) → critic (loop)
```

### Three-Agent System

- **Critic** — Evaluates OCR quality, returns confidence score + verdict (accept/needs_editing/needs_reocr)
- **Editor** — Fixes issues flagged by the Critic via LLM
- **Arbitrator** — Merges multiple OCR candidates when re-OCR produces a new version

### Preprocessing Strategies

Six strategies tried on plateau/re-OCR: `deskew`, `denoise`, `high_contrast`, `binarize`, `sharpen`, `remove_lines`. Configured in `config.py`.

## Environment & Setup

```bash
poetry install                        # core dependencies
poetry install --extras notebooks     # include Jupyter/matplotlib
```

- Python 3.10+, PyTorch with MPS/CUDA support
- HuggingFace token required in `.env` (see `.env.example`) — needed for gated model downloads
- Ollama must be running locally with the configured model (`qwen3:32b`)
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

## Output Format Convention

When saving OCR results, produce `.txt` (raw text), `.json` (full trace with timing/metrics), and `.md` (with metadata header).

## Data Layout

- `data/input/` — Place handwritten images here (PNG, JPG, JPEG, BMP, TIFF)
- `data/output/` — OCR results and ground truth files (`*_gt.md`)

## Conventions

- Use `pathlib.Path` for all file paths, not `os.path`
- Images are always converted to RGB before processing: `Image.open(path).convert("RGB")`
- Pydantic v2 models for all structured LLM outputs
- LangGraph `TypedDict` for pipeline state (not dataclasses)
- Lazy imports for heavy dependencies (`torch`, `transformers`) to keep CLI responsive
- Print progress with emoji checkmarks for user-facing status
- Never commit `.env`, model caches, or `data/input/` images (see `.gitignore`)
