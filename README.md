# Handwritten Notes OCR

A local-first OCR pipeline for converting handwritten notes to text using state-of-the-art vision-language models. Supports multiple OCR models for comparison and runs entirely on your machine (Apple Silicon MPS, NVIDIA CUDA, or CPU).

## Features

- 🔒 **100% Local** - No data leaves your machine, no API keys required (except HuggingFace for model downloads)
- 🍎 **Apple Silicon Support** - Optimized for MPS on M1/M2/M3 Macs
- 📄 **Full-Page OCR** - Handles entire pages of handwritten notes, not just single lines
- 🔬 **Multiple Models** - Compare results between GOT-OCR-2.0 and olmOCR-2
- 📓 **Interactive Notebook** - Jupyter notebook for easy experimentation

## Models

| Model | Source | Best For |
|-------|--------|----------|
| [GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) | StepFun AI | General document OCR |
| [olmOCR-2-7B-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) | Allen AI | Document understanding |

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate handwritten-ocr
```

### 2. HuggingFace Authentication

Some models require HuggingFace authentication. Create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your HuggingFace token
```

Get your token at: https://huggingface.co/settings/tokens

### 3. Register Jupyter Kernel (Optional)

```bash
python -m ipykernel install --user --name handwritten-ocr --display-name "Handwritten OCR"
```

## Project Structure

```
.
├── create_notes.py          # Main OCR module
├── environment.yml          # Conda environment
├── .env                     # HuggingFace token (not committed)
├── .env.example             # Template for .env
├── data/
│   ├── input/              # Place your handwritten images here
│   └── output/             # Extracted text files saved here
├── notebooks/
│   └── ocr_playground.ipynb # Interactive testing notebook
├── tests/
│   └── test_ocr.py         # End-to-end tests
└── README.md
```

## Usage

### Interactive Notebook (Recommended)

The easiest way to test is with the Jupyter notebook:

```bash
jupyter notebook notebooks/ocr_playground.ipynb
```

### Process Images Programmatically

```python
from create_notes import HandwrittenOCR

ocr = HandwrittenOCR()
text = ocr.process_image("path/to/your/image.png")
print(text)
```

### Batch Processing

Place images in `data/input/` and run:

```bash
python create_notes.py
```

## Supported Formats

- PNG, JPG, JPEG, BMP, TIFF

## Hardware Requirements

- **Minimum**: 8GB RAM, any modern CPU
- **Recommended**: 16GB+ RAM, Apple Silicon (M1/M2/M3) or NVIDIA GPU
- **Note**: First run downloads models (~5-15GB depending on models used)

## Troubleshooting

### Model not loading?
- Check HuggingFace authentication in `.env`
- Ensure enough disk space for model cache (~20GB)

### Slow inference?
- Ensure PyTorch is using MPS (Mac) or CUDA (NVIDIA)
- Check with: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Poor OCR quality?
- Try a different model (GOT-OCR vs olmOCR)
- Ensure good image quality and lighting
- Consider pre-processing (contrast, rotation)

## License

[MIT](LICENSE)
