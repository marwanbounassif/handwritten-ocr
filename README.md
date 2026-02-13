# Handwritten Notes OCR

A local-first OCR pipeline for converting handwritten notes to text using state-of-the-art vision-language models. Supports multiple OCR models for comparison and runs entirely on your machine (Apple Silicon MPS, NVIDIA CUDA, or CPU).

## CLI Commands

### Single image
python -m ocr_agent path/to/image.jpg

### With ground truth
python -m ocr_agent path/to/image.jpg --ground-truth path/to/gt.md

### Custom thresholds
python -m ocr_agent path/to/image.jpg --max-iterations 15 --accept-threshold 90

### Batch
python -m ocr_agent path/to/folder/ --output-dir results/

### Standalone eval
python -m ocr_agent.eval_final path/to/transcription.txt --ground-truth path/to/gt.md
