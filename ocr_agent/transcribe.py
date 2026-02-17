#!/usr/bin/env python3
"""
CLI entrypoint for the agentic OCR pipeline.

Usage:
    python -m ocr_agent.transcribe path/to/image.jpg
    python -m ocr_agent.transcribe path/to/image.jpg --ground-truth path/to/gt.md
    python -m ocr_agent.transcribe path/to/image.jpg --max-iterations 15 --accept-threshold 90
    python -m ocr_agent.transcribe path/to/folder/ --output-dir results/
"""

import argparse
import json
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def transcribe_single(
    image_path: Path,
    output_dir: Path,
    ground_truth_path: Path | None = None,
    max_iterations: int | None = None,
    accept_threshold: int | None = None,
) -> Path:
    """Transcribe a single image and save all outputs. Returns the transcription path."""
    import time

    from ocr_agent import config
    from ocr_agent.graph import build_ocr_graph
    from ocr_agent.tools import evaluate, parse_ground_truth
    from ocr_agent.trace import Trace

    name = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")

    # Build initial state
    initial_state = {
        "image_path": str(image_path),
        "candidates": [],
        "critiques": [],
        "edits": [],
        "current_best": "",
        "current_score": 0.0,
        "iteration": 0,
        "max_iterations": max_iterations or config.MAX_ITERATIONS,
        "status": "running",
        "reason": "",
        "strategies_used": [],
        "plateau_count": 0,
        "prev_score": 0.0,
        "prev_critique": None,
        "config": {
            "accept_threshold": accept_threshold or config.ACCEPT_THRESHOLD,
            "plateau_patience": config.PLATEAU_PATIENCE,
            "strategies": list(config.PREPROCESSING_STRATEGIES),
            "agreement_threshold": config.AGREEMENT_THRESHOLD,
        },
        "trace_events": [],
        "start_time": time.monotonic(),
    }

    # Run the graph
    graph = build_ocr_graph()
    final_state = graph.invoke(initial_state)

    # Reconstruct Trace for saving
    trace = Trace.from_events(final_state["trace_events"])

    # Save transcription
    transcription_path = output_dir / f"{name}_transcription.txt"
    transcription_path.write_text(final_state["current_best"], encoding="utf-8")
    print(f"\nSaved: {transcription_path}")

    # Save trace
    trace_json_path = output_dir / f"{name}_trace.json"
    trace.save_json(trace_json_path)
    print(f"Saved: {trace_json_path}")

    trace_summary_path = output_dir / f"{name}_trace_summary.txt"
    trace.save_summary(trace_summary_path)
    print(f"Saved: {trace_summary_path}")

    # Final evaluation (and GT comparison if provided)
    ground_truth = parse_ground_truth(ground_truth_path) if ground_truth_path else None

    eval_result = evaluate(final_state["current_best"], ground_truth=ground_truth)
    eval_result["pipeline_status"] = final_state["status"]
    eval_result["iterations"] = final_state["iteration"]
    eval_result["final_confidence"] = final_state["current_score"]

    eval_path = output_dir / f"{name}_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {eval_path}")

    # Print summary
    print(f"\n--- Result ---")
    print(f"Status: {final_state['status']}")
    print(f"Iterations: {final_state['iteration']}")
    print(f"Final confidence: {final_state['current_score']}")
    print(f"Transcription length: {len(final_state['current_best'])} chars")
    if ground_truth:
        raw = eval_result.get("tier1_raw_vs_gt", {})
        print(f"CER vs GT: {raw.get('cer', 'N/A')}")
        print(f"WER vs GT: {raw.get('wer_token', 'N/A')}")

    return transcription_path


def main():
    parser = argparse.ArgumentParser(
        description="Agentic OCR pipeline for handwritten documents"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an image file or a directory of images",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground truth text file (single image mode)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Directory of ground truth files (batch mode, matched by stem name)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations for the critique-edit loop",
    )
    parser.add_argument(
        "--accept-threshold",
        type=int,
        default=None,
        help="Critic confidence threshold to auto-accept (0-100)",
    )

    args = parser.parse_args()

    input_path: Path = args.input.resolve()

    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    elif input_path.is_dir():
        output_dir = input_path / "results"
    else:
        output_dir = input_path.parent

    # Single file mode
    if input_path.is_file():
        transcribe_single(
            image_path=input_path,
            output_dir=output_dir,
            ground_truth_path=args.ground_truth,
            max_iterations=args.max_iterations,
            accept_threshold=args.accept_threshold,
        )
        return

    # Batch mode
    images = sorted(
        f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No image files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images in {input_path}")

    for img_path in images:
        gt_path = None
        if args.ground_truth_dir:
            # Try matching by stem with common extensions
            for ext in [".md", ".txt"]:
                candidate = args.ground_truth_dir / f"{img_path.stem}{ext}"
                if candidate.exists():
                    gt_path = candidate
                    break

        transcribe_single(
            image_path=img_path,
            output_dir=output_dir,
            ground_truth_path=gt_path,
            max_iterations=args.max_iterations,
            accept_threshold=args.accept_threshold,
        )

    print(f"\nAll done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
