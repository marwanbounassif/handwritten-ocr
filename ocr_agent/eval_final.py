#!/usr/bin/env python3
"""
Standalone evaluation script for verifying transcription quality.
Computes hard metrics (CER/WER) against ground truth.

Usage:
    python -m ocr_agent.eval_final path/to/transcription.txt --ground-truth path/to/gt.md
    python -m ocr_agent.eval_final results/ --ground-truth-dir gt/
"""

import argparse
import json
import sys
from pathlib import Path


def eval_single(
    transcription_path: Path,
    ground_truth_path: Path | None = None,
) -> dict:
    """Evaluate a single transcription file."""
    from ocr_agent.tools import evaluate, parse_ground_truth

    transcription = transcription_path.read_text(encoding="utf-8")

    ground_truth = parse_ground_truth(ground_truth_path) if ground_truth_path else None

    result = evaluate(transcription, ground_truth=ground_truth)
    result["file"] = str(transcription_path)
    return result


def print_eval(result: dict, name: str):
    """Print a human-readable evaluation summary."""
    print(f"\n{'='*60}")
    print(f"Evaluation: {name}")
    print(f"{'='*60}")

    if "tier1_raw_vs_gt" in result:
        t = result["tier1_raw_vs_gt"]
        print(f"\n  Raw vs Ground Truth:")
        print(f"    CER:       {t['cer']:.2%}")
        print(f"    WER (tok): {t['wer_token']:.2%}")
        print(f"    Exact:     {t['exact_match']}")
        print(f"    GT chars:  {t['gt_chars']}  |  OCR chars: {t['ocr_chars']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR transcription quality"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a transcription .txt file or directory of them",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground truth text file (single file mode)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Directory of ground truth files (batch mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation JSON (default: print to stdout)",
    )
    args = parser.parse_args()
    input_path: Path = args.input.resolve()

    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Single file mode
    if input_path.is_file():
        result = eval_single(input_path, args.ground_truth)
        print_eval(result, input_path.name)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nSaved: {args.output}")
        return

    # Batch mode
    txt_files = sorted(input_path.glob("*_transcription.txt"))
    if not txt_files:
        txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(txt_files)} files from {input_path}")
    all_results = []

    for txt_path in txt_files:
        gt_path = None
        if args.ground_truth_dir:
            # Match by stem, stripping _transcription suffix
            stem = txt_path.stem
            if stem.endswith("_transcription"):
                stem = stem[: -len("_transcription")]
            for ext in [".md", ".txt"]:
                candidate = args.ground_truth_dir / f"{stem}{ext}"
                if candidate.exists():
                    gt_path = candidate
                    break
        
        result = eval_single(txt_path, gt_path)
        print_eval(result, txt_path.name)
        all_results.append(result)

    # Summary
    if all_results and any("tier1_raw_vs_gt" in r for r in all_results):
        gt_results = [r for r in all_results if "tier1_raw_vs_gt" in r]
        avg_cer = sum(r["tier1_raw_vs_gt"]["cer"] for r in gt_results) / len(gt_results)
        avg_wer = sum(r["tier1_raw_vs_gt"]["wer_token"] for r in gt_results) / len(gt_results)
        print(f"\n{'='*60}")
        print(f"Batch Summary ({len(gt_results)} files with GT)")
        print(f"  Avg CER: {avg_cer:.2%}")
        print(f"  Avg WER: {avg_wer:.2%}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
