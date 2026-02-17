"""
Deterministic tool functions for the agentic OCR pipeline.
Extracted and refactored from the playground and eval notebooks.

Functions:
  - run_ocr: Run olmOCR-2 on an image
  - evaluate: Full evaluation (hard metrics + LLM scoring/correction)
  - compare_versions: Character/word-level diff between two candidates
  - merge_versions: Majority-vote merge of multiple candidates
  - preprocess_image: Apply image preprocessing transforms
  - normalize_text: Unicode normalization and whitespace cleanup
  - levenshtein / cer / wer: Hard metrics
"""

import gc
import json
import re
import tempfile
from pathlib import Path

from PIL import Image

from ocr_agent import config

# ── Ground Truth Parsing ──────────────────────────────────────────


def parse_ground_truth(file_path: str | Path) -> str | None:
    """
    Read a ground-truth markdown file and extract only the text under
    the ``## Ground Truth`` header.  Returns None if the file doesn't
    exist or has no such section.
    """
    p = Path(file_path)
    if not p.exists():
        return None
    raw = p.read_text(encoding="utf-8")
    # Find the "## Ground Truth" header and take everything after it
    marker = "## Ground Truth"
    idx = raw.find(marker)
    if idx == -1:
        # No header found — assume the whole file is plain-text ground truth
        return raw.strip() or None
    text = raw[idx + len(marker) :].strip()
    return text or None


# ── Text Normalization ────────────────────────────────────────────


def normalize_text(text: str, lower: bool = False) -> str:
    """Unicode normalization, curly quotes, dashes, whitespace collapse."""
    t = text
    # Curly quotes → straight
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    # Dashes → hyphen
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    if lower:
        t = t.lower()
    return t


# ── Hard Metrics ──────────────────────────────────────────────────


def levenshtein(a: str, b: str) -> int:
    """Standard DP Levenshtein distance (character-level)."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if a[i - 1] == b[j - 1] else 1),
            )
            prev = cur
    return dp[m]


def _levenshtein_words(a: list[str], b: list[str]) -> int:
    """Levenshtein on word-token lists."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if a[i - 1] == b[j - 1] else 1),
            )
            prev = cur
    return dp[m]


def cer(ground_truth: str, ocr_output: str, lower: bool = False) -> float:
    """Character Error Rate: edit distance / ground truth length."""
    gt = normalize_text(ground_truth, lower)
    ocr = normalize_text(ocr_output, lower)
    return levenshtein(gt, ocr) / max(len(gt), 1)


def wer(ground_truth: str, ocr_output: str, lower: bool = False) -> float:
    """Word Error Rate (token-level): word edit distance / ground truth word count."""
    gt = normalize_text(ground_truth, lower)
    ocr = normalize_text(ocr_output, lower)
    gt_words = gt.split()
    ocr_words = ocr.split()
    return _levenshtein_words(gt_words, ocr_words) / max(len(gt_words), 1)


def tier1_metrics(ground_truth: str, ocr_output: str, lower: bool = False) -> dict:
    """CER, WER, exact-match between ground truth and OCR output."""
    gt = normalize_text(ground_truth, lower)
    ocr = normalize_text(ocr_output, lower)

    cer_val = levenshtein(gt, ocr) / max(len(gt), 1)
    gt_words, ocr_words = gt.split(), ocr.split()
    wer_char = levenshtein(" ".join(gt_words), " ".join(ocr_words)) / max(
        len(" ".join(gt_words)), 1
    )
    wer_tok = _levenshtein_words(gt_words, ocr_words) / max(len(gt_words), 1)

    return {
        "input": ocr_output,
        "cer": round(cer_val, 4),
        "wer": round(wer_char, 4),
        "wer_token": round(wer_tok, 4),
        "exact_match": gt == ocr,
        "gt_chars": len(gt),
        "ocr_chars": len(ocr),
    }


# ── LLM Helper ───────────────────────────────────────────────────

import sys

import ollama


def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool | None = None,
) -> str:
    """
    Call Ollama chat endpoint. Returns the assistant message text.
    Thinking mode is controlled via ollama's native `think` parameter —
    when enabled, reasoning goes into message.thinking and content stays clean.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    should_stream = stream if stream is not None else config.LLM_STREAM
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE
    options = {"temperature": temp, "num_predict": max_tokens or config.LLM_MAX_TOKENS}

    print(f"  [llm] Calling {config.OLLAMA_MODEL}...", flush=True)

    if should_stream:
        text = _call_llm_stream(messages, options)
    else:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            options=options,
            think=config.LLM_ENABLE_THINKING,
        )
        text = response["message"]["content"]

    print(f"  [llm] Done ({len(text)} chars)")
    return text


def _call_llm_stream(messages: list[dict], options: dict) -> str:
    """Stream an LLM response, printing tokens live. Returns the full assembled text."""
    chunks: list[str] = []

    for part in ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        stream=True,
        options=options,
        think=config.LLM_ENABLE_THINKING,
    ):
        token = part["message"]["content"]
        if not token:
            continue

        chunks.append(token)
        sys.stdout.write(token)
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(chunks)


def parse_json_response(raw: str) -> dict | None:
    """
    Robustly parse JSON from an LLM response.
    Handles markdown fences, preamble text, etc.
    Returns None if parsing fails.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } or [ ... ] block
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == start_char:
                depth += 1
            elif cleaned[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def call_llm_json(
    system_prompt: str,
    user_message: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_schema: dict | None = None,
) -> dict:
    """
    Call LLM and parse JSON response.
    If json_schema is provided, uses ollama's format parameter for structured output.
    Otherwise falls back to text parsing with retry.
    Returns parsed dict, or a fallback error dict.
    """
    if json_schema is not None:
        # Use ollama's native structured output
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        options = {"temperature": temp, "num_predict": max_tokens or config.LLM_MAX_TOKENS}

        print(f"  [llm] Calling {config.OLLAMA_MODEL} (structured)...", flush=True)
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            format=json_schema,
            options=options,
            think=config.LLM_ENABLE_THINKING,
        )
        text = response["message"]["content"]
        result = parse_json_response(text)
        if result is not None:
            return result

    # Fallback: free-text call with JSON parsing
    raw = call_llm(system_prompt, user_message, temperature, max_tokens)
    result = parse_json_response(raw)
    if result is not None:
        return result

    # Retry with explicit JSON instruction
    retry_msg = (
        user_message
        + "\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown, no preamble, no explanation."
    )
    raw = call_llm(system_prompt, retry_msg, temperature, max_tokens)
    result = parse_json_response(raw)
    if result is not None:
        return result

    # Fallback
    return {"error": "json_parse_failed", "raw_response": raw}


# ── Evaluation ────────────────────────────────────────────────────


def evaluate(
    transcription: str,
    ground_truth: str | None = None,
    lower: bool = False,
) -> dict:
    """
    Compute hard metrics (CER/WER) against ground truth.
    Returns structured dict with tier-1 metrics.
    """
    result = {}

    if ground_truth is not None:
        print("  [eval] Computing CER/WER against ground truth...")
        result["tier1_raw_vs_gt"] = tier1_metrics(ground_truth, transcription, lower)

    return result


# ── Compare Versions ──────────────────────────────────────────────


def compare_versions(v1: str, v2: str) -> dict:
    """
    Compute character-level and word-level diff between two transcription candidates.
    Returns agreement rate, edit distances, and differing segments.
    """
    n1 = normalize_text(v1)
    n2 = normalize_text(v2)

    char_dist = levenshtein(n1, n2)
    max_chars = max(len(n1), len(n2), 1)
    agreement_rate = round((1 - char_dist / max_chars) * 100, 1)

    w1 = n1.split()
    w2 = n2.split()
    word_dist = _levenshtein_words(w1, w2)

    # Find differing segments by aligning words
    differing_segments = _find_differing_segments(w1, w2)

    return {
        "agreement_rate": agreement_rate,
        "char_edit_distance": char_dist,
        "word_edit_distance": word_dist,
        "differing_segments": differing_segments,
    }


def _find_differing_segments(w1: list[str], w2: list[str]) -> list[dict]:
    """Find word-level differing segments between two word lists using LCS alignment."""
    segments = []
    # Use a simple sliding window approach for practicality
    i, j = 0, 0
    while i < len(w1) and j < len(w2):
        if w1[i] == w2[j]:
            i += 1
            j += 1
        else:
            # Collect the differing run
            start_i, start_j = i, j
            # Look ahead for next sync point
            found = False
            for look in range(1, min(10, max(len(w1) - i, len(w2) - j) + 1)):
                # Check if w1[i+look] matches w2[j] or w2[j+look] matches w1[i]
                if i + look < len(w1) and j < len(w2) and w1[i + look] == w2[j]:
                    segments.append({
                        "position": start_i,
                        "v1_text": " ".join(w1[start_i : i + look]),
                        "v2_text": " ".join(w2[start_j:j]) if start_j < j else "",
                    })
                    i = i + look
                    found = True
                    break
                if j + look < len(w2) and i < len(w1) and w2[j + look] == w1[i]:
                    segments.append({
                        "position": start_i,
                        "v1_text": " ".join(w1[start_i:i]) if start_i < i else "",
                        "v2_text": " ".join(w2[start_j : j + look]),
                    })
                    j = j + look
                    found = True
                    break
            if not found:
                # Skip both
                segments.append({
                    "position": start_i,
                    "v1_text": w1[i] if i < len(w1) else "",
                    "v2_text": w2[j] if j < len(w2) else "",
                })
                i += 1
                j += 1

    # Handle remaining words
    if i < len(w1) or j < len(w2):
        segments.append({
            "position": i,
            "v1_text": " ".join(w1[i:]),
            "v2_text": " ".join(w2[j:]),
        })

    return segments


# ── Merge Versions ────────────────────────────────────────────────


def merge_versions(versions: list[str]) -> str:
    """
    Given multiple transcription candidates, produce a merged best version.
    Uses word-level majority vote. Where no majority exists, keeps all variants
    as [v1|v2|v3] for the editor to resolve.
    """
    if not versions:
        return ""
    if len(versions) == 1:
        return versions[0]

    # Normalize and split into words
    word_lists = [normalize_text(v).split() for v in versions]

    # Use the longest version as the backbone for alignment
    backbone_idx = max(range(len(word_lists)), key=lambda i: len(word_lists[i]))
    backbone = word_lists[backbone_idx]

    # Align all other versions to the backbone using simple sequential alignment
    aligned = [_align_to_backbone(backbone, wl) for wl in word_lists]

    # Majority vote at each position
    merged_words = []
    for pos in range(len(backbone)):
        candidates = []
        for a in aligned:
            if pos < len(a) and a[pos] is not None:
                candidates.append(a[pos])

        if not candidates:
            merged_words.append(backbone[pos])
            continue

        # Count votes
        vote_counts: dict[str, int] = {}
        for c in candidates:
            vote_counts[c] = vote_counts.get(c, 0) + 1

        max_votes = max(vote_counts.values())
        winners = [w for w, count in vote_counts.items() if count == max_votes]

        if len(winners) == 1:
            merged_words.append(winners[0])
        else:
            # No clear majority — keep all unique variants
            unique = list(dict.fromkeys(candidates))  # preserve order, deduplicate
            if len(unique) == 1:
                merged_words.append(unique[0])
            else:
                merged_words.append("[" + "|".join(unique) + "]")

    return " ".join(merged_words)


def _align_to_backbone(backbone: list[str], words: list[str]) -> list[str | None]:
    """
    Align a word list to the backbone using longest-common-subsequence positions.
    Returns a list the same length as backbone, with aligned words or None for gaps.
    """
    n, m = len(backbone), len(words)
    # LCS table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if backbone[i - 1].lower() == words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find aligned positions
    aligned = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if backbone[i - 1].lower() == words[j - 1].lower():
            aligned[i - 1] = words[j - 1]
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return aligned


# ── Image Preprocessing ──────────────────────────────────────────
#
# Each transform is a standalone function: PIL Image → PIL Image.
# preprocess_image() accepts a single strategy string OR a list of
# strings to chain multiple transforms sequentially (e.g. ["deskew", "high_contrast", "sharpen"]).


def _apply_high_contrast(img: Image.Image) -> Image.Image:
    """CLAHE contrast enhancement. Falls back to PIL if OpenCV unavailable."""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return Image.fromarray(clahe.apply(gray))
    except ImportError:
        from PIL import ImageEnhance

        return ImageEnhance.Contrast(img).enhance(2.0)


def _apply_binarize(img: Image.Image) -> Image.Image:
    """Adaptive thresholding. Falls back to PIL point() if OpenCV unavailable."""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        return Image.fromarray(
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        )
    except ImportError:
        return img.convert("L").point(lambda x: 255 if x > 128 else 0)


def _apply_sharpen(img: Image.Image) -> Image.Image:
    """Unsharp-mask style sharpening. Falls back to PIL if OpenCV unavailable."""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return Image.fromarray(cv2.filter2D(img_array, -1, kernel))
    except ImportError:
        from PIL import ImageFilter

        return img.filter(ImageFilter.SHARPEN)


def _apply_deskew(img: Image.Image) -> Image.Image:
    """Detect skew angle via minAreaRect and rotate to correct. Needs OpenCV."""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) <= 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return Image.fromarray(rotated)
    except ImportError:
        return img


def _apply_denoise(img: Image.Image) -> Image.Image:
    """Non-local means denoising. Smooths scanner/camera noise while preserving edges."""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)
        if len(img_array.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
        return Image.fromarray(denoised)
    except ImportError:
        return img


def _apply_remove_lines(img: Image.Image) -> Image.Image:
    """Remove horizontal ruled lines from notebook paper. Needs OpenCV."""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array

        # Detect horizontal lines with a wide kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1] // 4, 1))
        lines_mask = cv2.morphologyEx(
            cv2.adaptiveThreshold(
                cv2.bitwise_not(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
            ),
            cv2.MORPH_OPEN,
            horizontal_kernel,
            iterations=1,
        )

        # Dilate the line mask slightly so we cover the full line width
        lines_mask = cv2.dilate(lines_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

        # Inpaint the lines away
        result = cv2.inpaint(img_array, lines_mask, 3, cv2.INPAINT_TELEA)
        return Image.fromarray(result)
    except ImportError:
        return img


# Registry of single-step transforms
_TRANSFORMS = {
    "high_contrast": _apply_high_contrast,
    "binarize": _apply_binarize,
    "sharpen": _apply_sharpen,
    "deskew": _apply_deskew,
    "denoise": _apply_denoise,
    "remove_lines": _apply_remove_lines,
}


def preprocess_image(image_path: str, strategy: str | list[str]) -> str:
    """
    Apply image preprocessing and save to a temp file. Returns the new path.

    strategy can be:
      - A single string:  "high_contrast"
      - A list of strings: ["deskew", "high_contrast", "sharpen"]  (applied left-to-right)
      - "original" (no-op, returns the input path)

    Available transforms: original, high_contrast, binarize, sharpen, deskew, denoise, remove_lines
    """
    # Normalise to a list
    if isinstance(strategy, str):
        steps = [strategy]
    else:
        steps = list(strategy)

    # No-op shortcut
    if steps == ["original"] or not steps:
        return image_path

    label = "+".join(s for s in steps if s != "original")
    print(f"  [preprocess] Applying {label}...")
    img = Image.open(image_path)

    for step in steps:
        if step == "original":
            continue
        fn = _TRANSFORMS.get(step)
        if fn is None:
            print(f"  [preprocess] Unknown transform '{step}', skipping")
            continue
        img = fn(img)

    # Build a label for the temp filename
    label = "+".join(s for s in steps if s != "original")
    suffix = Path(image_path).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix=f"ocr_{label}_")
    result_path = tmp.name
    img.save(result_path)
    return result_path


# ── OCR (olmOCR-2) ───────────────────────────────────────────────

# Module-level model cache to avoid reloading
_ocr_model = None
_ocr_processor = None


def _load_ocr_model():
    """Load olmOCR-2 model and processor. Cached at module level."""
    global _ocr_model, _ocr_processor
    if _ocr_model is not None:
        return _ocr_model, _ocr_processor

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"  [ocr] Loading {config.OLMOCR_MODEL} on {device}...")
    _ocr_processor = AutoProcessor.from_pretrained(
        config.OLMOCR_MODEL,
        min_pixels=config.OCR_MIN_PIXELS,
        max_pixels=config.OCR_MAX_PIXELS,
    )
    _ocr_model = AutoModelForImageTextToText.from_pretrained(
        config.OLMOCR_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"  [ocr] Model loaded.")
    return _ocr_model, _ocr_processor


def unload_ocr_model():
    """Free OCR model memory so Ollama has room."""
    global _ocr_model, _ocr_processor
    import torch

    del _ocr_model, _ocr_processor
    _ocr_model = None
    _ocr_processor = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("  [ocr] Model unloaded, memory freed.")


def run_ocr(image_path: str, params: dict | None = None) -> str:
    """
    Run olmOCR-2 on an image. Returns raw transcription text.
    params can include:
      - prompt: custom extraction prompt
      - max_new_tokens: override generation length
    """
    import torch

    print(f"  [ocr] Running OCR on {Path(image_path).name}...")
    params = params or {}
    model, processor = _load_ocr_model()

    prompt = params.get("prompt", config.OCR_PROMPT)
    max_new_tokens = params.get("max_new_tokens", config.OCR_MAX_NEW_TOKENS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    device = next(model.parameters()).device

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    result = processor.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    print(f"  [ocr] Done ({len(result)} chars)")
    return result
