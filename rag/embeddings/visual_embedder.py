"""CLIP ViT-B/32 visual embeddings via open_clip.

Models are lazily loaded on first call and cached at module level.
All inference runs on CPU to avoid GPU contention.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from PIL import Image

_model = None
_preprocess = None
_tokenizer = None


def _load_clip():
    """Lazily load CLIP model, preprocess, and tokenizer."""
    global _model, _preprocess, _tokenizer
    if _model is None:
        import open_clip

        logger.info("Loading CLIP ViT-B-32 model (first call — ~350MB RAM)...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device="cpu"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.eval()
        logger.info("CLIP model loaded on CPU.")
    return _model, _preprocess, _tokenizer


def _normalize(v: np.ndarray) -> list[float]:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


def embed_image(image: Image.Image) -> list[float]:
    """Compute CLIP visual embedding for a PIL image.

    Args:
        image: A PIL Image to embed.

    Returns:
        512-dim unit-normalized embedding vector.
    """
    model, preprocess, _ = _load_clip()
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
    return _normalize(features.squeeze().numpy())


def embed_image_from_base64(b64_string: str) -> list[float]:
    """Compute CLIP visual embedding from a base64-encoded image.

    Args:
        b64_string: Base64-encoded image bytes.

    Returns:
        512-dim unit-normalized embedding vector.
    """
    from PIL import Image

    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return embed_image(image)


def embed_text_clip(text: str) -> list[float]:
    """Compute CLIP text embedding — used as visual vector fallback when no image crop is available.

    Args:
        text: Text to embed via CLIP text encoder.

    Returns:
        512-dim unit-normalized embedding vector.
    """
    model, _, tokenizer = _load_clip()
    tokens = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(tokens)
    return _normalize(features.squeeze().numpy())
