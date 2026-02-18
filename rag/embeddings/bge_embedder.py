"""BGE dense and sparse text embeddings via fastembed.

Dense: BAAI/bge-large-en-v1.5 (1024-dim)
Sparse: Qdrant/bm42-all-minilm-l6-v2-attentions

Models are lazily loaded on first call and cached at module level.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from qdrant_client.models import SparseVector

_dense_model = None
_sparse_model = None


def _get_dense_model():
    """Lazily load and cache the BGE-M3 dense embedding model."""
    global _dense_model
    if _dense_model is None:
        from fastembed import TextEmbedding

        logger.info("Loading BGE-large-en-v1.5 dense model (first call — ~1.2GB)...")
        _dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
        logger.info("BGE dense model loaded.")
    return _dense_model


def _get_sparse_model():
    """Lazily load and cache the BGE-M3 sparse embedding model."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding

        logger.info("Loading BGE-M3 sparse model (first call)...")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        logger.info("BGE-M3 sparse model loaded.")
    return _sparse_model


def _normalize(vectors: list[np.ndarray]) -> list[list[float]]:
    """Normalize vectors to unit length."""
    result = []
    for v in vectors:
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        result.append(v.tolist())
    return result


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Compute dense embeddings for a list of texts using BGE-M3.

    Args:
        texts: Input texts to embed. Processed in batches of 32.

    Returns:
        List of 1024-dim unit-normalized embedding vectors.
    """
    model = _get_dense_model()
    embeddings = list(model.embed(texts, batch_size=32))
    return _normalize(embeddings)


def embed_sparse(texts: list[str]) -> list[SparseVector]:
    """Compute sparse embeddings for a list of texts using BGE-M3.

    Args:
        texts: Input texts to embed.

    Returns:
        List of Qdrant-compatible SparseVector objects.
    """
    model = _get_sparse_model()
    raw = list(model.embed(texts, batch_size=32))
    result = []
    for sparse_emb in raw:
        result.append(
            SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            )
        )
    return result
