"""Qdrant Cloud client singleton.

Provides a lazily-initialized, cached Qdrant client accessible via get_client().
"""

from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient

from RAG.config import settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return the singleton Qdrant client, creating it on first call.

    Verifies connectivity on first initialization by listing collections.

    Returns:
        QdrantClient connected to the configured Qdrant Cloud cluster.

    Raises:
        ConnectionError: If the Qdrant cluster is unreachable.
    """
    global _client
    if _client is not None:
        return _client

    try:
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        collections = client.get_collections()
        logger.info(
            "Connected to Qdrant Cloud at {}. Found {} collection(s).",
            settings.QDRANT_URL,
            len(collections.collections),
        )
        _client = client
        return _client
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to Qdrant Cloud at {settings.QDRANT_URL}. "
            "Check that QDRANT_END_POINT and QDRANT_API_KEY in .env are correct. "
            f"Original error: {exc}"
        ) from exc
