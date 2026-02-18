"""Collection schema and initialization for the handwritten_notes collection.

Defines named vectors, payload indexes, and provides idempotent collection creation.
"""

from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from rag.config import settings


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str | None = None,
) -> None:
    """Create the handwritten_notes collection if it doesn't already exist.

    Idempotent — safe to call on every startup.

    Args:
        client: An initialized QdrantClient.
        collection_name: Override collection name (useful for tests). Defaults to settings value.
    """
    name = collection_name or settings.COLLECTION_NAME

    if client.collection_exists(name):
        logger.info("Collection '{}' already exists — skipping creation.", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config={
            "text_dense": VectorParams(size=1024, distance=Distance.COSINE),
            "visual": VectorParams(size=512, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "text_sparse": SparseVectorParams(),
        },
    )
    logger.info("Created collection '{}'.", name)

    # Payload indexes for filtered search performance
    client.create_payload_index(
        collection_name=name,
        key="chunk_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=name,
        key="confidence_score",
        field_schema=PayloadSchemaType.FLOAT,
    )
    logger.info("Created payload indexes on 'chunk_type' and 'confidence_score'.")
