"""Tool: search_context — hybrid semantic search over transcribed notes."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    Prefetch,
    Range,
)

from rag.client import get_client
from rag.config import settings
from rag.embeddings.bge_embedder import embed_dense, embed_sparse


@dataclass
class SearchResult:
    """A single search result with payload metadata."""

    confirmed_text: str
    raw_ocr_text: str
    score: float
    source_image_id: str
    topic_tags: list[str]
    chunk_type: str
    confidence_score: float


def search_context(
    query: str,
    top_k: int = settings.TOP_K_DEFAULT,
    chunk_types: list[str] | None = None,
    min_confidence: float = 0.0,
) -> list[SearchResult]:
    """Search for contextually relevant transcribed notes using hybrid text search.

    Embeds the query with BGE-M3 (dense + sparse), runs prefetch on both vectors,
    and fuses results with Reciprocal Rank Fusion via Qdrant's built-in query API.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return.
        chunk_types: Optional filter — only return chunks of these types
            (e.g. ["sentence", "phrase"]).
        min_confidence: Minimum confidence_score threshold for results.

    Returns:
        List of SearchResult dataclasses sorted by fused relevance score.
    """
    client = get_client()
    dense_vecs = embed_dense([query])
    sparse_vecs = embed_sparse([query])

    query_filter_conditions = []
    if chunk_types:
        query_filter_conditions.append(
            FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types))
        )
    if min_confidence > 0.0:
        query_filter_conditions.append(
            FieldCondition(
                key="confidence_score",
                range=Range(gte=min_confidence),
            )
        )

    query_filter = Filter(must=query_filter_conditions) if query_filter_conditions else None

    prefetch = [
        Prefetch(
            query=dense_vecs[0],
            using="text_dense",
            limit=top_k * 2,
            filter=query_filter,
        ),
        Prefetch(
            query=sparse_vecs[0],
            using="text_sparse",
            limit=top_k * 2,
            filter=query_filter,
        ),
    ]

    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=prefetch,
        query=dense_vecs[0],
        using="text_dense",
        limit=top_k,
        with_payload=True,
    )

    logger.debug("search_context returned {} results for query='{}'", len(results.points), query)

    return [
        SearchResult(
            confirmed_text=point.payload.get("confirmed_text", ""),
            raw_ocr_text=point.payload.get("raw_ocr_text", ""),
            score=point.score,
            source_image_id=point.payload.get("source_image_id", ""),
            topic_tags=point.payload.get("topic_tags", []),
            chunk_type=point.payload.get("chunk_type", ""),
            confidence_score=point.payload.get("confidence_score", 0.0),
        )
        for point in results.points
    ]
