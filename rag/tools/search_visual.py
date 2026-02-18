"""Tool: search_visual — text-to-image search using CLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from qdrant_client.models import FieldCondition, Filter, Range

from rag.client import get_client
from rag.config import settings
from rag.embeddings.visual_embedder import embed_text_clip


@dataclass
class VisualSearchResult:
    """A single visual search result."""

    source_image_id: str
    region_coords: dict
    score: float
    confirmed_text: str
    topic_tags: list[str]
    confidence_score: float


def search_visual(
    query: str,
    top_k: int = 10,
    annotated_only: bool = False,
) -> list[VisualSearchResult]:
    """Search indexed images by text query using CLIP text-to-image matching.

    Encodes the query with CLIP's text encoder and searches the visual
    vector space. Works on all indexed images — including those without
    text annotations.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return.
        annotated_only: If True, only return images that have been
            manually annotated (confidence_score > 0).

    Returns:
        List of VisualSearchResult sorted by CLIP similarity score.
    """
    client = get_client()
    visual_vec = embed_text_clip(query)

    query_filter = None
    if annotated_only:
        query_filter = Filter(must=[
            FieldCondition(key="confidence_score", range=Range(gt=0.0)),
        ])

    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=visual_vec,
        using="visual",
        limit=top_k,
        with_payload=["source_image_id", "region_coords", "confirmed_text", "topic_tags", "confidence_score"],
        query_filter=query_filter,
    )

    logger.debug(
        "search_visual returned {} results for query='{}'",
        len(results.points),
        query,
    )

    return [
        VisualSearchResult(
            source_image_id=point.payload.get("source_image_id", ""),
            region_coords=point.payload.get("region_coords", {}),
            score=point.score,
            confirmed_text=point.payload.get("confirmed_text", ""),
            topic_tags=point.payload.get("topic_tags", []),
            confidence_score=point.payload.get("confidence_score", 0.0),
        )
        for point in results.points
    ]
