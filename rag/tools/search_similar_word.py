"""Tool: search_similar_word — find confirmed transcriptions matching a noisy OCR word."""

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
from rag.embeddings.visual_embedder import embed_image_from_base64
from rag.utils.scoring import reciprocal_rank_fusion


@dataclass
class WordMatch:
    """A word-level match with text and visual scores."""

    confirmed_text: str
    raw_ocr_text: str
    text_score: float
    visual_score: float
    fused_score: float
    source_image_id: str


def search_similar_word(
    word_ocr: str,
    image_crop_base64: str | None = None,
    top_k: int = settings.TOP_K_DEFAULT,
) -> list[WordMatch]:
    """Find confirmed transcriptions that best match a noisy OCR word.

    Runs a hybrid text search (dense + sparse) filtered to word/phrase chunks.
    If an image crop is provided, also runs a visual search and fuses results
    with Reciprocal Rank Fusion.

    Args:
        word_ocr: The raw OCR output for a single word (possibly misspelled/noisy).
        image_crop_base64: Optional base64-encoded image crop of the word region.
        top_k: Maximum number of results to return.

    Returns:
        List of WordMatch dataclasses sorted by fused score descending.
    """
    client = get_client()
    dense_vecs = embed_dense([word_ocr])
    sparse_vecs = embed_sparse([word_ocr])

    word_filter = Filter(
        must=[
            FieldCondition(key="confidence_score", range=Range(gt=0.0)),
            FieldCondition(
                key="chunk_type",
                match=MatchAny(any=["word", "phrase"]),
            ),
        ]
    )

    # Text path: hybrid prefetch with RRF
    prefetch = [
        Prefetch(
            query=dense_vecs[0],
            using="text_dense",
            limit=top_k * 2,
            filter=word_filter,
        ),
        Prefetch(
            query=sparse_vecs[0],
            using="text_sparse",
            limit=top_k * 2,
            filter=word_filter,
        ),
    ]

    text_results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=prefetch,
        query=dense_vecs[0],
        using="text_dense",
        limit=top_k,
        with_payload=True,
    )

    # Build text scores map
    text_scores: dict[str, float] = {}
    payload_map: dict[str, dict] = {}
    for point in text_results.points:
        pid = str(point.id)
        text_scores[pid] = point.score
        payload_map[pid] = point.payload

    # Visual path (optional)
    visual_scores: dict[str, float] = {}
    if image_crop_base64:
        visual_vec = embed_image_from_base64(image_crop_base64)
        visual_results = client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=visual_vec,
            using="visual",
            limit=top_k,
            with_payload=True,
            query_filter=word_filter,
        )
        for point in visual_results.points:
            pid = str(point.id)
            visual_scores[pid] = point.score
            if pid not in payload_map:
                payload_map[pid] = point.payload

    # Fuse with RRF if visual path was run
    if visual_scores:
        text_list = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)
        visual_list = sorted(visual_scores.items(), key=lambda x: x[1], reverse=True)
        fused = reciprocal_rank_fusion([text_list, visual_list])
    else:
        fused = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)

    # Deduplicate by confirmed_text keeping highest fused score
    seen_texts: set[str] = set()
    matches: list[WordMatch] = []
    for pid, fused_score in fused:
        payload = payload_map.get(pid, {})
        confirmed = payload.get("confirmed_text", "")
        if confirmed in seen_texts:
            continue
        seen_texts.add(confirmed)
        matches.append(
            WordMatch(
                confirmed_text=confirmed,
                raw_ocr_text=payload.get("raw_ocr_text", ""),
                text_score=text_scores.get(pid, 0.0),
                visual_score=visual_scores.get(pid, 0.0),
                fused_score=fused_score,
                source_image_id=payload.get("source_image_id", ""),
            )
        )
        if len(matches) >= top_k:
            break

    logger.debug(
        "search_similar_word returned {} matches for word_ocr='{}'",
        len(matches),
        word_ocr,
    )
    return matches
