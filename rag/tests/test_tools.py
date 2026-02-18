"""Integration tests for rag tools — requires a live Qdrant Cloud connection.

Uses a separate test collection to avoid touching production data.
Run with: pytest rag/tests/test_tools.py -v
"""

from __future__ import annotations

import base64
import io
import time

import pytest
from PIL import Image

from rag.client import get_client
from rag.ingest.upsert import upsert_transcription
from rag.schema import ensure_collection_exists
from rag.tools.search_by_topic import search_by_topic
from rag.tools.search_context import search_context
from rag.tools.search_similar_word import search_similar_word

TEST_COLLECTION = "test_handwritten_notes"

# Synthetic test data covering 2-3 distinct topics
SYNTHETIC_NOTES = [
    {
        "confirmed_text": "mitochondria is the powerhouse of the cell",
        "raw_ocr_text": "mitochondna is the powerhoue of the cel",
        "source_image_id": "img_001",
        "region_coords": {"x": 10, "y": 20, "w": 300, "h": 40},
        "confidence_score": 0.95,
        "topic_tags": ["biology", "cell-biology"],
        "chunk_type": "sentence",
    },
    {
        "confirmed_text": "photosynthesis converts light energy",
        "raw_ocr_text": "photosyntheis converts light enrgy",
        "source_image_id": "img_001",
        "region_coords": {"x": 10, "y": 70, "w": 300, "h": 40},
        "confidence_score": 0.90,
        "topic_tags": ["biology", "botany"],
        "chunk_type": "sentence",
    },
    {
        "confirmed_text": "chloroplast",
        "raw_ocr_text": "chloroplst",
        "source_image_id": "img_001",
        "region_coords": {"x": 10, "y": 120, "w": 100, "h": 30},
        "confidence_score": 0.88,
        "topic_tags": ["biology", "botany"],
        "chunk_type": "word",
    },
    {
        "confirmed_text": "Newton's second law F equals ma",
        "raw_ocr_text": "Newtons secnd law F equls ma",
        "source_image_id": "img_002",
        "region_coords": {"x": 5, "y": 10, "w": 350, "h": 45},
        "confidence_score": 0.92,
        "topic_tags": ["physics", "mechanics"],
        "chunk_type": "sentence",
    },
    {
        "confirmed_text": "acceleration",
        "raw_ocr_text": "accleration",
        "source_image_id": "img_002",
        "region_coords": {"x": 5, "y": 60, "w": 120, "h": 30},
        "confidence_score": 0.87,
        "topic_tags": ["physics"],
        "chunk_type": "word",
    },
    {
        "confirmed_text": "velocity is the rate of change of displacement",
        "raw_ocr_text": "velocty is the rate of chage of displacmnt",
        "source_image_id": "img_002",
        "region_coords": {"x": 5, "y": 110, "w": 380, "h": 40},
        "confidence_score": 0.91,
        "topic_tags": ["physics", "kinematics"],
        "chunk_type": "sentence",
    },
    {
        "confirmed_text": "the quadratic formula",
        "raw_ocr_text": "the quadratc formla",
        "source_image_id": "img_003",
        "region_coords": {"x": 20, "y": 15, "w": 200, "h": 35},
        "confidence_score": 0.93,
        "topic_tags": ["math", "algebra"],
        "chunk_type": "phrase",
    },
    {
        "confirmed_text": "derivative",
        "raw_ocr_text": "derivtive",
        "source_image_id": "img_003",
        "region_coords": {"x": 20, "y": 60, "w": 100, "h": 30},
        "confidence_score": 0.86,
        "topic_tags": ["math", "calculus"],
        "chunk_type": "word",
    },
    {
        "confirmed_text": "integral calculus deals with areas under curves",
        "raw_ocr_text": "intgral calculs deals with areas undr curves",
        "source_image_id": "img_003",
        "region_coords": {"x": 20, "y": 100, "w": 400, "h": 40},
        "confidence_score": 0.89,
        "topic_tags": ["math", "calculus"],
        "chunk_type": "sentence",
    },
    {
        "confirmed_text": "momentum",
        "raw_ocr_text": "momentm",
        "source_image_id": "img_002",
        "region_coords": {"x": 5, "y": 160, "w": 90, "h": 30},
        "confidence_score": 0.85,
        "topic_tags": ["physics", "mechanics"],
        "chunk_type": "word",
    },
]


def _make_dummy_image_base64() -> str:
    """Generate a small white PNG image encoded as base64."""
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def setup_test_collection():
    """Create test collection, upsert synthetic data, yield, then clean up."""
    import rag.config as cfg

    # Override collection name for all tools during tests
    original_name = cfg.settings.COLLECTION_NAME
    object.__setattr__(cfg.settings, "COLLECTION_NAME", TEST_COLLECTION)

    client = get_client()

    # Clean up any leftover test collection
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)

    ensure_collection_exists(client, collection_name=TEST_COLLECTION)

    # Upsert all synthetic notes (bypass confidence threshold)
    for note in SYNTHETIC_NOTES:
        upsert_transcription(**note, respect_confidence_threshold=False)

    # Allow Qdrant to index
    time.sleep(2)

    yield client

    # Teardown
    client.delete_collection(TEST_COLLECTION)
    object.__setattr__(cfg.settings, "COLLECTION_NAME", original_name)


def test_search_context_clean_query(setup_test_collection):
    """search_context returns relevant results for a clean query."""
    results = search_context("cell biology energy", top_k=3)
    assert len(results) > 0
    texts = [r.confirmed_text for r in results]
    assert any("mitochondria" in t or "photosynthesis" in t for t in texts)


def test_search_context_noisy_query(setup_test_collection):
    """search_context returns reasonable results for a garbled/noisy query."""
    results = search_context("mitochndra powrhouse cel", top_k=3)
    assert len(results) > 0


def test_search_context_chunk_type_filter(setup_test_collection):
    """search_context respects chunk_type filter."""
    results = search_context("physics", chunk_types=["word"], top_k=5)
    for r in results:
        assert r.chunk_type == "word"


def test_search_similar_word_misspelled(setup_test_collection):
    """search_similar_word finds correct word for a misspelling."""
    results = search_similar_word("accleration", top_k=3)
    assert len(results) > 0
    confirmed_texts = [r.confirmed_text for r in results]
    assert "acceleration" in confirmed_texts


def test_search_similar_word_with_image(setup_test_collection):
    """search_similar_word doesn't crash with a dummy base64 image and returns results."""
    dummy_b64 = _make_dummy_image_base64()
    results = search_similar_word("derivtive", image_crop_base64=dummy_b64, top_k=3)
    assert isinstance(results, list)


def test_search_by_topic_thematic(setup_test_collection):
    """search_by_topic returns thematically grouped results."""
    results = search_by_topic("physics and motion", top_k=5)
    assert len(results) > 0
    all_tags = set()
    for r in results:
        all_tags.update(r.topic_tags)
    assert "physics" in all_tags


def test_search_by_topic_unique_tags(setup_test_collection):
    """search_by_topic with return_unique_tags_only returns a flat string list."""
    tags = search_by_topic("math", return_unique_tags_only=True)
    assert isinstance(tags, list)
    assert all(isinstance(t, str) for t in tags)
    assert len(tags) > 0


def test_upsert_below_confidence_threshold(setup_test_collection):
    """Upsert below confidence threshold returns None and does not write."""
    result = upsert_transcription(
        confirmed_text="should not be stored",
        raw_ocr_text="shoud not be stord",
        source_image_id="img_skip",
        region_coords={"x": 0, "y": 0, "w": 50, "h": 20},
        confidence_score=0.50,
        topic_tags=["test"],
        chunk_type="word",
        respect_confidence_threshold=True,
    )
    assert result is None


def test_upsert_idempotency(setup_test_collection):
    """Upserting the same region twice produces one point, not two."""
    client = setup_test_collection
    coords = {"x": 999, "y": 999, "w": 50, "h": 20}

    id1 = upsert_transcription(
        confirmed_text="idempotent test",
        raw_ocr_text="idmpotent tst",
        source_image_id="img_idem",
        region_coords=coords,
        confidence_score=0.95,
        topic_tags=["test"],
        chunk_type="word",
        respect_confidence_threshold=False,
    )

    id2 = upsert_transcription(
        confirmed_text="idempotent test updated",
        raw_ocr_text="idmpotent tst updtd",
        source_image_id="img_idem",
        region_coords=coords,
        confidence_score=0.96,
        topic_tags=["test"],
        chunk_type="word",
        respect_confidence_threshold=False,
    )

    assert id1 == id2

    # Verify only one point exists with this ID
    points = client.retrieve(collection_name=TEST_COLLECTION, ids=[id1])
    assert len(points) == 1
    assert points[0].payload["confirmed_text"] == "idempotent test updated"
