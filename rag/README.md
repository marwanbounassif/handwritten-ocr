# rag Tooling Layer for Handwritten Note Transcription

A collection of tools and infrastructure for rag-based disambiguation of handwritten note transcriptions. This is **not** an agent — it provides typed, composable tools that an agent can call.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- A [Qdrant Cloud](https://cloud.qdrant.io) account (free tier available)

## Qdrant Cloud Setup

1. Create an account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster from the dashboard
3. Go to **Data Access Control** → create an API key
4. Copy the cluster URL and API key into your `.env` file

## Environment Variables

Create a `.env` file in the **project root** (not inside `rag/`):

```bash
# Qdrant Cloud connection
QDRANT_END_POINT=https://YOUR-CLUSTER.cloud.qdrant.io   # Cluster URL from dashboard
QDRANT_API_KEY=your_api_key_here                         # API key from Data Access Control

# Collection settings
COLLECTION_NAME=handwritten_notes                        # Name of the Qdrant collection
CONFIDENCE_WRITE_BACK_THRESHOLD=0.85                     # Min confidence to auto-upsert
TOP_K_DEFAULT=5                                          # Default number of search results

# Model cache — set to your existing HF cache to avoid redownloading
HF_HOME=~/.cache/huggingface
```

## Installation

```bash
poetry lock && poetry install
```

## Memory Footprint

Models load **lazily** on first tool call — nothing loads at import time.

| Model | RAM | Purpose |
|-------|-----|---------|
| BGE-M3 (fastembed) | ~570 MB | Dense + sparse text embeddings |
| CLIP ViT-B/32 (open_clip) | ~350 MB | Visual image/text embeddings |
| **Total** | **~1 GB** | |

## HF_HOME

Set `HF_HOME` in your `.env` to point to your existing HuggingFace cache directory. This avoids redownloading models if you already have them cached.

```bash
# Check your current cache location
echo $HF_HOME
# If unset, models are typically cached in ~/.cache/huggingface
```

## Usage

### Search for context

```python
from rag.tools.search_context import search_context

results = search_context("mitochondria energy production", top_k=5)
for r in results:
    print(f"{r.confirmed_text} (score={r.score:.3f}, type={r.chunk_type})")
```

### Find similar words (OCR disambiguation)

```python
from rag.tools.search_similar_word import search_similar_word

# Text-only search
matches = search_similar_word("accleration", top_k=3)

# With image crop for visual matching
matches = search_similar_word("accleration", image_crop_base64="...", top_k=3)
for m in matches:
    print(f"{m.confirmed_text} (fused={m.fused_score:.4f})")
```

### Search by topic

```python
from rag.tools.search_by_topic import search_by_topic

# Full results
results = search_by_topic("physics mechanics", top_k=10)

# Just unique tags
tags = search_by_topic("physics", return_unique_tags_only=True)
print(tags)  # ['kinematics', 'mechanics', 'physics']
```

### Upsert a confirmed transcription

```python
from rag.ingest.upsert import upsert_transcription

point_id = upsert_transcription(
    confirmed_text="acceleration",
    raw_ocr_text="accleration",
    source_image_id="img_002",
    region_coords={"x": 5, "y": 60, "w": 120, "h": 30},
    confidence_score=0.92,
    topic_tags=["physics"],
    chunk_type="word",
)
```

## Running Tests

Tests require a live Qdrant Cloud connection (reads from `.env`). They use a separate `test_handwritten_notes` collection and clean up after themselves.

```bash
pytest rag/tests/test_tools.py -v
```
