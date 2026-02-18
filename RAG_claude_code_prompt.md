# Claude Code Prompt: rag Tooling Layer for Handwritten Note Transcription

Create a new folder called `rag/` and implement a complete rag tooling layer for handwritten note transcription disambiguation. This is NOT an agent — it is a collection of tools and infrastructure that an agent will later call. Build everything production-ready with clean interfaces.

---

## Stack

- Vector store: Qdrant Cloud (hosted, REST API)
- Text embeddings: BGE-M3 via fastembed (local, handles dense + sparse in one model)
- Visual embeddings: CLIP ViT-B/32 via open_clip (local)
- Language: Python 3.11+
- Package manager: Poetry
- No external LLM API calls of any kind

---

## Folder Structure to Create

```
rag/
├── __init__.py
├── config.py
├── client.py
├── schema.py
├── embeddings/
│   ├── __init__.py
│   ├── bge_embedder.py      # Dense + sparse via BGE-M3
│   └── visual_embedder.py   # CLIP image patch embeddings
├── tools/
│   ├── __init__.py
│   ├── search_context.py
│   ├── search_similar_word.py
│   └── search_by_topic.py
├── ingest/
│   ├── __init__.py
│   └── upsert.py
├── utils/
│   ├── __init__.py
│   └── scoring.py
├── tests/
│   ├── __init__.py
│   └── test_tools.py
└── README.md
```

---

## .env File

The project expects a `.env` file in the project root. Do NOT hardcode any of these values. Required variables:

```
QDRANT_URL=https://YOUR-CLUSTER.qdrant.io     # Qdrant Cloud cluster URL
QDRANT_API_KEY=your_qdrant_api_key            # Qdrant Cloud API key
COLLECTION_NAME=handwritten_notes
CONFIDENCE_WRITE_BACK_THRESHOLD=0.85
TOP_K_DEFAULT=5
HF_HOME=/path/to/your/hf/cache               # Point to existing HF cache to avoid redownloads
```

Add `.env` to `.gitignore`. Never commit secrets.

---

## config.py

- Load all values from `.env` using `python-dotenv`
- Expose a single `Settings` dataclass or simple namespace with all config values
- Validate on import: if `QDRANT_URL` or `QDRANT_API_KEY` are missing, raise a clear `EnvironmentError` with instructions
- All other modules import config from here — never call `os.getenv` directly elsewhere

---

## client.py

- Create a Qdrant Cloud client using `QdrantClient(url=..., api_key=...)`
- Expose a single module-level singleton `get_client()` that lazily initializes and caches the client
- On first call, verify connectivity with a lightweight `client.get_collections()` call and log success
- Raise a clear error with troubleshooting hints if connection fails

---

## Schema / Collection Design

Collection name: `handwritten_notes`

Named vectors:

1. `text_dense` — 1024 dims, Cosine distance (BGE-M3 dense output)
2. `text_sparse` — sparse vector, Dot distance (BGE-M3 sparse output)
3. `visual` — 512 dims, Cosine distance (CLIP ViT-B/32)

Payload schema per point:

```json
{
  "confirmed_text": "string",
  "raw_ocr_text": "string",
  "source_image_id": "string",
  "region_coords": {"x": int, "y": int, "w": int, "h": int},
  "confidence_score": float,
  "topic_tags": ["string"],
  "chunk_type": "sentence | phrase | word",
  "created_at": "ISO timestamp",
  "from_human_review": bool
}
```

Write `ensure_collection_exists(client)` in `schema.py`:
- Idempotent — safe to call every startup
- Creates collection with all three named vectors if it doesn't exist
- Creates payload index on `chunk_type` and `confidence_score` for filtered search performance

---

## Embeddings Layer

### bge_embedder.py

Use `fastembed` library with `BAAI/bge-m3` model.

- Lazy-load the model on first call (do not load at import time)
- Cache the loaded model at module level after first load
- Method: `embed_dense(texts: list[str]) -> list[list[float]]`
  - Uses BGE-M3's dense passage embeddings
  - Batch size max 32 (BGE-M3 is heavier than lighter models)
- Method: `embed_sparse(texts: list[str]) -> list[SparseVector]`
  - Uses BGE-M3's sparse embeddings
  - Returns Qdrant-compatible `SparseVector` objects
- Both methods must call the model in a single pass per batch — do not load the model twice
- Normalize dense vectors to unit length before returning

### visual_embedder.py

Use `open_clip` library, model `ViT-B-32`, pretrained `openai`.

- Lazy-load model and transforms on first call
- Method: `embed_image(image: PIL.Image) -> list[float]`
- Method: `embed_image_from_base64(b64_string: str) -> list[float]`
- Method: `embed_text_clip(text: str) -> list[float]`
  - CLIP text encoder — used as visual vector fallback when no image crop is available
- Normalize all outputs to unit vectors
- Run on CPU explicitly (do not assume GPU availability — Qwen3:32B may be holding GPU memory)

---

## Tool 1: search_context

File: `tools/search_context.py`

```python
def search_context(
    query: str,
    top_k: int = settings.TOP_K_DEFAULT,
    chunk_types: list[str] | None = None,
    min_confidence: float = 0.0,
) -> list[SearchResult]
```

- Embed `query` with `embed_dense` and `embed_sparse` from `bge_embedder`
- Run Qdrant hybrid search: prefetch on `text_dense` and `text_sparse` separately, fuse with RRF
- Apply payload filters for `chunk_type` (if provided) and `confidence_score >= min_confidence`
- Return list of `SearchResult` dataclasses:

```python
@dataclass
class SearchResult:
    confirmed_text: str
    raw_ocr_text: str
    score: float
    source_image_id: str
    topic_tags: list[str]
    chunk_type: str
    confidence_score: float
```

---

## Tool 2: search_similar_word

File: `tools/search_similar_word.py`

```python
def search_similar_word(
    word_ocr: str,
    image_crop_base64: str | None = None,
    top_k: int = settings.TOP_K_DEFAULT,
) -> list[WordMatch]
```

- Text path: embed `word_ocr` with dense + sparse, search with hybrid fusion, filter `chunk_type` in `["word", "phrase"]`
- Visual path (if `image_crop_base64` provided): decode to PIL, embed with `visual_embedder`, search `visual` named vector separately with same `top_k`
- Merge text path and visual path results using RRF from `utils/scoring.py`
- Deduplicate by `confirmed_text` keeping highest fused score
- Return list of `WordMatch` dataclasses:

```python
@dataclass
class WordMatch:
    confirmed_text: str
    raw_ocr_text: str
    text_score: float
    visual_score: float        # 0.0 if visual path not run
    fused_score: float
    source_image_id: str
```

---

## Tool 3: search_by_topic

File: `tools/search_by_topic.py`

```python
def search_by_topic(
    inferred_topic: str,
    top_k: int = 10,
    return_unique_tags_only: bool = False,
) -> list[SearchResult] | list[str]
```

- Embed `inferred_topic` with `embed_dense` only (broad topic queries don't benefit from sparse)
- Search `text_dense` vector
- If `return_unique_tags_only=True`: extract all `topic_tags` from results, flatten, deduplicate, return sorted list of strings
- Otherwise return full `SearchResult` list

---

## ingest/upsert.py

```python
def upsert_transcription(
    confirmed_text: str,
    raw_ocr_text: str,
    source_image_id: str,
    region_coords: dict,
    confidence_score: float,
    topic_tags: list[str],
    chunk_type: str,
    image_crop_base64: str | None = None,
    from_human_review: bool = False,
    respect_confidence_threshold: bool = True,
) -> str | None
```

- If `respect_confidence_threshold=True` and `confidence_score < CONFIDENCE_WRITE_BACK_THRESHOLD`: log warning, return `None`
- Generate deterministic UUID5 from `f"{source_image_id}_{region_coords}"` for idempotent upserts
- Embed `confirmed_text` with `embed_dense` and `embed_sparse`
- For visual vector: use `embed_image_from_base64` if crop provided, else `embed_text_clip(confirmed_text)` as fallback
- Build named vectors dict and upsert single point to Qdrant
- Return the UUID string on success

---

## utils/scoring.py

```python
def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
```

Standard RRF: `score = sum(1 / (k + rank))` across all lists, where rank is 1-indexed. Sort descending by fused score. Input is list of `(point_id, score)` lists.

---

## pyproject.toml Dependencies

Add these to the existing Poetry project — do not create a new `pyproject.toml`:

```toml
python-dotenv = "^1.0"
qdrant-client = "^1.9"
fastembed = "^0.3"
open-clip-torch = "^2.24"
Pillow = "^10.0"
loguru = "^0.7"
torch = {version = "^2.2"}   # CPU only, no cuda extra needed
```

After adding, remind the user to run:
```
poetry lock && poetry install
```

---

## tests/test_tools.py

Write integration tests requiring a live Qdrant Cloud connection (reads credentials from `.env`):

1. Use a separate test collection `"test_handwritten_notes"` — never touch the production collection
2. Create the test collection fresh at test start, delete it on teardown (use pytest fixtures)
3. Upsert 8–10 synthetic note chunks covering 2–3 distinct topics with varied chunk types and confidence scores
4. Test `search_context` returns semantically relevant results for a clean query
5. Test `search_context` still returns reasonable results for a deliberately garbled/noisy query
6. Test `search_similar_word` with a misspelled word — assert correct word appears in top 3
7. Test `search_similar_word` with a dummy base64 image (generate a small white PIL image, encode to base64) — assert no crash and returns results
8. Test `search_by_topic` returns thematically grouped results
9. Test `search_by_topic` with `return_unique_tags_only=True` returns a flat string list
10. Test upsert below confidence threshold returns `None` and does not write to Qdrant
11. Test upsert idempotency — upserting the same region twice produces one point, not two

---

## README.md

Include:

- Prerequisites (Python 3.11+, Poetry)
- Qdrant Cloud setup instructions (where to get URL and API key from the Qdrant Cloud console)
- Full `.env` template with comments explaining each variable
- Memory footprint note: BGE-M3 ~570MB RAM, CLIP ~350MB RAM, total ~1GB — models load lazily on first tool call
- Example usage snippets for all three tools and the upsert function
- Note on `HF_HOME`: set this to your existing HuggingFace cache path to avoid redownloading models

---

## General Requirements

- All tools return typed dataclasses, not raw dicts
- All Qdrant interactions go through the singleton in `client.py`
- Models load lazily — nothing loads at import time
- Every public function has a docstring explaining purpose, args, and return type
- Use `loguru` for all logging — no `print` statements
- Handle Qdrant Cloud connection errors with clear messages (mention checking API key and URL)
- Do not use LangChain, LlamaIndex, or any rag framework — implement directly against the `qdrant-client` SDK
- Do not make any external API calls for embeddings or LLM inference — everything runs locally

---

## Before Running This Prompt

1. **Qdrant Cloud**: Create a free account at [cloud.qdrant.io](https://cloud.qdrant.io), spin up a free cluster, and copy the cluster URL and API key from the dashboard into your `.env`.

2. **HF_HOME**: Run `echo $HF_HOME` in your terminal. If unset, your models are likely in `~/.cache/huggingface`. Set that path in `.env` so BGE-M3 and CLIP don't redownload if already cached.
