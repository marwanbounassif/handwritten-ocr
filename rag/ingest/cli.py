"""CLI for image ingestion pipeline.

Usage:
    python -m rag.ingest.cli upload FILE [FILE ...]   Upload images to S3
    python -m rag.ingest.cli index                     Sync S3 → local → Qdrant
    python -m rag.ingest.cli stats                     Show indexing progress
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any rag imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _upload(files: list[str]) -> None:
    """Upload local image files to S3."""
    from rag.storage.s3 import upload_image

    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"  [skip] {f} — file not found")
            continue
        key = upload_image(path)
        print(f"  [ok] {path.name} → {key}")


def _index() -> None:
    """Sync from S3 and auto-index all images into Qdrant."""
    from rag.ingest.index_images import index_all

    total = index_all(sync_s3=True)
    print(f"\nIndexed {total} total crops.")


def _stats() -> None:
    """Show indexing and annotation progress."""
    from rag.client import get_client
    from rag.config import settings
    from rag.schema import ensure_collection_exists

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_client()
    ensure_collection_exists(client)

    info = client.get_collection(settings.COLLECTION_NAME)
    total_points = info.points_count

    # Count annotated points (confirmed_text != "")
    # Qdrant doesn't support "not empty string" directly, so we scroll and count
    annotated = 0
    unannotated = 0
    topic_counts: dict[str, int] = {}
    source_ids: set[str] = set()

    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            break
        for point in results:
            payload = point.payload
            source_ids.add(payload.get("source_image_id", ""))
            if payload.get("confirmed_text", ""):
                annotated += 1
                for tag in payload.get("topic_tags", []):
                    topic_counts[tag] = topic_counts.get(tag, 0) + 1
            else:
                unannotated += 1
        if offset is None:
            break

    print(f"Collection: {settings.COLLECTION_NAME}")
    print(f"Total points:  {total_points}")
    print(f"  Annotated:   {annotated}")
    print(f"  Unannotated: {unannotated}")
    print(f"Source images: {len(source_ids)}")
    if topic_counts:
        print("\nTopic tags:")
        for tag, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
            print(f"  {tag}: {count}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "upload":
        if len(sys.argv) < 3:
            print("Usage: python -m rag.ingest.cli upload FILE [FILE ...]")
            sys.exit(1)
        _upload(sys.argv[2:])
    elif command == "index":
        _index()
    elif command == "stats":
        _stats()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
