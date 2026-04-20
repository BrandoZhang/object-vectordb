"""Build the demo collection and populate it with the sample YouTube-style dataset.

Run this once before `query.py` or `visualize.py`:

    python seed.py

What it does:

1. Opens an `ObjectVectorDB` at ``./data/youtube.lance`` (relative to this file).
2. Drops and recreates a ``videos`` collection so re-running is idempotent.
3. Registers a single vector field, ``audio_caption_embedding`` (384-dim).
4. Encodes every ``audio_caption`` string with sentence-transformers
   ``all-MiniLM-L6-v2`` (cosine-normalized so ``metric="cosine"`` works cleanly).
5. Inserts all ~60 objects in one `add_many` call.

No index is created — at this scale the default flat scan is fast and avoids the
IVF index-metric pinning behavior described in the project README.
"""

from __future__ import annotations

from pathlib import Path

from sample_videos import load_videos
from sentence_transformers import SentenceTransformer

from object_vectordb import ObjectVectorDB

DB_URI = str(Path(__file__).resolve().parent / "data" / "youtube.lance")
COLLECTION_NAME = "videos"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_FIELD = "audio_caption_embedding"
EMBEDDING_DIM = 384


def main() -> None:
    videos = load_videos()
    print(f"Loaded {len(videos)} sample videos.")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Encoding audio captions...")
    captions = [v["audio_caption"] for v in videos]
    embeddings = model.encode(
        captions,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    assert embeddings.shape == (len(videos), EMBEDDING_DIM), embeddings.shape

    print(f"Opening database at {DB_URI}")
    db = ObjectVectorDB(uri=DB_URI)
    if db.has_collection(COLLECTION_NAME):
        print(f"Dropping existing '{COLLECTION_NAME}' collection for a clean reseed.")
        db.drop_collection(COLLECTION_NAME)
    collection = db.collection(COLLECTION_NAME)
    collection.register_vector_field(
        VECTOR_FIELD,
        dim=EMBEDDING_DIM,
        description="all-MiniLM-L6-v2 over the audio_caption property",
    )

    items = []
    for video, vector in zip(videos, embeddings, strict=True):
        items.append(
            {
                "object_id": video["object_id"],
                "properties": {
                    "title": video["title"],
                    "subtitle": video["subtitle"],
                    "audio_caption": video["audio_caption"],
                    "category": video["category"],
                    "thumbnail_url": video["thumbnail_url"],
                    "video_url": video["video_url"],
                },
                "vectors": {VECTOR_FIELD: vector.tolist()},
            }
        )
    collection.add_many(items)

    print(f"Inserted {len(items)} objects into '{COLLECTION_NAME}'.")
    print('Next: run `python query.py "meow"` or `python visualize.py`.')


if __name__ == "__main__":
    main()
