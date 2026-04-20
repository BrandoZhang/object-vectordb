"""Text-to-video nearest-neighbor query against the seeded collection.

Usage:

    python query.py "meow"
    python query.py "guitar solo" --k 5
    python query.py "thunderstorm" --k 3 --category nature

The query string is embedded with the same sentence-transformers model used at
seed time and compared against the ``audio_caption_embedding`` field via
`Collection.search`. Results are printed ranked by cosine similarity.
"""

from __future__ import annotations

import argparse

from seed import COLLECTION_NAME, DB_URI, EMBED_MODEL_NAME, VECTOR_FIELD
from sentence_transformers import SentenceTransformer

from object_vectordb import ObjectVectorDB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="natural-language description to search for")
    parser.add_argument("--k", type=int, default=5, help="number of results to return")
    parser.add_argument(
        "--category",
        default=None,
        help="optional exact-match category filter (cats, dogs, music, cooking, gaming, nature)",
    )
    args = parser.parse_args()

    db = ObjectVectorDB(uri=DB_URI)
    if not db.has_collection(COLLECTION_NAME):
        raise SystemExit(
            f"No '{COLLECTION_NAME}' collection at {DB_URI}. Run `python seed.py` first."
        )
    collection = db.collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    query_vec = model.encode([args.text], normalize_embeddings=True, convert_to_numpy=True)[0]

    where = f"category = '{args.category}'" if args.category else None
    hits = collection.search(
        query_vector=query_vec,
        vector_field=VECTOR_FIELD,
        limit=args.k,
        metric="cosine",
        where=where,
        select=["title", "audio_caption", "category"],
    )

    print(f"\nTop {len(hits)} matches for {args.text!r}:\n")
    for rank, hit in enumerate(hits, start=1):
        props = hit.properties
        print(f"  {rank}. [{props['category']:7s}] score={hit.score:+.3f}  {props['title']}")
        print(f"         audio: {props['audio_caption']}")
    print()


if __name__ == "__main__":
    main()
