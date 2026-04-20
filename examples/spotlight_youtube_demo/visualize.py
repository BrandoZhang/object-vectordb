"""Launch Renumics Spotlight on the seeded collection.

Usage:

    python visualize.py

Bridges `object_vectordb` -> pandas DataFrame -> Spotlight:

1. `Collection.list()` pulls every object with its properties.
2. `Collection.export_vectors()` pulls the embedding matrix in one shot and we
   join it onto the DataFrame by ``object_id`` (row order is not guaranteed to
   match ``list()``, hence the explicit join).
3. We hand Spotlight explicit dtype hints so the thumbnail column renders in the
   gallery lens, the video column plays in the detail panel, and the embedding
   column drives the Similarity Map lens (UMAP/PCA run in-browser).

Nothing about this bridge is library-specific — if the demo proves useful,
`build_dataframe` is the natural candidate to promote into a reusable
`object_vectordb.integrations.spotlight` module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from renumics import spotlight
from seed import COLLECTION_NAME, DB_URI, VECTOR_FIELD

from object_vectordb import Collection, ObjectVectorDB

PROPERTY_COLUMNS = [
    "title",
    "subtitle",
    "audio_caption",
    "category",
    "thumbnail_url",
    "video_url",
]


def build_dataframe(collection: Collection, vector_field: str) -> pd.DataFrame:
    """Materialize a collection as a pandas DataFrame for Spotlight."""
    objects = collection.list()
    if not objects:
        raise RuntimeError("Collection is empty. Run `python seed.py` before launching the viewer.")

    rows = []
    for obj in objects:
        row = {"object_id": obj.object_id}
        for col in PROPERTY_COLUMNS:
            row[col] = obj.properties.get(col)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("object_id")

    ids, vectors = collection.export_vectors(vector_field)
    vector_df = pd.DataFrame(
        {vector_field: [np.asarray(v, dtype=np.float32) for v in vectors]},
        index=ids,
    )
    df = df.join(vector_df, how="left")

    missing = df[vector_field].isna().sum()
    if missing:
        print(f"Warning: {missing} rows are missing a '{vector_field}' embedding.")
    return df.reset_index()


def main() -> None:
    db = ObjectVectorDB(uri=DB_URI)
    if not db.has_collection(COLLECTION_NAME):
        raise SystemExit(
            f"No '{COLLECTION_NAME}' collection at {DB_URI}. Run `python seed.py` first."
        )
    collection = db.collection(COLLECTION_NAME)

    df = build_dataframe(collection, VECTOR_FIELD)
    print(f"Loaded {len(df)} objects; launching Spotlight...")
    print("In the UI: open the Similarity Map lens on")
    print(f"  '{VECTOR_FIELD}', color by 'category', and click a row to see its neighbors.")
    print("Close the browser tab and press Ctrl+C to stop the server.\n")

    spotlight.show(
        df,
        dtype={
            "thumbnail_url": spotlight.Image,
            "video_url": spotlight.Video,
            VECTOR_FIELD: spotlight.Embedding,
        },
        wait=True,
    )


if __name__ == "__main__":
    main()
