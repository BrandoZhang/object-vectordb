# Spotlight YouTube demo

End-to-end example that wires `object_vectordb` to [Renumics Spotlight] so you
can explore an embedding space in the browser and look up nearest-neighbor
objects — without building a bespoke dashboard.

[Renumics Spotlight]: https://github.com/Renumics/spotlight

## What it shows

A tiny synthetic dataset of ~60 YouTube-style video objects across six
categories (`cats`, `dogs`, `music`, `cooking`, `gaming`, `nature`). Each object
carries four annotation fields:

| field            | meaning                                             |
| ---------------- | --------------------------------------------------- |
| `title`          | video title                                         |
| `subtitle`       | snippet of the spoken-script (human speech)         |
| `audio_caption`  | description of the non-speech soundscape            |
| `thumbnail_url`  | picsum.photos placeholder                           |
| `video_url`      | a public CC-BY sample MP4 (cycled across rows)      |

A single vector field `audio_caption_embedding` (384-dim, from
`all-MiniLM-L6-v2`) is registered on the collection. The embedding is the only
signal used for search and visualization, which covers both scenarios the demo
targets:

1. **Text-to-neighbors query.** `python query.py "meow"` embeds the query with
   the same model, runs `Collection.search` against
   `audio_caption_embedding`, and prints the top-K matches — cat-category rows
   should dominate.
2. **Embedding-space exploration.** `python visualize.py` materializes the
   collection into a pandas DataFrame, hands it to Spotlight, and Spotlight's
   built-in Similarity Map lens (UMAP / PCA in the browser) lets you see the
   per-category clusters and pick any row to inspect its neighbors, with the
   thumbnail and video rendered in the detail panel.

## Install

From the repo root:

```bash
uv sync --extra spotlight
```

That pulls `renumics-spotlight`, `sentence-transformers`, and `pandas` on top of
the normal runtime deps. The first run of `seed.py` / `query.py` will download
the MiniLM model (~80 MB) into the Hugging Face cache.

## Run

```bash
cd examples/spotlight_youtube_demo

# 1. Build the on-disk collection under ./data/youtube.lance.
python seed.py

# 2. Top-K query on the audio_caption embedding.
python query.py "meow"
python query.py "guitar solo" --k 5
python query.py "thunderstorm" --k 3 --category nature

# 3. Launch the Spotlight viewer (opens http://localhost:<port> in your browser).
python visualize.py
```

### What you should see

`query.py "meow"` — top hits are cat-category rows whose `audio_caption` is
about meowing / purring. `query.py "guitar solo"` — mostly music rows. If that
doesn't happen, the MiniLM model likely didn't download; re-run `seed.py`.

`visualize.py` — Spotlight opens in your browser with the full table. Switch
the main view to **Similarity Map**, pick `audio_caption_embedding` as the
column, and color by `category`. You should see the six categories form
visible, mostly-disjoint clusters. Click any point to see its neighbors and
the video/thumbnail in the detail panel.

## Swapping in real data

Replace `sample_videos.py` with your own list of dicts of the same shape and
rerun `seed.py`. If your real data has more than a few thousand rows, enable an
IVF index once at the end of `seed.py`:

```python
collection.create_index(
    VECTOR_FIELD,
    index_type="IVF_PQ",
    metric="cosine",           # must match what query.py passes to search()
    num_partitions=256,
    num_sub_vectors=16,
)
```

See the project-level README for the full IVF/HNSW options and the
metric-mismatch caveats.

## Files

- `sample_videos.py` — synthetic dataset (no library deps).
- `seed.py` — builds the collection, encodes captions, inserts objects.
- `query.py` — CLI nearest-neighbor search for requirement 1.
- `visualize.py` — `object_vectordb` → pandas → Spotlight bridge for
  requirement 2. `build_dataframe` is the only non-trivial piece and is a
  candidate to promote into a reusable
  `object_vectordb.integrations.spotlight` module later.
