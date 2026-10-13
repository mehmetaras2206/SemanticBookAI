"""Stage 2: embed books and passages (dense + BM25) and load them into Qdrant."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import models

from app.config import Settings
from app.store import (
    BOOKS, DENSE, PASSAGES, SPARSE, embed_dense, embed_sparse, get_client, point_id, recreate_collection,
    unload_embedding_model,
)

BATCH = 128
DENSE_WORKERS = 2  # keeps the GPU busy while BM25 and the upsert of the previous batch run


def book_search_text(book: dict) -> str:
    return "\n".join([
        f"{book['title']} by {book['author']}",
        book["full_title"],
        ", ".join(book["subjects"]),
        book["description"],
    ])


def passage_search_text(p: dict) -> str:
    # Title and chapter are prepended so passages stay findable by book context.
    return f"{p['title']} – {p['chapter_title']}\n{p['text']}"


def _existing_ids(collection: str) -> set[str]:
    client = get_client()
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=4096, offset=offset, with_payload=False, with_vectors=False)
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


def _upsert(collection: str, rows: list[dict], texts: list[str], keys: list[str], label: str) -> None:
    client = get_client()
    batches = [(rows[i:i + BATCH], texts[i:i + BATCH], keys[i:i + BATCH]) for i in range(0, len(rows), BATCH)]
    t0, done = time.time(), 0
    with ThreadPoolExecutor(DENSE_WORKERS) as pool:
        # bounded look-ahead: at most DENSE_WORKERS + 1 batches of vectors are held in memory
        pending = {i: pool.submit(embed_dense, batches[i][1]) for i in range(min(DENSE_WORKERS + 1, len(batches)))}
        for n, (b_rows, b_texts, b_keys) in enumerate(batches):
            sparse = embed_sparse(b_texts)
            dense = pending.pop(n).result()
            nxt = n + DENSE_WORKERS + 1
            if nxt < len(batches):
                pending[nxt] = pool.submit(embed_dense, batches[nxt][1])
            client.upsert(collection, wait=False, points=[
                models.PointStruct(id=point_id(key), vector={DENSE: d, SPARSE: s}, payload=row)
                for key, row, d, s in zip(b_keys, b_rows, dense, sparse)
            ])
            done += len(b_rows)
            if done % 2048 < BATCH or done == len(rows):
                print(f"  {label}: {done}/{len(rows)} ({done / (time.time() - t0):.0f}/s)", flush=True)


def build_index(settings: Settings, resume: bool = False) -> tuple[int, int]:
    client = get_client()
    # A runner loaded for CPU query embeddings would be reused for the bulk run; unload it so Ollama uses the GPU.
    unload_embedding_model()
    books = [json.loads(line) for line in (settings.processed_dir / "books.jsonl").open(encoding="utf-8")]
    passages = [json.loads(line) for line in (settings.processed_dir / "passages.jsonl").open(encoding="utf-8")]
    eras = {b["book_id"]: b["era"] for b in books}
    for p in passages:
        p["era"] = eras[p["book_id"]]

    jobs = [
        (BOOKS, [{k: v for k, v in b.items() if k != "chapters"} for b in books],
         [book_search_text(b) for b in books], [b["book_id"] for b in books]),
        (PASSAGES, passages, [passage_search_text(p) for p in passages],
         [f"{p['book_id']}-{p['chapter_idx']}-{p['passage_idx']}" for p in passages]),
    ]
    for collection, rows, texts, keys in jobs:
        if resume and client.collection_exists(collection):
            existing = _existing_ids(collection)
            todo = [i for i, k in enumerate(keys) if point_id(k) not in existing]
            print(f"  {collection}: {len(existing)} present, {len(todo)} to index", flush=True)
            rows, texts, keys = [rows[i] for i in todo], [texts[i] for i in todo], [keys[i] for i in todo]
        else:
            recreate_collection(client, collection)
        _upsert(collection, rows, texts, keys, collection)
    return len(books), len(passages)
