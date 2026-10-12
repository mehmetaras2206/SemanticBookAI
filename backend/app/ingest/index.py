"""Stage 2: embed books and passages (dense + BM25) and load them into Qdrant."""

import json
import time

from qdrant_client import models

from app.config import Settings
from app.store import BOOKS, DENSE, PASSAGES, SPARSE, embed_documents, get_client, point_id, recreate_collection

BATCH = 256


def book_search_text(book: dict) -> str:
    return "\n".join([
        f"{book['title']} by {book['author']}",
        book["full_title"],
        ", ".join(book["subjects"]),
        book["description"],
    ])


def _upsert(collection: str, rows: list[dict], texts: list[str], keys: list[str]) -> None:
    client = get_client()
    for start in range(0, len(rows), BATCH):
        dense, sparse = embed_documents(texts[start:start + BATCH])
        client.upsert(collection, points=[
            models.PointStruct(id=point_id(key), vector={DENSE: d, SPARSE: s}, payload=row)
            for key, row, d, s in zip(keys[start:start + BATCH], rows[start:start + BATCH], dense, sparse)
        ])


def build_index(settings: Settings) -> tuple[int, int]:
    client = get_client()
    books = [json.loads(line) for line in (settings.processed_dir / "books.jsonl").open(encoding="utf-8")]
    eras = {b["book_id"]: b["era"] for b in books}

    recreate_collection(client, BOOKS, settings)
    book_rows = [{k: v for k, v in b.items() if k != "chapters"} for b in books]
    _upsert(BOOKS, book_rows, [book_search_text(b) for b in books], [b["book_id"] for b in books])

    recreate_collection(client, PASSAGES, settings)
    passages = [json.loads(line) for line in (settings.processed_dir / "passages.jsonl").open(encoding="utf-8")]
    for p in passages:
        p["era"] = eras[p["book_id"]]
    t0 = time.time()
    for start in range(0, len(passages), 4096):
        part = passages[start:start + 4096]
        _upsert(
            PASSAGES, part,
            # Title and chapter are prepended so passages stay findable by book context.
            [f"{p['title']} – {p['chapter_title']}\n{p['text']}" for p in part],
            [f"{p['book_id']}-{p['chapter_idx']}-{p['passage_idx']}" for p in part],
        )
        done = start + len(part)
        print(f"  indexed {done}/{len(passages)} passages ({done / (time.time() - t0):.0f}/s)", flush=True)
    return len(books), len(passages)
