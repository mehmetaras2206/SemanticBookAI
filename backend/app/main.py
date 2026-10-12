import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import llm
from app.chat import answer_stream
from app.config import get_settings
from app.store import BOOKS, PASSAGES, build_filter, get_client, search

app = FastAPI(title="SemanticBookAI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health() -> dict:
    client = get_client()
    counts = {}
    for name in (BOOKS, PASSAGES):
        counts[name] = client.count(name).count if client.collection_exists(name) else 0
    return {"status": "ok", "index": counts, "llm": await llm.status()}


@app.get("/api/search")
def search_books(
    q: str = Query(..., min_length=1, max_length=500),
    language: Literal["en", "de"] | None = None,
    era: str | None = None,
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    hits = search(BOOKS, q, mode=mode, limit=limit, query_filter=build_filter(language=language, era=era))
    return {
        "query": q,
        "mode": mode,
        "results": [
            {**{k: h.payload.get(k) for k in (
                "book_id", "title", "author", "language", "year", "era", "subjects", "description",
                "gutenberg_url", "num_passages")},
             "score": round(h.score, 4)}
            for h in hits
        ],
    }


@app.get("/api/passages")
def search_passages(
    q: str = Query(..., min_length=1, max_length=500),
    book_id: str | None = None,
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    hits = search(PASSAGES, q, mode=mode, limit=limit, query_filter=build_filter(book_id=book_id))
    return {"query": q, "mode": mode, "results": [{**h.payload, "score": round(h.score, 4)} for h in hits]}


@app.get("/api/books/{book_id}")
def book_detail(book_id: str) -> dict:
    s = get_settings()
    with (s.processed_dir / "books.jsonl").open(encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            if book["book_id"] == book_id:
                return book
    raise HTTPException(404, "book not found")


@app.get("/api/facets")
def facets() -> dict:
    s = get_settings()
    eras: dict[str, int] = {}
    languages: dict[str, int] = {}
    with (s.processed_dir / "books.jsonl").open(encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            eras[b["era"]] = eras.get(b["era"], 0) + 1
            languages[b["language"]] = languages.get(b["language"], 0) + 1
    order = ["Antiquity & Middle Ages", "1500-1799", "1800-1849", "1850-1899", "1900-1930"]
    return {"eras": [{"name": e, "count": eras[e]} for e in order if e in eras],
            "languages": [{"name": k, "count": v} for k, v in sorted(languages.items())]}


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=4000)
    book_id: str | None = None
    history: list[ChatTurn] = []


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        async for ev in answer_stream(req.question, req.book_id, [t.model_dump() for t in req.history]):
            yield f"event: {ev['event']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
