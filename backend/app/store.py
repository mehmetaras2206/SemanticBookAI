"""Qdrant access: collection layout, embeddings and hybrid (dense + BM25, RRF) retrieval."""

import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import httpx
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from app.config import get_settings

BOOKS = "books"
PASSAGES = "passages"
DENSE = "dense"
SPARSE = "bm25"

SearchMode = Literal["hybrid", "dense", "sparse"]


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, timeout=60)
    s.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(s.qdrant_path))


@lru_cache
def _ollama_http() -> httpx.Client:
    # one shared client: creating a client per call costs ~0.5 s (TLS context setup) on Windows
    return httpx.Client(base_url=get_settings().ollama_url, timeout=600)


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Dense embeddings from Ollama (runs on the GPU when available)."""
    s = get_settings()
    r = _ollama_http().post("/api/embed", json={"model": s.embed_model, "input": texts})
    r.raise_for_status()
    return r.json()["embeddings"]


@lru_cache
def dense_dim() -> int:
    return len(embed_dense(["dimension probe"])[0])


@lru_cache
def get_sparse() -> SparseTextEmbedding:
    return SparseTextEmbedding(get_settings().sparse_model)


def point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def recreate_collection(client: QdrantClient, name: str, dim: int | None = None) -> None:
    dim = dim or dense_dim()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field, schema in [
        ("language", models.PayloadSchemaType.KEYWORD),
        ("era", models.PayloadSchemaType.KEYWORD),
        ("book_id", models.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(name, field, schema)


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    return [
        models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in get_sparse().embed(texts, batch_size=64)
    ]


def embed_query(text: str) -> tuple[list[float], models.SparseVector]:
    dense = embed_dense([text])[0]
    sp = next(iter(get_sparse().query_embed(text)))
    return dense, models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())


@dataclass
class Hit:
    id: str
    score: float
    payload: dict
    dense_score: float | None = None


def build_filter(**conditions: str | None) -> models.Filter | None:
    must = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in conditions.items()
        if v
    ]
    return models.Filter(must=must) if must else None


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def search(
    collection: str,
    query: str,
    mode: SearchMode = "hybrid",
    limit: int = 10,
    query_filter: models.Filter | None = None,
    prefetch_limit: int = 50,
) -> list[Hit]:
    """Dense, sparse (BM25) or hybrid retrieval. Hybrid fuses both rankings with Reciprocal Rank Fusion."""
    client = get_client()
    dense_vec, sparse_vec = embed_query(query)
    fetch = limit if mode != "hybrid" else prefetch_limit

    dense_points = sparse_points = []
    if mode in ("hybrid", "dense"):
        dense_points = client.query_points(collection, query=dense_vec, using=DENSE, limit=fetch,
                                           query_filter=query_filter, with_payload=True).points
    if mode in ("hybrid", "sparse"):
        sparse_points = client.query_points(collection, query=sparse_vec, using=SPARSE, limit=fetch,
                                            query_filter=query_filter, with_payload=True).points

    payloads = {str(p.id): p.payload or {} for p in [*dense_points, *sparse_points]}
    dense_scores = {str(p.id): p.score for p in dense_points}

    if mode == "dense":
        ranked = [(str(p.id), p.score) for p in dense_points]
    elif mode == "sparse":
        ranked = [(str(p.id), p.score) for p in sparse_points]
    else:
        ranked = rrf_fuse(
            [[str(p.id) for p in dense_points], [str(p.id) for p in sparse_points]],
            k=get_settings().rrf_k,
        )[:limit]

    # Dense cosine similarity is kept per hit: it is the calibrated signal the relevance guardrail needs.
    return [Hit(id=i, score=s, payload=payloads[i], dense_score=dense_scores.get(i)) for i, s in ranked]
