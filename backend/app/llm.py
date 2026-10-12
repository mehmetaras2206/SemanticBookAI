"""Streaming client for a local LLM served by Ollama."""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings


class LLMUnavailable(Exception):
    pass


async def stream_chat(messages: list[dict], model: str | None = None) -> AsyncIterator[str]:
    s = get_settings()
    body = {
        "model": model or s.llm_model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": s.llm_temperature, "num_ctx": s.llm_num_ctx},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5)) as client:
            async with client.stream("POST", f"{s.ollama_url}/api/chat", json=body) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode(errors="replace")[:300]
                    raise LLMUnavailable(f"Ollama returned {r.status_code}: {detail}")
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMUnavailable(chunk["error"])
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"Ollama not reachable at {s.ollama_url}: {e}") from e


async def complete(messages: list[dict], model: str | None = None) -> str:
    return "".join([t async for t in stream_chat(messages, model)])


async def status() -> dict:
    s = get_settings()
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{s.ollama_url}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {"reachable": True, "model": s.llm_model, "model_ready": s.llm_model in models}
    except (httpx.HTTPError, ValueError):
        return {"reachable": False, "model": s.llm_model, "model_ready": False}
