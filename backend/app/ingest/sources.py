"""Fetching from Gutendex (Project Gutenberg catalog), Gutenberg mirrors and Open Library, cached on disk."""

import json
import re
import time
from pathlib import Path

import httpx

GUTENDEX = "https://gutendex.com/books/"
OPENLIBRARY = "https://openlibrary.org"
USER_AGENT = "SemanticBookAI/1.0 (https://github.com/mehmetaras2206/SemanticBookAI)"


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(20, connect=10), follow_redirects=True)


def _get_json(client: httpx.Client, url: str, params: dict | None = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_gutendex(ids: list[int], cache: Path) -> dict[int, dict]:
    """Catalog records for the given Gutenberg IDs (title, authors, languages, formats, subjects)."""
    records: dict[int, dict] = {}
    if cache.exists():
        records = {int(k): v for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
    missing = [i for i in ids if i not in records]
    # Single-ID requests: multi-ID queries are very slow on gutendex.com (~10 s per ID).
    with _client() as client:
        for gid in missing:
            records[gid] = _get_json(client, f"{GUTENDEX}{gid}/")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return records


def download_text(record: dict, raw_dir: Path) -> str:
    path = raw_dir / f"{record['id']}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    formats = record["formats"]
    url = next(
        (u for k, u in formats.items() if k.startswith("text/plain") and "utf-8" in k),
        None,
    ) or next(u for k, u in formats.items() if k.startswith("text/plain"))
    with _client() as client:
        for attempt in range(4):
            try:
                r = client.get(url)
                r.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    text = r.content.decode("utf-8", errors="replace")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    time.sleep(1)  # be polite to gutenberg.org
    return text


def fetch_openlibrary(title: str, author: str, cache_dir: Path, key: str) -> dict:
    """Best-effort Open Library lookup: description and subjects of the matching work."""
    path = cache_dir / f"ol_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result: dict = {}
    try:
        with _client() as client:
            surname = author.split()[-1] if author else ""
            data = _get_json(
                client,
                f"{OPENLIBRARY}/search.json",
                {"title": title, "author": surname, "limit": 1, "fields": "key,title,subject,first_publish_year"},
            )
            if data.get("docs"):
                doc = data["docs"][0]
                result = {"ol_key": doc["key"], "subjects": doc.get("subject", [])[:15]}
                work = _get_json(client, f"{OPENLIBRARY}{doc['key']}.json")
                desc = work.get("description")
                if isinstance(desc, dict):
                    desc = desc.get("value")
                if desc:
                    result["description"] = _clean_description(desc)
    except httpx.HTTPError:
        result = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.5)
    return result


def _clean_description(desc: str) -> str:
    desc = re.split(r"\n-{3,}|\n\*{3,}|\(\[source\]", desc)[0]
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)
    return re.sub(r"\s+", " ", desc).strip()[:1500]
