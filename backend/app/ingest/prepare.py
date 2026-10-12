"""Stage 1: download and process the curated corpus into books.jsonl and passages.jsonl."""

import json
import re
from pathlib import Path

import yaml

from app.config import Settings
from app.ingest.sources import download_text, fetch_gutendex, fetch_openlibrary
from app.ingest.text import chunk_text, split_chapters, strip_gutenberg


def era_for_year(year: int) -> str:
    if year < 1500:
        return "Antiquity & Middle Ages"
    if year < 1800:
        return "1500-1799"
    if year < 1850:
        return "1800-1849"
    if year < 1900:
        return "1850-1899"
    return "1900-1930"


def display_author(name: str | None) -> str:
    """'Carroll, Lewis' -> 'Lewis Carroll'; 'Sacher-Masoch, Leopold, Ritter von' -> 'Leopold von Sacher-Masoch'."""
    if not name:
        return "Unknown"
    name = re.sub(r"\s*\([^)]*\)", "", name)
    parts = [p.strip() for p in name.split(",")]
    if len(parts) == 1:
        return parts[0]
    if " of " in parts[1] or parts[1].startswith("Emperor"):
        return parts[0]
    particle = ""
    if len(parts) >= 3 and parts[2].split()[-1] in {"von", "van", "de"}:
        particle = parts[2].split()[-1] + " "
    return f"{parts[1]} {particle}{parts[0]}"


def short_title(title: str) -> str:
    title = title.split("$b")[0]
    title = re.split(r"\s*[:;]\s*", title)[0]
    return title.strip(" .,:;")


def load_corpus(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["books"]


def prepare(settings: Settings) -> tuple[int, int]:
    corpus = load_corpus(settings.data_dir / "books.yaml")
    years = {b["gutenberg_id"]: b["year"] for b in corpus}
    records = fetch_gutendex(list(years), settings.raw_dir / "gutendex.json")

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    books_out = (settings.processed_dir / "books.jsonl").open("w", encoding="utf-8")
    passages_out = (settings.processed_dir / "passages.jsonl").open("w", encoding="utf-8")
    n_passages = 0
    with books_out, passages_out:
        for gid, year in years.items():
            rec = records[gid]
            if rec.get("media_type") != "Text":
                raise SystemExit(f"Gutenberg #{gid} is not a text edition (media_type={rec.get('media_type')})")
            title = short_title(rec["title"])
            author = display_author(rec["authors"][0]["name"] if rec["authors"] else None)
            language = rec["languages"][0]

            text = strip_gutenberg(download_text(rec, settings.raw_dir))
            chapters = split_chapters(text)

            ol = fetch_openlibrary(title, author, settings.raw_dir / "openlibrary", str(gid))
            summaries = rec.get("summaries") or []
            description = ol.get("description") or (summaries[0] if summaries else "")
            subjects = sorted({
                *(s.split(" -- ")[0] for s in rec.get("subjects", [])),
                *(b.replace("Category: ", "") for b in rec.get("bookshelves", [])),
            })

            book_id = f"pg{gid}"
            book_passages = 0
            chapter_list = []
            for ch in chapters:
                chunks = chunk_text(ch.text)
                if not chunks:
                    continue
                chapter_list.append({"idx": ch.idx, "title": ch.title, "passages": len(chunks)})
                for p_idx, chunk in enumerate(chunks):
                    passages_out.write(json.dumps({
                        "book_id": book_id,
                        "title": title,
                        "author": author,
                        "language": language,
                        "chapter_idx": ch.idx,
                        "chapter_title": ch.title,
                        "passage_idx": p_idx,
                        "text": chunk,
                    }, ensure_ascii=False) + "\n")
                book_passages += len(chunks)

            books_out.write(json.dumps({
                "book_id": book_id,
                "gutenberg_id": gid,
                "title": title,
                "full_title": rec["title"].replace("$b", "").replace("  ", " "),
                "author": author,
                "language": language,
                "year": year,
                "era": era_for_year(year),
                "subjects": subjects[:12],
                "description": description,
                "description_source": "Open Library" if ol.get("description") else "Project Gutenberg",
                "gutenberg_url": f"https://www.gutenberg.org/ebooks/{gid}",
                "chapters": chapter_list,
                "num_passages": book_passages,
            }, ensure_ascii=False) + "\n")
            n_passages += book_passages
            print(f"  {book_id:8} {language} {len(chapter_list):3} chapters {book_passages:5} passages  {title}")
    return len(years), n_passages
