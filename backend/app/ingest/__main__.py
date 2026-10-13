"""python -m app.ingest [prepare|index [--resume]|all|check|export [path]|restore <path-or-url>]"""

import sys
import time
from pathlib import Path

from app.config import get_settings
from app.ingest.index import build_index
from app.ingest.prepare import prepare


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    settings = get_settings()
    if stage == "check":
        # exit code 0 when the passages collection holds every processed passage
        from app.store import PASSAGES, get_client
        client = get_client()
        path = settings.processed_dir / "passages.jsonl"
        expected = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 1
        ok = ((settings.processed_dir / "books.jsonl").exists()
              and client.collection_exists(PASSAGES) and client.count(PASSAGES).count >= expected)
        sys.exit(0 if ok else 1)
    t0 = time.time()
    if stage == "export":
        from app.ingest.snapshot import export_index
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else settings.data_dir / "index.tar.gz"
        print(f"wrote {export_index(settings, out)}")
    if stage == "restore":
        from app.ingest.snapshot import restore_index
        restore_index(settings, sys.argv[2])
    if stage in ("prepare", "all"):
        n_books, n_passages = prepare(settings)
        print(f"prepared {n_books} books, {n_passages} passages")
    if stage in ("index", "all"):
        n_books, n_passages = build_index(settings, resume="--resume" in sys.argv)
        print(f"indexed {n_books} books, {n_passages} passages")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
