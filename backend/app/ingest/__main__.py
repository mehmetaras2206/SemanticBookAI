"""python -m app.ingest [prepare|index|all]"""

import sys
import time

from app.config import get_settings
from app.ingest.index import build_index
from app.ingest.prepare import prepare


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    settings = get_settings()
    t0 = time.time()
    if stage in ("prepare", "all"):
        n_books, n_passages = prepare(settings)
        print(f"prepared {n_books} books, {n_passages} passages")
    if stage in ("index", "all"):
        n_books, n_passages = build_index(settings)
        print(f"indexed {n_books} books, {n_passages} passages")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
