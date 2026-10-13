#!/bin/sh
# First start: fill the empty index from the prebuilt archive (INDEX_URL) or build it from the books.
set -e
if ! python -m app.ingest check; then
  if [ -n "$INDEX_URL" ] && python -m app.ingest restore "$INDEX_URL"; then
    echo "Index restored from $INDEX_URL"
  else
    echo "Building the index from the books (one-time; fast with a GPU, slow on CPU)…"
    python -m app.ingest all
  fi
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
