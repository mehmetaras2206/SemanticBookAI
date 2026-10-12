#!/bin/sh
# Build the corpus and the search index on first start, then serve the API.
set -e
if [ ! -f /srv/data/processed/.indexed ]; then
  echo "First start: downloading books and building the index (one-time, several minutes)…"
  python -m app.ingest all
  touch /srv/data/processed/.indexed
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
