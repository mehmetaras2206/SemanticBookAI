"""Portable index archive: export the Qdrant collections (vectors + payloads) and restore them elsewhere.

Building the index needs ~43k embedding calls, which takes minutes on a GPU but hours on a CPU. The archive
lets `docker compose up` start from a prebuilt index instead (published as a GitHub release asset).
"""

import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx
import numpy as np
from qdrant_client import models

from app.config import Settings
from app.store import BOOKS, DENSE, PASSAGES, SPARSE, get_client, recreate_collection

COLLECTIONS = (BOOKS, PASSAGES)
UPSERT_BATCH = 512


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def export_index(settings: Settings, out: Path) -> Path:
    client = get_client()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name in ("books.jsonl", "passages.jsonl"):
            tar.add(settings.processed_dir / name, arcname=f"processed/{name}")
        for coll in COLLECTIONS:
            ids, dense, sparse, payloads = [], [], [], []
            offset = None
            while True:
                points, offset = client.scroll(coll, limit=1024, offset=offset, with_vectors=True, with_payload=True)
                for p in points:
                    ids.append(str(p.id))
                    dense.append(p.vector[DENSE])
                    sv = p.vector[SPARSE]
                    sparse.append({"i": sv.indices, "v": [round(x, 4) for x in sv.values]})
                    payloads.append(p.payload)
                if offset is None:
                    break
            buf = io.BytesIO()
            np.save(buf, np.asarray(dense, dtype=np.float16))
            _add_bytes(tar, f"{coll}/dense.npy", buf.getvalue())
            rows = (json.dumps({"id": i, "sparse": s, "payload": p}, ensure_ascii=False)
                    for i, s, p in zip(ids, sparse, payloads))
            _add_bytes(tar, f"{coll}/points.jsonl", "\n".join(rows).encode("utf-8"))
            print(f"  exported {coll}: {len(ids)} points")
    return out


def restore_index(settings: Settings, source: str) -> None:
    """Restore from a local archive path or an http(s) URL."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "index.tar.gz"
        if source.startswith(("http://", "https://")):
            print(f"  downloading {source}", flush=True)
            with httpx.stream("GET", source, follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as r:
                r.raise_for_status()
                with archive.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
        else:
            shutil.copy(source, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")

        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        for name in ("books.jsonl", "passages.jsonl"):
            shutil.copy(Path(tmp) / "processed" / name, settings.processed_dir / name)

        client = get_client()
        for coll in COLLECTIONS:
            dense = np.load(Path(tmp) / coll / "dense.npy").astype(np.float32)
            rows = [json.loads(line) for line in (Path(tmp) / coll / "points.jsonl").open(encoding="utf-8")]
            recreate_collection(client, coll, dim=dense.shape[1])
            for start in range(0, len(rows), UPSERT_BATCH):
                client.upsert(coll, points=[
                    models.PointStruct(
                        id=row["id"],
                        vector={DENSE: vec.tolist(),
                                SPARSE: models.SparseVector(indices=row["sparse"]["i"], values=row["sparse"]["v"])},
                        payload=row["payload"],
                    )
                    for row, vec in zip(rows[start:start + UPSERT_BATCH], dense[start:start + UPSERT_BATCH])
                ])
            print(f"  restored {coll}: {len(rows)} points", flush=True)
