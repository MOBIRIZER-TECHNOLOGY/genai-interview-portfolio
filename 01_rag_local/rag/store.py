"""
On-disk vector store: FAISS index + a sidecar JSONL of chunk metadata.

Deliberately not a hosted vector DB. For a corpus under ~1M chunks, a flat
FAISS index on one machine beats a network round-trip to a managed service on
both latency and honesty: exact search, no approximation error to explain away.

Index choice:
  IndexFlatIP  - exact, brute force. O(N) per query but N is tiny and it is
                 fully deterministic, which matters when you are measuring
                 retrieval quality. Use this until you can prove you need more.
  IndexHNSWFlat - the upgrade path past ~1M vectors. Approximate, so recall
                 becomes a tunable (efSearch) rather than a guarantee.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    rank: int
    text: str
    body: str
    source: str
    breadcrumb: str


class VectorStore:
    def __init__(self, index, records: list[dict], model_name: str):
        self.index = index
        self.records = records
        self.model_name = model_name

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, vectors: np.ndarray, records: list[dict], model_name: str) -> "VectorStore":
        import faiss

        assert len(vectors) == len(records), "vector/record count mismatch"
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, records, model_name)

    # ------------------------------------------------------------ persist

    def save(self, dirpath: Path) -> None:
        import faiss

        dirpath.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dirpath / "index.faiss"))
        with open(dirpath / "chunks.jsonl", "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (dirpath / "meta.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "n_chunks": len(self.records),
                    "dim": self.index.d,
                    "index_type": type(self.index).__name__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dirpath: Path) -> "VectorStore":
        import faiss

        meta = json.loads((dirpath / "meta.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(dirpath / "index.faiss"))
        records = [
            json.loads(line)
            for line in (dirpath / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(index, records, meta["model_name"])

    # ------------------------------------------------------------- search

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[SearchHit]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        k = min(k, len(self.records))
        scores, idxs = self.index.search(query_vec, k)
        hits: list[SearchHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            if idx < 0:
                continue
            r = self.records[idx]
            hits.append(
                SearchHit(
                    chunk_id=r["id"],
                    score=float(score),
                    rank=rank,
                    text=r["text"],
                    body=r["body"],
                    source=r["source"],
                    breadcrumb=r["breadcrumb"],
                )
            )
        return hits

    def __len__(self) -> int:
        return len(self.records)
