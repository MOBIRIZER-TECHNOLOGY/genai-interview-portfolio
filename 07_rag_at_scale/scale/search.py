"""
Two-stage search over the memmapped index: binary scan, then int8 rescore.

    from scale.search import ScaleIndex
    idx = ScaleIndex.open("C:/genai-data/index")
    hits = idx.search("how do warehouse robots avoid collisions?", k=10)

## The two stages, and why the split exists

**Stage 1 — binary.** The whole binary index is `n x dim/8` bytes: at 143 M
vectors that is 6.9 GB, which fits in RAM. Hamming distance is XOR plus a
popcount lookup, so a full scan is memory-bandwidth work over a small array
rather than 220 GB of float math. This stage decides *which* candidates matter,
and it only has to be good enough to get the true neighbours into a set of a few
hundred.

**Stage 2 — int8 rescore.** Read only the candidate rows from a 55 GB memmap on
disk (500 x 384 bytes = 192 KB), decode, dot product, sort. This stage decides
the *order*, and it is where the accuracy comes from.

Measured on real embeddings (`validate_quantization.py`): binary alone gives
recall@10 of 0.624; adding the int8 rescore at 500 candidates takes it to 0.985
with a quality ratio of 1.0000 against exact float32 search.

## Why the text is not in the index

Each chunk stores `(shard, row, char_start, char_end)` — 32 bytes — not its text.
Storing the text would re-materialise the 200 GB corpus alongside the vectors.
Text is read back from the parquet only for the handful of chunks that reach an
answer, which is a few milliseconds for 10 rows and saves 200 GB.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .quantize import Int8Calibration, POPCOUNT, binary_encode, int8_decode


@dataclass
class Hit:
    rank: int
    chunk_id: int
    score: float
    hamming: int
    shard: int
    row: int
    char_start: int
    char_end: int
    text: str | None = None


class ScaleIndex:
    """Memmapped binary + int8 index with lazy text fetch."""

    def __init__(self, path: Path, binary: np.ndarray, int8: np.ndarray,
                 coords: np.ndarray, cal: Int8Calibration, manifest: dict):
        self.path = path
        self.binary = binary
        self.int8 = int8
        self.coords = coords
        self.cal = cal
        self.manifest = manifest
        self.dim = manifest["dim"]
        self.n = len(coords)
        self._model = None

    # ------------------------------------------------------------- open

    @classmethod
    def open(cls, path: str | Path, load_binary_to_ram: bool = True) -> "ScaleIndex":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        dim = manifest["dim"]
        cal = Int8Calibration.load(path / "int8_calib.json")

        # Trust the MANIFEST for the row count, not the file length.
        #
        # The data files are append-only and the manifest is the commit point,
        # so a file can legitimately be longer than the manifest claims -- those
        # trailing rows are uncommitted work from an interrupted build. Sizing
        # the index from the file length silently reads them, which is how an
        # earlier benchmark run "successfully" measured 3.4M rows against a
        # manifest that said zero.
        n = int(manifest.get("n_chunks", 0))
        coords_all = np.memmap(path / "coords.i64", dtype=np.int64, mode="r").reshape(-1, 4)
        on_disk = len(coords_all)
        if n == 0:
            raise RuntimeError(
                f"index at {path} has no committed chunks "
                f"({on_disk:,} uncommitted rows on disk). Run build_index.py first."
            )
        if on_disk < n:
            raise RuntimeError(
                f"index is shorter than the manifest claims ({on_disk:,} < {n:,}); "
                "index and manifest disagree, rebuild required"
            )
        if on_disk > n:
            print(f"  note: ignoring {on_disk - n:,} uncommitted rows past the last commit")
        coords = coords_all[:n]
        binary = np.memmap(path / "binary.u8", dtype=np.uint8, mode="r",
                           shape=(n, dim // 8))
        # The binary index is scanned in full on every query, so it belongs in
        # RAM. The int8 array is touched a few hundred rows at a time and stays
        # a memmap -- the OS page cache handles it better than we would.
        if load_binary_to_ram:
            binary = np.ascontiguousarray(binary)
        int8 = np.memmap(path / "int8.i8", dtype=np.int8, mode="r", shape=(n, dim))
        return cls(path, binary, int8, coords, cal, manifest)

    # ----------------------------------------------------------- encode

    def _embedder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.manifest["model"], device="cuda")
            self._model.half()
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        # BGE is asymmetric: the instruction prefix belongs on queries only.
        prefix = "Represent this sentence for searching relevant passages: "
        v = self._embedder().encode([prefix + query], convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
        return v[0].astype(np.float32)

    # ----------------------------------------------------------- search

    def search(self, query: str | np.ndarray, k: int = 10, candidates: int = 500,
               fetch_text: bool = True) -> list[Hit]:
        vec = self.encode_query(query) if isinstance(query, str) else query.astype(np.float32)
        return self.search_vector(vec, k=k, candidates=candidates, fetch_text=fetch_text)

    def search_vector(self, vec: np.ndarray, k: int = 10, candidates: int = 500,
                      fetch_text: bool = True) -> list[Hit]:
        cand_idx, cand_ham = self._binary_stage(vec, candidates)
        order, scores = self._rescore_stage(vec, cand_idx, k)

        hits = []
        for rank, local in enumerate(order):
            gid = int(cand_idx[local])
            s, r, a, b = (int(x) for x in self.coords[gid])
            hits.append(Hit(rank=rank, chunk_id=gid, score=float(scores[rank]),
                            hamming=int(cand_ham[local]), shard=s, row=r,
                            char_start=a, char_end=b))
        if fetch_text:
            self.attach_text(hits)
        return hits

    def _binary_stage(self, vec: np.ndarray, candidates: int) -> tuple[np.ndarray, np.ndarray]:
        q = binary_encode(vec)[0]
        # popcount via a 256-entry LUT: far faster in numpy than bit arithmetic
        d = POPCOUNT[np.bitwise_xor(self.binary, q)].sum(axis=1)
        c = min(candidates, self.n)
        part = np.argpartition(d, c - 1)[:c]      # O(n), no full sort
        return part, d[part]

    def _rescore_stage(self, vec: np.ndarray, cand_idx: np.ndarray,
                       k: int) -> tuple[np.ndarray, np.ndarray]:
        # np.sort makes the memmap reads sequential, which matters on disk
        rows = self.int8[np.sort(cand_idx)]
        order_map = np.argsort(cand_idx)
        vecs = int8_decode(rows, self.cal)
        scores = np.empty(len(cand_idx), dtype=np.float32)
        scores[order_map] = vecs @ vec
        k = min(k, len(scores))
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return order, scores[order]

    # ------------------------------------------------------- text fetch

    @lru_cache(maxsize=8)
    def _shard_path(self, shard: int) -> Path | None:
        shards = sorted(Path(self.manifest.get("cache", "C:/genai-data/hf")).rglob("*.parquet"))
        return shards[shard] if shard < len(shards) else None

    def attach_text(self, hits: list[Hit]) -> None:
        """Read the source text for just these hits, one parquet touch per shard."""
        import pyarrow.parquet as pq

        by_shard: dict[int, list[Hit]] = {}
        for h in hits:
            by_shard.setdefault(h.shard, []).append(h)

        for shard, group in by_shard.items():
            path = self._shard_path(shard)
            if path is None or not path.exists():
                continue
            wanted = {h.row for h in group}
            pf = pq.ParquetFile(path)
            seen = 0
            texts: dict[int, str] = {}
            for batch in pf.iter_batches(batch_size=20000, columns=["text"]):
                lo, hi = seen, seen + batch.num_rows
                for r in wanted:
                    if lo <= r < hi:
                        texts[r] = batch.column("text")[r - lo].as_py()
                seen = hi
                if len(texts) == len(wanted):
                    break
            for h in group:
                doc = texts.get(h.row)
                if doc:
                    h.text = doc[h.char_start:h.char_end]

    # ------------------------------------------------------------ stats

    def stats(self) -> dict:
        return {
            "n_chunks": self.n,
            "dim": self.dim,
            "model": self.manifest["model"],
            "text_gb": round(self.manifest.get("bytes_text", 0) / 1e9, 2),
            "binary_gb": round(self.n * self.dim / 8 / 1e9, 3),
            "int8_gb": round(self.n * self.dim / 1e9, 2),
            "float32_avoided_gb": round(self.n * self.dim * 4 / 1e9, 2),
            "shards_indexed": len(self.manifest.get("shards_done", [])),
        }
