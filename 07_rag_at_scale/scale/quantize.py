"""
Binary and int8 quantisation of embeddings — the technique that makes
hundred-million-vector retrieval fit on one machine.

## The problem

143 M chunks x 384 dims x 4 bytes = **220 GB** of float32 vectors. That does not
fit in this machine's RAM, let alone its 16 GB of VRAM. Any exact search over it
is bound by reading 220 GB per query.

## The fix: a two-stage precision cascade

    binary (1 bit/dim)   48 bytes/vec ->   6.9 GB   fits in RAM, Hamming distance
    int8   (8 bits/dim)  384 bytes/vec ->  54.9 GB  on disk, memmap, rescore only
    float32 (reference)  1536 bytes/vec -> 219.6 GB never materialised at scale

Search runs in two passes:

1. **Binary search over everything.** Hamming distance is `popcount(a XOR b)` —
   a handful of CPU instructions over 48 bytes, and the whole index is in RAM.
   Retrieve a generous candidate set (say 1000).
2. **int8 rescore of just those candidates.** Read 1000 x 384 bytes = 384 KB from
   a memmap, compute exact-ish dot products, keep the top k.

You get ~32x the memory reduction and recall within a couple of points of exact
search, because binary search only has to be good enough to get the true
neighbours *into* the candidate set — the rescore fixes the ordering.

## Why binarising at zero works

The embedding model is trained to produce L2-normalised vectors whose *direction*
carries the meaning. Each dimension is roughly zero-centred, so `> 0` keeps the
sign — the dominant bit of information — and discards the magnitude. Empirically
this holds ~95% of retrieval quality on modern embedding models; it is what
Qdrant, Vespa and sentence-transformers all ship.

The int8 calibration is the part people get wrong: you cannot pick the scale
per-vector, because then two vectors' codes are not comparable. The range has to
be **global**, computed once from a sample, and then frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ binary


def binary_encode(vectors: np.ndarray) -> np.ndarray:
    """float32 [n, d] -> packed bits uint8 [n, d/8]. d must be a multiple of 8."""
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    return np.packbits(vectors > 0, axis=-1)


def hamming_search(query_codes: np.ndarray, corpus_codes: np.ndarray, k: int,
                   popcount: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force Hamming top-k. Returns (distances, indices), both [nq, k].

    Vectorised via a 256-entry popcount lookup table, which is far faster in
    numpy than any bit-twiddling expression and is what makes a pure-numpy
    implementation viable at tens of millions of vectors.
    """
    if query_codes.ndim == 1:
        query_codes = query_codes[None, :]
    if popcount is None:
        popcount = POPCOUNT

    nq = query_codes.shape[0]
    k = min(k, corpus_codes.shape[0])
    dists = np.empty((nq, k), dtype=np.int32)
    idxs = np.empty((nq, k), dtype=np.int64)

    for i in range(nq):
        # XOR broadcasts the query across the corpus, then popcount+sum per row
        d = popcount[np.bitwise_xor(corpus_codes, query_codes[i])].sum(axis=1)
        part = np.argpartition(d, k - 1)[:k]        # O(n), not a full sort
        order = part[np.argsort(d[part])]
        idxs[i] = order
        dists[i] = d[order]
    return dists, idxs


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# -------------------------------------------------------------------- int8


@dataclass
class Int8Calibration:
    """Global min/max per dimension, frozen after calibration."""
    mins: list[float]
    maxs: list[float]
    dim: int
    n_calibration: int

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Int8Calibration":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    @property
    def np_mins(self) -> np.ndarray:
        return np.asarray(self.mins, dtype=np.float32)

    @property
    def np_ranges(self) -> np.ndarray:
        r = np.asarray(self.maxs, dtype=np.float32) - self.np_mins
        return np.where(r == 0, 1.0, r)      # guard dead dimensions


def calibrate_int8(sample: np.ndarray, percentile: float = 99.9) -> Int8Calibration:
    """Fit the global int8 range from a sample of vectors.

    Uses a percentile rather than absolute min/max so a single outlier
    dimension cannot squash the resolution available to every other vector.
    """
    lo = np.percentile(sample, 100 - percentile, axis=0)
    hi = np.percentile(sample, percentile, axis=0)
    return Int8Calibration(lo.tolist(), hi.tolist(), sample.shape[1], len(sample))


def int8_encode(vectors: np.ndarray, cal: Int8Calibration) -> np.ndarray:
    """float32 [n, d] -> int8 [n, d] using the frozen global range."""
    scaled = (vectors - cal.np_mins) / cal.np_ranges      # -> ~[0, 1]
    return np.clip(np.round(scaled * 255.0) - 128.0, -128, 127).astype(np.int8)


def int8_decode(codes: np.ndarray, cal: Int8Calibration) -> np.ndarray:
    return ((codes.astype(np.float32) + 128.0) / 255.0) * cal.np_ranges + cal.np_mins


def rescore(query: np.ndarray, candidate_codes: np.ndarray,
            cal: Int8Calibration, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact-ish dot products against decoded int8 candidates. Returns (scores, local_idx)."""
    vecs = int8_decode(candidate_codes, cal)
    scores = vecs @ query.astype(np.float32)
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part])]
    return scores[order], order


# ------------------------------------------------------------------ report


def memory_report(n_vectors: int, dim: int) -> dict:
    """What each precision costs for a given corpus size."""
    f32 = n_vectors * dim * 4
    f16 = n_vectors * dim * 2
    i8 = n_vectors * dim
    binr = n_vectors * dim // 8
    return {
        "n_vectors": n_vectors,
        "dim": dim,
        "float32_gb": round(f32 / 1e9, 2),
        "float16_gb": round(f16 / 1e9, 2),
        "int8_gb": round(i8 / 1e9, 2),
        "binary_gb": round(binr / 1e9, 3),
        # an empty index is a legitimate state -- a run that aborted before its
        # first commit leaves n_chunks=0 with dim already set
        "binary_reduction": f"{f32 / binr:.0f}x" if binr else "n/a",
    }
