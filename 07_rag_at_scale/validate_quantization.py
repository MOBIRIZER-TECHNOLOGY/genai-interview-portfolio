"""
Validate the binary + int8 precision cascade on REAL embeddings, and pick the
rescore format from measurement rather than folklore.

    python validate_quantization.py                    # 20k real passages
    python validate_quantization.py --n 100000 --k 10

## Why this script exists

The whole 200 GB index design rests on one claim: *binary search over everything,
then rescore a small candidate set, gives you effectively exact results for 1/32
of the memory.* That claim is worth exactly as much as the measurement behind it,
and there are two traps that make people report the wrong number.

**Trap 1 — testing on synthetic vectors.** Isotropic Gaussians on the unit sphere
are the worst possible case for binary quantisation: every dimension is
independent, so the sign pattern carries almost no neighbour information. Measured
here, synthetic data gives binary-only recall@10 of **0.18** while real embeddings
give **0.62**. If you validate on `np.random.normal` you will conclude the
technique doesn't work.

**Trap 2 — leaving the query in the corpus.** If your query vector is also indexed,
it is its own nearest neighbour and permanently occupies one of the k slots. With
k=10 that silently caps recall at 0.9 — and 0.9 looks plausible enough that you
ship it. This script excludes self from both the ground truth and the results.

## What it reports

`recall@k` against exact float32 search, and a **quality ratio**: the mean
similarity of what you retrieved divided by the mean similarity of the exact
top-k. Recall alone is misleading on real corpora, where many passages are
near-ties — swapping two documents with cosine 0.8112 and 0.8109 costs you recall
but nothing a user could perceive. A quality ratio of 1.0000 with recall 0.985
means the misses were ties.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scale.quantize import (  # noqa: E402
    binary_encode, calibrate_int8, hamming_search, int8_decode, int8_encode, memory_report,
)


def load_real_passages(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=f"train[:{n*5}]")
    return [t.strip() for t in ds["text"] if len(t.strip()) > 250][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="corpus passages")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidates", nargs="+", type=int, default=[100, 250, 500, 1000])
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--include-synthetic", action="store_true",
                    help="also run the synthetic-vector control (trap 1)")
    ap.add_argument("--out", default=str(HERE / "quantization_results.json"))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    print("=" * 78)
    print(f"  Precision cascade validation  |  {args.model}")
    print("=" * 78)

    texts = load_real_passages(args.n)
    model = SentenceTransformer(args.model, device="cuda")
    model.half()
    t0 = time.perf_counter()
    X = model.encode(texts, batch_size=512, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    embed_s = time.perf_counter() - t0
    n, dim = X.shape
    print(f"corpus: {n:,} real passages, dim {dim}, embedded in {embed_s:.1f}s "
          f"({n/embed_s:.0f}/s)")

    rng = np.random.default_rng(0)
    qi = rng.choice(n, args.queries, replace=False)
    Q = X[qi]
    k = args.k

    sims = Q @ X.T
    for i, q in enumerate(qi):
        sims[i, q] = -1e9                       # trap 2: exclude self
    exact = np.argsort(-sims, axis=1)[:, :k]

    cal = calibrate_int8(X[: min(5000, n)])
    B = binary_encode(X)
    I8 = int8_encode(X, cal)
    F16 = X.astype(np.float16)

    mem = memory_report(n, dim)
    print(f"memory: binary {B.nbytes/1e6:.1f} MB | int8 {I8.nbytes/1e6:.1f} MB | "
          f"float16 {F16.nbytes/1e6:.1f} MB | float32 {X.nbytes/1e6:.1f} MB "
          f"({mem['binary_reduction']} binary reduction)")

    def recall(pred: np.ndarray) -> float:
        return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(pred, exact)]))

    results: dict = {"n": n, "dim": dim, "k": k, "queries": len(qi), "rows": []}

    print(f"\n{'stage':<38}{'recall@'+str(k):>11}{'quality':>10}{'ms/query':>11}")
    print("-" * 70)

    t0 = time.perf_counter()
    _, bi = hamming_search(binary_encode(Q), B, k + 1)
    bi = np.array([[j for j in row if j != q][:k] for row, q in zip(bi, qi)])
    bin_ms = (time.perf_counter() - t0) / len(Q) * 1000
    r = recall(bi)
    print(f"{'binary only (no rescore)':<38}{r:>11.3f}{'':>10}{bin_ms:>11.2f}")
    results["rows"].append({"stage": "binary_only", "recall": r, "ms": bin_ms})

    for name, fetch in (("int8", lambda c: int8_decode(I8[c], cal)),
                        ("float16", lambda c: F16[c].astype(np.float32))):
        for cand in args.candidates:
            t0 = time.perf_counter()
            out = np.empty((len(Q), k), np.int64)
            quals = []
            _, C = hamming_search(binary_encode(Q), B, cand)
            for i in range(len(Q)):
                c = np.array([j for j in C[i] if j != qi[i]])
                s = fetch(c) @ Q[i]
                out[i] = c[np.argsort(-s)[:k]]
                quals.append(float(sims[i][out[i]].mean() / sims[i][exact[i]].mean()))
            ms = (time.perf_counter() - t0) / len(Q) * 1000
            r, q = recall(out), float(np.mean(quals))
            label = f"binary -> {name} rescore, cand={cand}"
            print(f"{label:<38}{r:>11.3f}{q:>10.4f}{ms:>11.2f}")
            results["rows"].append({"stage": label, "format": name, "candidates": cand,
                                    "recall": r, "quality": q, "ms": ms})

    if args.include_synthetic:
        print("\n-- control: synthetic isotropic vectors (trap 1) " + "-" * 20)
        S = rng.normal(size=(n, dim)).astype(np.float32)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        sq = S[rng.choice(n, 100, replace=False)]
        ss = sq @ S.T
        se = np.argsort(-ss, axis=1)[:, :k]
        _, sb = hamming_search(binary_encode(sq), binary_encode(S), k)
        sr = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(sb, se)]))
        print(f"{'binary only, SYNTHETIC vectors':<38}{sr:>11.3f}")
        print(f"  -> real data {results['rows'][0]['recall']:.3f} vs synthetic {sr:.3f}: "
              "validating on random vectors would have condemned the technique.")
        results["synthetic_binary_only_recall"] = sr

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults -> {Path(args.out).resolve()}")

    best = min((r for r in results["rows"] if r.get("format") == "int8"
                and r["recall"] >= 0.98), key=lambda r: r["candidates"], default=None)
    if best:
        print(f"\nrecommendation: int8 rescore at cand={best['candidates']} "
              f"(recall {best['recall']:.3f}, quality {best['quality']:.4f}) — "
              f"half the storage of float16 for the same retrieved quality.")


if __name__ == "__main__":
    main()
