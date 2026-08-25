"""
Measure query latency against the real index, and project it to 200 GB scale.

    python bench_latency.py                          # full sweep
    python bench_latency.py --sizes 1000000 5000000  # specific corpus sizes
    python bench_latency.py --concurrency 1 4 16     # load test

## What is actually being measured

A RAG query is four costs, and they scale completely differently. Reporting one
blended number hides which one will break first:

| stage | cost model | scales with |
|---|---|---|
| embed query | one GPU forward pass | nothing (constant) |
| **binary scan** | XOR + popcount over the whole index | **O(n)** — the one that grows |
| int8 rescore | read `candidates` rows from a memmap | candidate depth, not n |
| fetch text | parquet read for k rows | k |

So the honest question is not "how fast is the index" but **"at what n does the
binary scan stop fitting inside the latency budget"** — and everything else is
roughly constant.

## How the scaling curve is produced

The index is subsampled to each target size and re-benchmarked. Subsampling is
legitimate here because the binary scan is a **full linear scan** — its cost is
exactly proportional to the number of rows and does not depend on which rows.
That is not true for a graph index like HNSW, where you would have to build the
graph at each size to measure honestly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scale.quantize import POPCOUNT, binary_encode, int8_decode  # noqa: E402
from scale.search import ScaleIndex  # noqa: E402

QUERIES = [
    "how do photosynthesis and cellular respiration relate to each other",
    "what caused the decline of the roman empire",
    "explain the difference between mitosis and meiosis",
    "how does a jet engine generate thrust",
    "what are the main causes of inflation in an economy",
    "describe the water cycle and its stages",
    "how do vaccines train the immune system",
    "what is the significance of the magna carta",
    "explain how neural networks learn from data",
    "what are the health effects of long term sleep deprivation",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[i]


def bench_size(index: ScaleIndex, qvecs: np.ndarray, n: int, candidates: int,
               k: int, repeats: int) -> dict:
    """Latency at a simulated corpus size of `n` vectors."""
    binary = np.ascontiguousarray(index.binary[:n])
    int8 = index.int8

    bin_ms, res_ms, tot_ms = [], [], []
    for _ in range(repeats):
        for qv in qvecs:
            qcode = binary_encode(qv)[0]

            t0 = time.perf_counter()
            d = POPCOUNT[np.bitwise_xor(binary, qcode)].sum(axis=1)
            c = min(candidates, n)
            cand = np.argpartition(d, c - 1)[:c]
            t1 = time.perf_counter()

            rows = int8[np.sort(cand)]
            scores = int8_decode(rows, index.cal) @ qv
            kk = min(k, len(scores))
            part = np.argpartition(-scores, kk - 1)[:kk]
            part[np.argsort(-scores[part])]
            t2 = time.perf_counter()

            bin_ms.append((t1 - t0) * 1000)
            res_ms.append((t2 - t1) * 1000)
            tot_ms.append((t2 - t0) * 1000)

    return {
        "n_vectors": n,
        "binary_gb": round(n * index.dim / 8 / 1e9, 3),
        "candidates": candidates,
        "binary_p50_ms": round(statistics.median(bin_ms), 2),
        "rescore_p50_ms": round(statistics.median(res_ms), 2),
        "p50_ms": round(statistics.median(tot_ms), 2),
        "p95_ms": round(percentile(tot_ms, 95), 2),
        "p99_ms": round(percentile(tot_ms, 99), 2),
        "qps_single_thread": round(1000 / statistics.median(tot_ms), 1),
    }


def bench_concurrency(index: ScaleIndex, qvecs: np.ndarray, n: int, candidates: int,
                      workers: int, total_queries: int) -> dict:
    """Throughput and tail latency under concurrent load."""
    binary = np.ascontiguousarray(index.binary[:n])
    lat: list[float] = []

    def one(i: int) -> float:
        qv = qvecs[i % len(qvecs)]
        t0 = time.perf_counter()
        d = POPCOUNT[np.bitwise_xor(binary, binary_encode(qv)[0])].sum(axis=1)
        c = min(candidates, n)
        cand = np.argpartition(d, c - 1)[:c]
        rows = index.int8[np.sort(cand)]
        int8_decode(rows, index.cal) @ qv
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        lat = list(ex.map(one, range(total_queries)))
    wall = time.perf_counter() - t0

    return {
        "workers": workers,
        "queries": total_queries,
        "wall_s": round(wall, 2),
        "throughput_qps": round(total_queries / wall, 1),
        "p50_ms": round(statistics.median(lat), 2),
        "p95_ms": round(percentile(lat, 95), 2),
        "p99_ms": round(percentile(lat, 99), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="C:/genai-data/index")
    ap.add_argument("--sizes", nargs="*", type=int, default=None)
    ap.add_argument("--candidates", nargs="+", type=int, default=[100, 500, 2000])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--concurrency-queries", type=int, default=200)
    ap.add_argument("--skip-e2e", action="store_true", help="skip the end-to-end query test")
    ap.add_argument("--out", default=str(HERE / "latency_results.json"))
    args = ap.parse_args()

    index = ScaleIndex.open(args.index)
    st = index.stats()

    print("=" * 84)
    # shards indexed before bytes_text was tracked report 0; say so rather
    # than printing "from 0.0 GB of text" next to 13.6M chunks
    text_note = (f"from {st['text_gb']} GB of text" if st["text_gb"]
                 else "text size not recorded for these shards")
    print(f"  Latency benchmark  |  {st['n_chunks']:,} chunks, {text_note}")
    print(f"  binary {st['binary_gb']} GB in RAM | int8 {st['int8_gb']} GB memmapped | "
          f"float32 avoided: {st['float32_avoided_gb']} GB")
    print("=" * 84)

    print("\nembedding benchmark queries ...")
    qvecs = np.stack([index.encode_query(q) for q in QUERIES])

    # time the query embedding itself -- it is constant but not free
    t0 = time.perf_counter()
    for q in QUERIES:
        index.encode_query(q)
    embed_ms = (time.perf_counter() - t0) / len(QUERIES) * 1000
    print(f"  query embedding: {embed_ms:.1f} ms each (constant, independent of corpus size)")

    n_max = index.n
    sizes = args.sizes or [s for s in (100_000, 500_000, 1_000_000, 2_000_000,
                                       5_000_000, 10_000_000) if s <= n_max]
    if n_max not in sizes:
        sizes.append(n_max)
    sizes = sorted(set(s for s in sizes if s <= n_max))

    results: dict = {"index": st, "embed_ms": round(embed_ms, 2),
                     "scaling": [], "concurrency": [], "e2e": []}

    print(f"\n## Scaling: latency vs index size  (k={args.k})\n")
    print(f"{'vectors':>12}{'binary GB':>11}{'cand':>7}{'binary ms':>11}"
          f"{'rescore ms':>12}{'p50 ms':>9}{'p99 ms':>9}{'QPS/thread':>12}")
    print("-" * 84)
    for n in sizes:
        for cand in args.candidates:
            row = bench_size(index, qvecs, n, cand, args.k, args.repeats)
            results["scaling"].append(row)
            print(f"{row['n_vectors']:>12,}{row['binary_gb']:>11.3f}{row['candidates']:>7}"
                  f"{row['binary_p50_ms']:>11.2f}{row['rescore_p50_ms']:>12.2f}"
                  f"{row['p50_ms']:>9.2f}{row['p99_ms']:>9.2f}{row['qps_single_thread']:>12.1f}")

    # ---- project to the full 200 GB corpus -----------------------------
    biggest = max((r for r in results["scaling"] if r["candidates"] == 500),
                  key=lambda r: r["n_vectors"], default=None)
    if biggest and biggest["n_vectors"] > 0:
        # measured: 3.4M chunks per 2.15 GB shard x 93 shards
        TARGET = 316_000_000
        factor = TARGET / biggest["n_vectors"]
        proj_bin = biggest["binary_p50_ms"] * factor      # O(n)
        proj_res = biggest["rescore_p50_ms"]              # O(candidates)
        print(f"\n## Projection to the full 200 GB corpus (~{TARGET/1e6:.0f}M chunks)\n")
        print(f"  measured at {biggest['n_vectors']:,} vectors: "
              f"binary {biggest['binary_p50_ms']:.2f} ms + rescore {biggest['rescore_p50_ms']:.2f} ms")
        print(f"  binary scan is O(n), so x{factor:.0f} -> {proj_bin:.0f} ms")
        print(f"  rescore is O(candidates), so unchanged -> {proj_res:.2f} ms")
        print(f"  projected p50 ~ {proj_bin + proj_res + embed_ms:.0f} ms "
              f"(incl. {embed_ms:.0f} ms query embedding)")
        print(f"  binary index at that size: {TARGET * index.dim / 8 / 1e9:.1f} GB in RAM")
        results["projection_200gb"] = {
            "target_vectors": TARGET, "factor": round(factor, 1),
            "projected_binary_ms": round(proj_bin, 1),
            "projected_p50_ms": round(proj_bin + proj_res + embed_ms, 1),
            "binary_index_gb": round(TARGET * index.dim / 8 / 1e9, 1),
        }
        if proj_bin > 300:
            print("\n  -> a flat scan is past a sane interactive budget at this size.")
            print("     This is the measured argument for IVF/HNSW: partition the")
            print("     space so a query touches a fraction of it instead of all of it.")

    print(f"\n## Concurrency  (n={sizes[-1]:,}, candidates=500)\n")
    print(f"{'workers':>9}{'QPS':>10}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}")
    print("-" * 49)
    for w in args.concurrency:
        row = bench_concurrency(index, qvecs, sizes[-1], 500, w, args.concurrency_queries)
        results["concurrency"].append(row)
        print(f"{row['workers']:>9}{row['throughput_qps']:>10.1f}{row['p50_ms']:>10.2f}"
              f"{row['p95_ms']:>10.2f}{row['p99_ms']:>10.2f}")
    print("\n  numpy releases the GIL inside the XOR/popcount, so threads do scale --")
    print("  but they contend for the same memory bandwidth, which is the real ceiling.")

    if not args.skip_e2e:
        print("\n## End-to-end (embed -> search -> fetch text)\n")
        for q in QUERIES[:3]:
            t0 = time.perf_counter()
            hits = index.search(q, k=5, candidates=500, fetch_text=True)
            ms = (time.perf_counter() - t0) * 1000
            results["e2e"].append({"query": q, "ms": round(ms, 1), "hits": len(hits)})
            print(f"  {ms:7.0f} ms   {q[:58]}")
            if hits and hits[0].text:
                print(f"              -> {hits[0].text[:100].strip()}...")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
