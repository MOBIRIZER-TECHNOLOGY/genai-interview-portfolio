"""
Build a hundred-million-vector index from the FineWeb-Edu corpus, incrementally.

    python build_index.py                      # process every downloaded shard
    python build_index.py --max-shards 1       # a quick slice
    python build_index.py --status             # progress, no work
    python build_index.py --chunkers 4         # more producer threads

Runs happily **while `download_corpus.py` is still going** — it processes
whatever shards are complete, records what it finished, and picks up new ones on
the next run.

## Overlapped, not sequential

Reading parquet, chunking text, embedding and writing all happen concurrently
(`scale/pipeline.py`). The single-threaded version hit 3,100 chunks/s against a
4,900 chunks/s GPU ceiling — the GPU idled a third of the time while Python
decoded parquet and sliced strings. Threads work here despite the GIL because
pyarrow, HF `tokenizers` and every CUDA op all release it.

The run prints GPU-busy percentage so you can see whether the GPU is actually
the bottleneck or still waiting on Python.

## Crash safety

The three data files are append-only and the manifest is the commit point. If
the process dies mid-shard, the appended-but-uncommitted rows are **truncated on
the next run** back to the manifest's count. Without that, resuming would
re-process the shard and append its vectors a second time — silent duplicates
that no error message would ever tell you about.

## The memory discipline that makes this possible

At hundreds of millions of chunks nothing may be held in full:

| what | naive | here |
|---|---|---|
| corpus text | 344 GB in RAM | streamed per parquet row-group |
| float32 vectors | 515 GB | never materialised past one batch |
| searchable index | 515 GB | **16 GB binary**, in RAM |
| rescore data | — | 129 GB int8, memmapped from disk |
| chunk text for display | 344 GB | 32 bytes/chunk of *coordinates* |

## Output layout (in --out, outside OneDrive)

    binary.u8        [n, dim/8] uint8   memmap, loaded to RAM for search
    int8.i8          [n, dim]   int8    memmap, read only for rescoring
    coords.i64       [n, 4]     int64   (shard, row, char_start, char_end)
    int8_calib.json  frozen global quantisation range
    manifest.json    shard bookkeeping, dims, counts -- drives resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scale.pipeline import ShardPipeline, iter_shards  # noqa: E402
from scale.quantize import (  # noqa: E402
    Int8Calibration, binary_encode, calibrate_int8, int8_encode, memory_report,
)

DEFAULT_DATA = Path("C:/genai-data")
MODEL = "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------ chunking


def chunk_text(text: str, target_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    """Return (start, end) char offsets. Splits on paragraph then sentence bounds.

    Returns offsets rather than strings so the caller never has to hold a copy of
    the corpus -- at hundreds of GB that distinction is the whole design.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    if n < 60:
        return spans

    pos = 0
    while pos < n:
        end = min(pos + target_chars, n)
        if end < n:
            # prefer a paragraph break, then a sentence end, then a space
            window_start = max(pos, end - 300)
            window = text[window_start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut != -1:
                    end = window_start + cut + len(sep)
                    break

        if end - pos >= 60:                      # skip degenerate tails
            spans.append((pos, end))

        # Reaching the end of the text terminates the loop.
        #
        # Without this, a short document is catastrophic: once `end == n` the
        # separator search is skipped, so `end` stops moving, and
        # `pos = max(pos + 1, end - overlap)` advances by ONE CHARACTER per
        # iteration. A 500-char document then yields ~180 near-identical chunks
        # instead of 1. Measured on real FineWeb text this produced 42x too many
        # chunks with a mean length of 178 chars against a 1400-char target --
        # 42x the embedding cost, for near-duplicates that poison retrieval.
        if end >= n:
            break

        nxt = end - overlap_chars
        if nxt <= pos:                           # guarantee forward progress
            nxt = end
        pos = nxt
    return spans


# -------------------------------------------------------------------- store


class ShardWriter:
    """Append-only writers for the three parallel arrays."""

    def __init__(self, out: Path, dim: int):
        self.out = out
        self.dim = dim
        out.mkdir(parents=True, exist_ok=True)
        self.f_bin = open(out / "binary.u8", "ab")
        self.f_i8 = open(out / "int8.i8", "ab")
        self.f_co = open(out / "coords.i64", "ab")

    def write(self, binary: np.ndarray, i8: np.ndarray, coords: np.ndarray) -> None:
        self.f_bin.write(binary.astype(np.uint8, copy=False).tobytes())
        self.f_i8.write(i8.astype(np.int8, copy=False).tobytes())
        self.f_co.write(coords.astype(np.int64, copy=False).tobytes())

    def flush(self) -> None:
        for f in (self.f_bin, self.f_i8, self.f_co):
            f.flush()
            os.fsync(f.fileno())

    def close(self) -> None:
        self.flush()
        for f in (self.f_bin, self.f_i8, self.f_co):
            f.close()


def load_manifest(out: Path) -> dict:
    p = out / "manifest.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"model": MODEL, "dim": None, "n_chunks": 0, "shards_done": [],
            "bytes_text": 0, "cache": None}


def save_manifest(out: Path, m: dict) -> None:
    """Write the manifest atomically -- it is the commit point for a shard.

    A torn manifest is worse than no manifest: it would make the truncation
    below compute the wrong offset and corrupt the index.
    """
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
    os.replace(tmp, out / "manifest.json")


def truncate_uncommitted(out: Path, manifest: dict) -> int:
    """Roll the data files back to the last committed shard boundary.

    Returns how many uncommitted rows were discarded. This is what makes a
    killed run safe to resume: the alternative is re-processing a shard whose
    partial vectors are already on disk, silently doubling them.
    """
    dim = manifest.get("dim")
    if not dim:
        return 0
    n = manifest["n_chunks"]
    expected = {
        "binary.u8": n * (dim // 8),
        "int8.i8": n * dim,
        "coords.i64": n * 4 * 8,
    }
    discarded = 0
    for name, want in expected.items():
        p = out / name
        if not p.exists():
            continue
        have = p.stat().st_size
        if have > want:
            row_bytes = want // n if n else 1
            discarded = max(discarded, (have - want) // max(row_bytes, 1))
            with open(p, "r+b") as f:
                f.truncate(want)
        elif have < want:
            raise SystemExit(
                f"{p} is SHORTER than the manifest claims ({have} < {want}). "
                "The index and manifest disagree; delete the index directory and rebuild."
            )
    return discarded


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=None, help="default: <data-root>/index")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--target-chars", type=int, default=1400, help="~350 tokens")
    ap.add_argument("--overlap-chars", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=512, help="GPU embedding batch")
    ap.add_argument("--embed-batch", type=int, default=2048,
                    help="chunks handed to the GPU stage at once")
    ap.add_argument("--chunkers", type=int, default=3, help="producer threads")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all available")
    ap.add_argument("--row-group-rows", type=int, default=20000)
    ap.add_argument("--calib-sample", type=int, default=50000)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    cache = root / "hf"
    out = Path(args.out) if args.out else root / "index"
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(out)
    shards = list(iter_shards(cache))
    done = set(manifest["shards_done"])
    todo = [s for s in shards if s.name not in done]

    print(f"index    : {out}")
    print(f"shards   : {len(shards)} downloaded, {len(done)} indexed, {len(todo)} to do")
    print(f"chunks   : {manifest['n_chunks']:,}")
    if manifest["dim"]:
        r = memory_report(manifest["n_chunks"], manifest["dim"])
        print(f"memory   : binary {r['binary_gb']} GB | int8 {r['int8_gb']} GB "
              f"| float32 avoided {r['float32_gb']} GB ({r['binary_reduction']})")
    if args.status:
        return

    discarded = truncate_uncommitted(out, manifest)
    if discarded:
        print(f"\nrolled back {discarded:,} uncommitted rows from an interrupted run")

    if not todo:
        print("\nnothing new to index (download may still be running)")
        return
    if args.max_shards:
        todo = todo[: args.max_shards]
        print(f"limiting to {len(todo)} shard(s)")

    import torch
    from sentence_transformers import SentenceTransformer

    print(f"\nloading {args.model} ...")
    model = SentenceTransformer(args.model, device="cuda")
    model.half()                       # fp16: measured 4900 chunks/s vs 215 fp32
    getter = getattr(model, "get_embedding_dimension", None) or \
        model.get_sentence_embedding_dimension
    dim = getter()
    manifest["dim"] = dim
    manifest["cache"] = str(cache)
    print(f"  dim={dim}  fp16  gpu_batch={args.batch_size}  chunkers={args.chunkers}")

    calib_path = out / "int8_calib.json"
    calib: Int8Calibration | None = (
        Int8Calibration.load(calib_path) if calib_path.exists() else None
    )
    calib_buf: list[np.ndarray] = []
    # Until calibration is frozen we cannot write final int8. Normalised
    # embeddings live in [-1, 1], so this is a safe provisional bound.
    provisional = Int8Calibration([-1.0] * dim, [1.0] * dim, dim, 0)

    writer = ShardWriter(out, dim)
    shard_index = {s.name: i for i, s in enumerate(shards)}

    def embed_fn(texts: list[str]):
        nonlocal calib
        vecs = model.encode(
            texts, batch_size=args.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

        if calib is None:
            if sum(len(c) for c in calib_buf) < args.calib_sample:
                calib_buf.append(vecs[:: max(1, len(vecs) // 256)].copy())
            if sum(len(c) for c in calib_buf) >= args.calib_sample:
                calib = calibrate_int8(np.concatenate(calib_buf))
                calib.save(calib_path)
                calib_buf.clear()
                print(f"      int8 calibration frozen from {calib.n_calibration:,} vectors",
                      flush=True)
        cal = calib or provisional
        return binary_encode(vecs), int8_encode(vecs, cal)

    def write_fn(arrays, coords) -> None:
        binary, i8 = arrays
        writer.write(binary, i8, np.asarray(coords, dtype=np.int64))

    t_start = time.perf_counter()
    total_new = 0

    try:
        for si, shard in enumerate(todo, 1):
            print(f"\n  [{si}/{len(todo)}] {shard.name}", flush=True)
            pipe = ShardPipeline(
                shard_path=shard,
                shard_id=shard_index[shard.name],
                chunk_fn=lambda t: chunk_text(t, args.target_chars, args.overlap_chars),
                embed_fn=embed_fn,
                write_fn=write_fn,
                n_chunkers=args.chunkers,
                row_group_rows=args.row_group_rows,
                embed_batch=args.embed_batch,
            )
            stats = pipe.run()
            writer.flush()

            if stats.errors:
                for e in stats.errors[:5]:
                    print(f"      ERROR {e}")
                raise SystemExit("aborting: shard failed, manifest not committed")

            # commit point
            manifest["shards_done"].append(shard.name)
            manifest["n_chunks"] += stats.chunks
            manifest["bytes_text"] = manifest.get("bytes_text", 0)
            total_new += stats.chunks
            save_manifest(out, manifest)

            print(f"      {stats.summary()}")
            print(f"      total {manifest['n_chunks']:,} chunks", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted -- committed shards are safe, partial work rolls back on resume")
    finally:
        writer.close()
        save_manifest(out, manifest)

    el = time.perf_counter() - t_start
    r = memory_report(manifest["n_chunks"], dim)
    print(f"\n{total_new:,} new chunks in {el/60:.1f} min "
          f"({total_new/max(el,1e-9):.0f}/s)")
    print(f"total: {manifest['n_chunks']:,} chunks")
    print(f"binary {r['binary_gb']} GB | int8 {r['int8_gb']} GB | "
          f"float32 avoided {r['float32_gb']} GB ({r['binary_reduction']})")
    print(f"\nNext:  python bench_latency.py")


if __name__ == "__main__":
    main()
