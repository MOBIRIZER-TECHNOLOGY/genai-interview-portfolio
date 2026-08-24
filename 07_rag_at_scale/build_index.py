"""
Build a hundred-million-vector index from the FineWeb-Edu corpus, incrementally.

    python build_index.py                      # process every downloaded shard
    python build_index.py --max-shards 4       # a quick slice
    python build_index.py --status             # progress, no work

Runs happily **while `download_corpus.py` is still going** — it processes
whatever shards are complete, records what it finished, and picks up new ones on
the next run. Nothing is ever loaded whole.

## The memory discipline that makes this possible

At 143 M chunks nothing may be held in full:

| what | naive | here |
|---|---|---|
| corpus text | 200 GB in RAM | streamed per parquet row-group |
| float32 vectors | 220 GB | never materialised past one batch |
| searchable index | 220 GB | **6.9 GB binary**, in RAM |
| rescore data | — | 55 GB int8, memmapped from disk |
| chunk text for display | 200 GB | 20 bytes/chunk of *coordinates*, text fetched lazily |

That last row is the one people miss. Storing the chunk text alongside the
vectors would re-materialise the whole corpus. Instead each chunk records
`(shard, row, start, end)` — 20 bytes — and the text is read back out of the
parquet only for the handful of chunks that reach an answer.

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

from scale.quantize import (  # noqa: E402
    Int8Calibration, binary_encode, calibrate_int8, int8_encode, memory_report,
)

DEFAULT_DATA = Path("C:/genai-data")
MODEL = "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------ chunking


def chunk_text(text: str, target_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    """Return (start, end) char offsets. Splits on paragraph then sentence bounds.

    Returns offsets rather than strings so the caller never has to hold a copy of
    the corpus -- at 200 GB that distinction is the whole design.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + target_chars, n)
        if end < n:
            # prefer a paragraph break, then a sentence end, then a space
            window = text[max(pos, end - 300):end]
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut != -1:
                    end = max(pos, end - 300) + cut + len(sep)
                    break
        if end - pos < 60:                       # skip degenerate tails
            break
        spans.append((pos, end))
        pos = max(pos + 1, end - overlap_chars)
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
    return {"model": MODEL, "dim": None, "n_chunks": 0, "shards_done": [], "bytes_text": 0}


def save_manifest(out: Path, m: dict) -> None:
    (out / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- main


def find_shards(cache: Path) -> list[Path]:
    return sorted(p for p in cache.rglob("*.parquet") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=None, help="default: <data-root>/index")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--target-chars", type=int, default=1400, help="~350 tokens")
    ap.add_argument("--overlap-chars", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=512, help="embedding batch")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all available")
    ap.add_argument("--row-group-rows", type=int, default=20000,
                    help="parquet rows held in memory at once")
    ap.add_argument("--calib-sample", type=int, default=50000)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    cache = root / "hf"
    out = Path(args.out) if args.out else root / "index"
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(out)
    shards = find_shards(cache)
    done = set(manifest["shards_done"])
    todo = [s for s in shards if s.name not in done]

    print(f"index    : {out}")
    print(f"shards   : {len(shards)} downloaded, {len(done)} indexed, {len(todo)} to do")
    print(f"chunks   : {manifest['n_chunks']:,}")
    if manifest["dim"]:
        r = memory_report(manifest["n_chunks"], manifest["dim"])
        print(f"memory   : binary {r['binary_gb']} GB | int8 {r['int8_gb']} GB "
              f"| float32 would be {r['float32_gb']} GB ({r['binary_reduction']} saved)")
    if args.status:
        return
    if not todo:
        print("\nnothing new to index (download may still be running)")
        return
    if args.max_shards:
        todo = todo[: args.max_shards]
        print(f"limiting to {len(todo)} shard(s)")

    import pyarrow.parquet as pq
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"\nloading {args.model} ...")
    model = SentenceTransformer(args.model, device="cuda")
    model.half()                       # fp16: measured 4900 chunks/s vs 215 fp32
    dim = model.get_sentence_embedding_dimension() if hasattr(
        model, "get_sentence_embedding_dimension") else model.get_embedding_dimension()
    manifest["dim"] = dim
    print(f"  dim={dim}  fp16  batch={args.batch_size}")

    calib_path = out / "int8_calib.json"
    calib: Int8Calibration | None = (
        Int8Calibration.load(calib_path) if calib_path.exists() else None
    )
    calib_buf: list[np.ndarray] = []

    writer = ShardWriter(out, dim)
    shard_index = {s.name: i for i, s in enumerate(shards)}
    t_start = time.perf_counter()
    total_new = 0

    try:
        for si, shard in enumerate(todo, 1):
            sid = shard_index[shard.name]
            t0 = time.perf_counter()
            pf = pq.ParquetFile(shard)
            shard_chunks = 0
            shard_bytes = 0

            for batch in pf.iter_batches(batch_size=args.row_group_rows, columns=["text"]):
                texts = batch.column("text").to_pylist()

                pending_txt: list[str] = []
                pending_co: list[tuple[int, int, int, int]] = []
                for row_i, text in enumerate(texts):
                    if not text:
                        continue
                    shard_bytes += len(text)
                    for (a, b) in chunk_text(text, args.target_chars, args.overlap_chars):
                        pending_txt.append(text[a:b])
                        pending_co.append((sid, row_i, a, b))

                    if len(pending_txt) >= args.batch_size * 4:
                        n = _flush(model, writer, pending_txt, pending_co,
                                   args.batch_size, calib, calib_buf, args.calib_sample)
                        if calib is None and sum(len(c) for c in calib_buf) >= args.calib_sample:
                            calib = calibrate_int8(np.concatenate(calib_buf))
                            calib.save(calib_path)
                            calib_buf.clear()
                            print(f"    int8 calibration frozen from {calib.n_calibration:,} vectors")
                        shard_chunks += n
                        pending_txt, pending_co = [], []

                if pending_txt:
                    shard_chunks += _flush(model, writer, pending_txt, pending_co,
                                           args.batch_size, calib, calib_buf, args.calib_sample)

            writer.flush()
            manifest["shards_done"].append(shard.name)
            manifest["n_chunks"] += shard_chunks
            manifest["bytes_text"] += shard_bytes
            total_new += shard_chunks
            save_manifest(out, manifest)

            el = time.perf_counter() - t0
            rate = shard_chunks / el if el else 0
            gb = manifest["bytes_text"] / 1e9
            print(f"  [{si}/{len(todo)}] {shard.name}  {shard_chunks:,} chunks in "
                  f"{el/60:.1f} min ({rate:.0f}/s)  | total {manifest['n_chunks']:,} chunks, "
                  f"{gb:.1f} GB text", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted -- manifest saved, re-run to continue")
    finally:
        writer.close()
        save_manifest(out, manifest)

    el = time.perf_counter() - t_start
    r = memory_report(manifest["n_chunks"], dim)
    print(f"\n{total_new:,} new chunks in {el/60:.1f} min")
    print(f"total: {manifest['n_chunks']:,} chunks from {manifest['bytes_text']/1e9:.1f} GB text")
    print(f"binary index {r['binary_gb']} GB | int8 {r['int8_gb']} GB | "
          f"float32 avoided {r['float32_gb']} GB ({r['binary_reduction']})")
    print(f"\nNext:  python bench_latency.py")


def _flush(model, writer, texts, coords, batch_size, calib, calib_buf, calib_target) -> int:
    """Embed a batch, quantise, append. Returns chunk count."""
    vecs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)

    if calib is None:
        if sum(len(c) for c in calib_buf) < calib_target:
            calib_buf.append(vecs[: max(1, len(vecs) // 4)].copy())
        # Until calibration is frozen we cannot write int8. Use a provisional
        # symmetric range: normalised embeddings live in [-1, 1], so this is a
        # safe bound that the frozen calibration will later tighten.
        provisional = Int8Calibration([-1.0] * vecs.shape[1], [1.0] * vecs.shape[1],
                                      vecs.shape[1], 0)
        i8 = int8_encode(vecs, provisional)
    else:
        i8 = int8_encode(vecs, calib)

    writer.write(binary_encode(vecs), i8, np.asarray(coords, dtype=np.int64))
    return len(texts)


if __name__ == "__main__":
    main()
