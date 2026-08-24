"""
Download a real multi-hundred-GB text corpus, resumably.

    python download_corpus.py --target-gb 200
    python download_corpus.py --target-gb 200 --status     # just report progress

Corpus: **FineWeb-Edu** `sample/100BT` — 140 parquet shards, 286 GB total, of
high-quality educational web text filtered from Common Crawl. Chosen over raw
FineWeb because the educational filter gives denser, more coherent prose, which
makes retrieval quality numbers mean something.

## Two things this gets right that matter at this scale

1. **It stores outside OneDrive.** `C:\\genai-data` by default. Downloading 200 GB
   into a synced folder would try to push all of it to the cloud, thrash the sync
   client for days, and can corrupt files mid-write. This is not a style
   preference; it is the difference between working and not.

2. **It is resumable at file granularity.** 200 GB at the ~9.5 MB/s measured on
   this connection is ~6 hours. Something will interrupt it. Each parquet shard
   is downloaded independently and `huggingface_hub` skips any shard already
   present with a matching size, so re-running continues where it stopped.

Progress is appended to `download_progress.jsonl` after every shard so you can
poll it from another terminal without touching the download.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/100BT"
DEFAULT_ROOT = Path("C:/genai-data")


def shard_list(target_gb: float) -> tuple[list[tuple[str, int]], float]:
    """The smallest prefix of shards whose total size reaches target_gb."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(REPO, repo_type="dataset", files_metadata=True)
    shards = sorted(
        (s.rfilename, s.size)
        for s in info.siblings
        if s.rfilename.startswith(SUBSET + "/") and s.rfilename.endswith(".parquet")
    )
    picked, total = [], 0
    for name, size in shards:
        picked.append((name, size))
        total += size
        if total / 1e9 >= target_gb:
            break
    return picked, total / 1e9


def downloaded_bytes(cache_dir: Path) -> int:
    return sum(f.stat().st_size for f in cache_dir.rglob("*.parquet") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="storage root; MUST be outside OneDrive")
    ap.add_argument("--target-gb", type=float, default=200.0)
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel shard downloads; >8 rarely helps on one connection")
    args = ap.parse_args()

    root = Path(args.root)
    if "onedrive" in str(root).lower():
        raise SystemExit(
            f"refusing to download {args.target_gb:.0f} GB into a OneDrive path: {root}\n"
            "  pick a --root outside the synced folder"
        )
    cache = root / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from huggingface_hub import snapshot_download

    shards, total_gb = shard_list(args.target_gb)
    have = downloaded_bytes(cache)

    print(f"corpus : {REPO}  {SUBSET}")
    print(f"target : {len(shards)} shards, {total_gb:.1f} GB")
    print(f"store  : {cache}")
    print(f"present: {have/1e9:.1f} GB ({100*have/1e9/total_gb:.0f}%)")

    if args.status:
        prog = root / "download_progress.jsonl"
        if prog.exists():
            lines = prog.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                print(f"last   : {last['gb']:.1f} GB at {last['elapsed_h']:.2f} h "
                      f"({last['mb_s']:.1f} MB/s avg)")
        return

    t0 = time.perf_counter()
    start_bytes = have
    progress = root / "download_progress.jsonl"

    # snapshot_download handles retries, parallelism and skip-if-present for us.
    # allow_patterns restricts it to exactly the shards we picked.
    patterns = [name for name, _ in shards]
    print(f"\ndownloading with {args.workers} workers ...\n", flush=True)

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        max_workers=args.workers,
        cache_dir=str(cache),
    )

    elapsed = time.perf_counter() - t0
    now = downloaded_bytes(cache)
    gained = now - start_bytes
    with open(progress, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "gb": now / 1e9,
            "elapsed_h": elapsed / 3600,
            "mb_s": (gained / 1e6 / elapsed) if elapsed > 0 else 0,
            "shards": len(shards),
        }) + "\n")

    print(f"\ndone: {now/1e9:.1f} GB in {elapsed/3600:.2f} h "
          f"({gained/1e6/max(elapsed,1):.1f} MB/s)")
    print(f"\nNext:  python build_index.py --target-gb {args.target_gb}")


if __name__ == "__main__":
    main()
