"""
Producer/consumer pipeline so the GPU stops waiting on Python.

    from scale.pipeline import ShardPipeline

## The problem it solves

The single-threaded build did this, in order, forever:

    read parquet -> chunk text -> EMBED ON GPU -> quantise -> write to disk
    [--------- CPU, GIL-bound ---------]  [-- GPU --]  [-- CPU --]

Measured: **4,900 chunks/s** when the GPU is fed continuously, but only
**3,100 chunks/s** end-to-end. The GPU sat idle for roughly a third of the run
while Python decoded parquet and sliced strings.

## Why threads work here despite the GIL

Chunking is pure-Python string work and *does* hold the GIL. This would be
pointless if the GPU path also held it — but it doesn't:

- `pyarrow` releases the GIL during parquet decode.
- HuggingFace `tokenizers` is Rust and releases the GIL.
- `torch` releases the GIL for the duration of every CUDA op.

So while the GPU is busy on a batch, the interpreter is free, and chunker
threads run in that window. The GIL is only contended for the brief Python-level
glue between those calls.

## Shape

    [reader]  parquet row-batches         -> raw_q
    [chunker x N]  text -> (texts, coords) -> embed_q
    [main]  embed on GPU + quantise        -> write_q
    [writer]  append to the three files

Every queue is bounded. That is not incidental: an unbounded queue in front of a
GPU turns a producer/consumer speedup into an out-of-memory crash, because the
readers will happily materialise the entire shard in RAM. `maxsize` makes a fast
producer block instead, which is exactly the backpressure you want.

The pipeline reports how long each stage spent **blocked on a queue**, which is
what tells you where the bottleneck actually moved to.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

SENTINEL = object()


@dataclass
class PipelineStats:
    """Where the wall clock went. The point of the whole exercise."""
    chunks: int = 0
    read_batches: int = 0
    gpu_wait_s: float = 0.0        # main thread blocked waiting for input
    gpu_busy_s: float = 0.0        # main thread embedding
    write_wait_s: float = 0.0      # main thread blocked because writer is behind
    reader_block_s: float = 0.0    # reader blocked because chunkers are behind
    chunker_block_s: float = 0.0   # chunkers blocked because GPU is behind
    writer_busy_s: float = 0.0
    wall_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.wall_s <= 0:
            return "no timing"
        starved = 100 * self.gpu_wait_s / self.wall_s
        busy = 100 * self.gpu_busy_s / self.wall_s
        return (
            f"{self.chunks:,} chunks in {self.wall_s:.0f}s "
            f"({self.chunks / max(self.wall_s, 1e-9):.0f}/s)\n"
            f"      GPU busy {busy:.0f}% | GPU starved {starved:.0f}% | "
            f"blocked on writer {100*self.write_wait_s/self.wall_s:.0f}%\n"
            f"      producers blocked (backpressure, healthy): "
            f"reader {self.reader_block_s:.0f}s, chunkers {self.chunker_block_s:.0f}s"
        )


class ShardPipeline:
    """Overlapped read -> chunk -> embed -> write for one parquet shard.

    One shard at a time on purpose. Shard boundaries are where the manifest is
    committed, and interleaving shards would make "which shards are safely on
    disk" ambiguous after a crash -- which is the exact bug that makes a resume
    duplicate data.
    """

    def __init__(
        self,
        shard_path: Path,
        shard_id: int,
        chunk_fn: Callable[[str], list[tuple[int, int]]],
        embed_fn: Callable[[list[str]], object],
        write_fn: Callable[[object, list], None],
        n_chunkers: int = 3,
        row_group_rows: int = 20_000,
        embed_batch: int = 2048,
        raw_queue_size: int = 4,
        embed_queue_size: int = 6,
        write_queue_size: int = 8,
    ):
        self.shard_path = shard_path
        self.shard_id = shard_id
        self.chunk_fn = chunk_fn
        self.embed_fn = embed_fn
        self.write_fn = write_fn
        self.n_chunkers = n_chunkers
        self.row_group_rows = row_group_rows
        self.embed_batch = embed_batch

        self.raw_q: queue.Queue = queue.Queue(maxsize=raw_queue_size)
        self.embed_q: queue.Queue = queue.Queue(maxsize=embed_queue_size)
        self.write_q: queue.Queue = queue.Queue(maxsize=write_queue_size)
        self.stats = PipelineStats()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ stages

    def _reader(self) -> None:
        """Decode parquet row-batches. pyarrow releases the GIL in here."""
        import pyarrow.parquet as pq

        try:
            pf = pq.ParquetFile(self.shard_path)
            base_row = 0
            for batch in pf.iter_batches(batch_size=self.row_group_rows, columns=["text"]):
                texts = batch.column("text").to_pylist()
                t0 = time.perf_counter()
                self.raw_q.put((base_row, texts))
                with self._lock:
                    self.stats.reader_block_s += time.perf_counter() - t0
                    self.stats.read_batches += 1
                base_row += len(texts)
        except Exception as exc:
            with self._lock:
                self.stats.errors.append(f"reader: {type(exc).__name__}: {exc}")
        finally:
            for _ in range(self.n_chunkers):
                self.raw_q.put(SENTINEL)

    def _chunker(self) -> None:
        """Pure-Python string slicing. Holds the GIL, runs while the GPU works."""
        pending_txt: list[str] = []
        pending_co: list[tuple[int, int, int, int]] = []

        def flush() -> None:
            nonlocal pending_txt, pending_co
            if not pending_txt:
                return
            t0 = time.perf_counter()
            self.embed_q.put((pending_txt, pending_co))
            with self._lock:
                self.stats.chunker_block_s += time.perf_counter() - t0
            pending_txt, pending_co = [], []

        try:
            while True:
                item = self.raw_q.get()
                if item is SENTINEL:
                    break
                base_row, texts = item
                for i, text in enumerate(texts):
                    if not text:
                        continue
                    row = base_row + i
                    for (a, b) in self.chunk_fn(text):
                        pending_txt.append(text[a:b])
                        pending_co.append((self.shard_id, row, a, b))
                    if len(pending_txt) >= self.embed_batch:
                        flush()
            flush()
        except Exception as exc:
            with self._lock:
                self.stats.errors.append(f"chunker: {type(exc).__name__}: {exc}")
        finally:
            self.embed_q.put(SENTINEL)

    def _writer(self) -> None:
        """Disk append, off the main thread so fsync never stalls the GPU."""
        while True:
            item = self.write_q.get()
            if item is SENTINEL:
                break
            arrays, coords = item
            t0 = time.perf_counter()
            try:
                self.write_fn(arrays, coords)
            except Exception as exc:
                with self._lock:
                    self.stats.errors.append(f"writer: {type(exc).__name__}: {exc}")
            with self._lock:
                self.stats.writer_busy_s += time.perf_counter() - t0

    # -------------------------------------------------------------- run

    def run(self, progress_every: int = 500_000) -> PipelineStats:
        """Drive the pipeline; the GPU work happens on THIS thread.

        Embedding stays on the calling thread deliberately: one GPU, one CUDA
        stream, and moving it to a worker would add queue hops for no gain.
        """
        t_start = time.perf_counter()

        reader = threading.Thread(target=self._reader, name="reader", daemon=True)
        chunkers = [
            threading.Thread(target=self._chunker, name=f"chunker{i}", daemon=True)
            for i in range(self.n_chunkers)
        ]
        writer = threading.Thread(target=self._writer, name="writer", daemon=True)

        reader.start()
        for c in chunkers:
            c.start()
        writer.start()

        finished = 0
        next_report = progress_every
        try:
            while finished < self.n_chunkers:
                t0 = time.perf_counter()
                item = self.embed_q.get()
                self.stats.gpu_wait_s += time.perf_counter() - t0

                if item is SENTINEL:
                    finished += 1
                    continue

                texts, coords = item
                t0 = time.perf_counter()
                arrays = self.embed_fn(texts)
                self.stats.gpu_busy_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                self.write_q.put((arrays, coords))
                self.stats.write_wait_s += time.perf_counter() - t0

                self.stats.chunks += len(texts)
                if self.stats.chunks >= next_report:
                    el = time.perf_counter() - t_start
                    print(f"      {self.stats.chunks:,} chunks  "
                          f"{self.stats.chunks/el:.0f}/s  "
                          f"GPU busy {100*self.stats.gpu_busy_s/el:.0f}%", flush=True)
                    next_report += progress_every
        finally:
            self.write_q.put(SENTINEL)
            writer.join(timeout=300)
            reader.join(timeout=30)
            for c in chunkers:
                c.join(timeout=30)

        self.stats.wall_s = time.perf_counter() - t_start
        return self.stats


def iter_shards(cache: Path) -> Iterator[Path]:
    yield from sorted(p for p in cache.rglob("*.parquet") if p.is_file())
