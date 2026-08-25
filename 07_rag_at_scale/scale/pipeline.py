"""
Producer/consumer pipeline so the GPU stops waiting on Python.

    from scale.pipeline import ShardPipeline

## The problem it solves

The single-threaded build did this, in order, forever:

    read parquet -> chunk text -> EMBED ON GPU -> quantise -> write to disk
    [--------- CPU, GIL-bound ---------]  [-- GPU --]  [-- CPU --]

The GPU sat idle for roughly a third of the run while Python decoded parquet and
sliced strings.

## Why threads work here despite the GIL

Chunking is pure-Python string work and *does* hold the GIL. This would be
pointless if the GPU path also held it — but it doesn't:

- `pyarrow` releases the GIL during parquet decode.
- HuggingFace `tokenizers` is Rust and releases the GIL.
- `torch` releases the GIL for the duration of every CUDA op.

So while the GPU is busy on a batch, the interpreter is free, and chunker
threads run in that window.

## Shape

    [reader]       parquet row-batches      -> raw_q
    [chunker x N]  text -> (texts, coords)  -> embed_q
    [main]         embed on GPU, quantise, AND write

**Writing stays on the main thread deliberately.** An earlier version had a
separate writer thread and it deadlocked: the main thread blocked on a full
`write_q` while the writer was stuck, and with no timeouts anywhere the process
sat at 0% CPU forever. The writer bought almost nothing — appending ~1 MB to the
OS buffer is sub-millisecond, and the expensive `fsync` happens once per shard —
so removing it eliminated a deadlock surface for no measurable cost. The real win
is overlapping *read and chunk* with the GPU, and that is preserved.

Every queue is bounded, so a fast producer blocks instead of materialising the
whole shard in RAM. Every blocking call has a **timeout** and checks a shutdown
event, so a stall surfaces as a diagnosable message instead of a silent hang.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

SENTINEL = object()
POLL = 1.0          # seconds; how often a blocked stage re-checks for shutdown


@dataclass
class PipelineStats:
    """Where the wall clock went. The point of the whole exercise."""
    chunks: int = 0
    text_bytes: int = 0
    read_batches: int = 0
    gpu_wait_s: float = 0.0        # main blocked waiting for input  -> producers too slow
    gpu_busy_s: float = 0.0        # main embedding                  -> GPU is the bottleneck
    write_s: float = 0.0           # main writing to disk
    reader_block_s: float = 0.0    # reader blocked (healthy backpressure)
    chunker_block_s: float = 0.0   # chunkers blocked (healthy backpressure)
    wall_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.wall_s <= 0:
            return "no timing"
        return (
            f"{self.chunks:,} chunks in {self.wall_s:.0f}s "
            f"({self.chunks / max(self.wall_s, 1e-9):.0f}/s)\n"
            f"      GPU busy {100*self.gpu_busy_s/self.wall_s:.0f}% | "
            f"GPU starved {100*self.gpu_wait_s/self.wall_s:.0f}% | "
            f"write {100*self.write_s/self.wall_s:.0f}%\n"
            f"      producers blocked (backpressure, healthy): "
            f"reader {self.reader_block_s:.0f}s, chunkers {self.chunker_block_s:.0f}s"
        )


class PipelineStalled(RuntimeError):
    pass


class ShardPipeline:
    """Overlapped read -> chunk -> embed for one parquet shard.

    One shard at a time on purpose. Shard boundaries are where the manifest is
    committed, and interleaving shards would make "which shards are safely on
    disk" ambiguous after a crash -- the exact bug that makes a resume duplicate
    data.
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
        stall_timeout_s: float = 900.0,
    ):
        self.shard_path = shard_path
        self.shard_id = shard_id
        self.chunk_fn = chunk_fn
        self.embed_fn = embed_fn
        self.write_fn = write_fn
        self.n_chunkers = n_chunkers
        self.row_group_rows = row_group_rows
        self.embed_batch = embed_batch
        self.stall_timeout_s = stall_timeout_s

        self.raw_q: queue.Queue = queue.Queue(maxsize=raw_queue_size)
        self.embed_q: queue.Queue = queue.Queue(maxsize=embed_queue_size)
        self.stop = threading.Event()
        # Completion is signalled by Events, NOT by sentinels travelling through
        # the bounded queues. A sentinel `put` into a full queue can time out and
        # be dropped, after which the consumer waits forever for a message that
        # no longer exists -- which is exactly how this pipeline aborted a shard
        # at 3.25M chunks with "chunker: waited >900s for input". Events cannot
        # be lost to backpressure.
        self.reader_done = threading.Event()
        self.chunkers_live = n_chunkers
        self.stats = PipelineStats()
        self._lock = threading.Lock()

    # ----------------------------------------------------- queue helpers

    def _put(self, q: queue.Queue, item, what: str) -> float:
        """Blocking put that honours shutdown and never hangs forever."""
        t0 = time.perf_counter()
        while not self.stop.is_set():
            try:
                q.put(item, timeout=POLL)
                return time.perf_counter() - t0
            except queue.Full:
                if time.perf_counter() - t0 > self.stall_timeout_s:
                    with self._lock:
                        self.stats.errors.append(
                            f"{what}: blocked >{self.stall_timeout_s:.0f}s on a full queue")
                    self.stop.set()
                    return time.perf_counter() - t0
        return time.perf_counter() - t0

    # ------------------------------------------------------------ stages

    def _reader(self) -> None:
        """Decode parquet row-batches. pyarrow releases the GIL in here."""
        import pyarrow.parquet as pq

        try:
            pf = pq.ParquetFile(self.shard_path)
            base_row = 0
            for batch in pf.iter_batches(batch_size=self.row_group_rows, columns=["text"]):
                if self.stop.is_set():
                    break
                texts = batch.column("text").to_pylist()
                blocked = self._put(self.raw_q, (base_row, texts), "reader")
                with self._lock:
                    self.stats.reader_block_s += blocked
                    self.stats.read_batches += 1
                base_row += len(texts)
        except Exception as exc:
            with self._lock:
                self.stats.errors.append(f"reader: {type(exc).__name__}: {exc}")
            self.stop.set()
        finally:
            self.reader_done.set()

    def _chunker(self) -> None:
        """Pure-Python string slicing. Holds the GIL, runs while the GPU works."""
        pending_txt: list[str] = []
        pending_co: list[tuple[int, int, int, int]] = []

        def flush() -> None:
            nonlocal pending_txt, pending_co
            if not pending_txt:
                return
            blocked = self._put(self.embed_q, (pending_txt, pending_co), "chunker")
            with self._lock:
                self.stats.chunker_block_s += blocked
            pending_txt, pending_co = [], []

        try:
            waited = 0.0
            while not self.stop.is_set():
                try:
                    item = self.raw_q.get(timeout=POLL)
                    waited = 0.0
                except queue.Empty:
                    # Exit only when the reader is finished AND nothing is left.
                    # Checking both, in that order, is what makes this race-free.
                    if self.reader_done.is_set() and self.raw_q.empty():
                        break
                    # Stall detection. The event-based rewrite originally
                    # dropped this by polling the queue directly -- a wedged
                    # (alive but stuck) reader then starved chunkers FOREVER
                    # with no error, resurrecting the silent-hang failure the
                    # detector was built to kill. Caught by a test.
                    waited += POLL
                    if waited > self.stall_timeout_s:
                        with self._lock:
                            self.stats.errors.append(
                                f"chunker: waited >{self.stall_timeout_s:.0f}s for input")
                        self.stop.set()
                        break
                    continue
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
            self.stop.set()
        finally:
            with self._lock:
                self.chunkers_live -= 1

    # -------------------------------------------------------------- run

    def run(self, progress_every: int = 250_000) -> PipelineStats:
        """Drive the pipeline; GPU work and disk writes happen on THIS thread."""
        t_start = time.perf_counter()

        reader = threading.Thread(target=self._reader, name="reader", daemon=True)
        chunkers = [
            threading.Thread(target=self._chunker, name=f"chunker{i}", daemon=True)
            for i in range(self.n_chunkers)
        ]
        reader.start()
        for c in chunkers:
            c.start()

        next_report = progress_every
        waited = 0.0
        try:
            while not self.stop.is_set():
                t0 = time.perf_counter()
                try:
                    item = self.embed_q.get(timeout=POLL)
                    waited = 0.0
                except queue.Empty:
                    self.stats.gpu_wait_s += time.perf_counter() - t0
                    with self._lock:
                        live = self.chunkers_live
                    if live == 0 and self.embed_q.empty():
                        break
                    # same stall detection as the chunkers: live producers that
                    # never produce must become an error, not an eternal wait
                    waited += POLL
                    if waited > self.stall_timeout_s:
                        self.stats.errors.append(
                            f"main: waited >{self.stall_timeout_s:.0f}s for input "
                            f"({live} chunker(s) alive but producing nothing)")
                        self.stop.set()
                        break
                    continue
                self.stats.gpu_wait_s += time.perf_counter() - t0

                texts, coords = item

                t0 = time.perf_counter()
                arrays = self.embed_fn(texts)
                self.stats.gpu_busy_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                self.write_fn(arrays, coords)
                self.stats.write_s += time.perf_counter() - t0

                self.stats.chunks += len(texts)
                # bytes of chunk text actually indexed. Tracked because the
                # manifest field existed, was never populated, and made
                # bench_latency print "from 0.0 GB of text" for a 13.6M-chunk
                # index -- a number that was wrong rather than merely absent.
                self.stats.text_bytes += sum(len(t) for t in texts)
                if self.stats.chunks >= next_report:
                    el = time.perf_counter() - t_start
                    print(f"      {self.stats.chunks:,} chunks  "
                          f"{self.stats.chunks/el:.0f}/s  "
                          f"GPU busy {100*self.stats.gpu_busy_s/el:.0f}%  "
                          f"starved {100*self.stats.gpu_wait_s/el:.0f}%", flush=True)
                    next_report += progress_every
        except Exception as exc:
            self.stats.errors.append(f"main: {type(exc).__name__}: {exc}")
            self.stop.set()
        finally:
            self.stop.set()          # release any producer still blocked
            reader.join(timeout=60)
            for c in chunkers:
                c.join(timeout=60)

        self.stats.wall_s = time.perf_counter() - t_start
        return self.stats


def iter_shards(cache: Path) -> Iterator[Path]:
    yield from sorted(p for p in cache.rglob("*.parquet") if p.is_file())
