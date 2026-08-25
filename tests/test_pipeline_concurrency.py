"""
Concurrency tests for `scale/pipeline.py` — the producer/consumer index builder.

## The bug these exist for

The pipeline aborted a shard after successfully embedding **3.25 M chunks**:

    ERROR chunker: waited >900s for input
    aborting: shard failed, manifest not committed

Completion was signalled by sentinels pushed through the **bounded** queues:

    try: self.raw_q.put(SENTINEL, timeout=30)
    except queue.Full: pass          # <- silently drops the shutdown signal

Under sustained backpressure that `put` times out, the sentinel is dropped, and
a consumer waits forever for a message that no longer exists.

**It only reproduces under backpressure.** A test with roomy queues and a fast
consumer passes on the broken code, because the sentinel always fits. So every
test here deliberately runs with tiny queues and a slow consumer — the state a
real 50-minute shard spends most of its time in.

## Why a hang must fail rather than hang

A deadlocked pipeline consumes 0% CPU and produces no output; left alone it
would hang the whole test session. Each test therefore runs `run()` on a daemon
thread and joins with a timeout, so a deadlock surfaces as a **failed assertion
with a diagnostic**, not a stuck CI job.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "07_rag_at_scale"))

from scale.pipeline import ShardPipeline  # noqa: E402

pytestmark = pytest.mark.slow


# --------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    """A small real parquet shard, so the reader exercises its actual code path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path_factory.mktemp("shard")
    path = d / "test_000.parquet"
    # 800 docs, each long enough to yield several chunks
    texts = [f"Document {i}. " + ("warehouse robotics telemetry dispatch auction. " * 40)
             for i in range(800)]
    pq.write_table(pa.table({"text": texts}), path)
    return path


def fixed_chunker(text: str) -> list[tuple[int, int]]:
    """Deterministic 400-char chunks — no dependence on the real chunker's tuning."""
    return [(i, min(i + 400, len(text))) for i in range(0, len(text), 400)
            if min(i + 400, len(text)) - i >= 60]


def expected_chunk_count(shard: Path) -> int:
    import pyarrow.parquet as pq

    total = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=1000, columns=["text"]):
        for t in batch.column("text").to_pylist():
            if t:
                total += len(fixed_chunker(t))
    return total


class Collector:
    """Thread-safe sink standing in for embed + write."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.texts: list[str] = []
        self.coords: list[tuple] = []
        self.lock = threading.Lock()

    def embed(self, texts: list[str]):
        if self.delay:
            time.sleep(self.delay)          # simulate GPU work -> backpressure
        return ("EMBEDDED", list(texts))

    def write(self, arrays, coords) -> None:
        _, texts = arrays
        with self.lock:
            self.texts.extend(texts)
            self.coords.extend(coords)


def run_with_timeout(pipe: ShardPipeline, timeout: float = 90.0):
    """Run the pipeline on a daemon thread; a deadlock fails instead of hanging."""
    box = {}

    def target():
        try:
            box["stats"] = pipe.run(progress_every=10**9)
        except BaseException as exc:        # noqa: BLE001 - surfaced below
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), (
        f"pipeline did not terminate within {timeout}s -- DEADLOCK. "
        "This is the lost-sentinel failure mode: a completion signal was dropped "
        "by a full bounded queue and a consumer is waiting for it forever."
    )
    if "error" in box:
        raise box["error"]
    return box["stats"]


# --------------------------------------- the exact lost-sentinel condition


def test_completion_is_never_signalled_through_a_bounded_queue(shard):
    """THE regression test — structural, deterministic, and it actually fires.

    My first attempt tried to *recreate* the bug's timing: stall the consumer so
    a sentinel `put` times out. It passed on known-broken code, twice. The
    condition is genuinely hard to hit on demand — the consumer must be stalled
    while `raw_q` is full AND the reader has finished, but a gated consumer
    blocks the reader before it can finish, so the state is unreachable.

    Chasing the timing was the wrong instinct. The **invariant** is what matters:

        a completion signal must not travel through a channel that can drop it.

    A bounded queue can always reject a `put` under backpressure, so any
    sentinel-based shutdown is broken *by construction* regardless of whether a
    given run happens to hit the timeout. This test spies on both queues and
    asserts no sentinel is ever enqueued — which fails immediately on the old
    design and cannot flake on the new one.
    """
    import queue as _queue

    from scale import pipeline as pl

    enqueued: list[str] = []

    class SpyQueue(_queue.Queue):
        def __init__(self, name, maxsize):
            super().__init__(maxsize=maxsize)
            self._name = name

        def put(self, item, *a, **kw):
            if item is pl.SENTINEL:
                enqueued.append(self._name)
            return super().put(item, *a, **kw)

    sink = Collector(delay=0.005)
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=3, row_group_rows=50, embed_batch=64,
        raw_queue_size=2, embed_queue_size=2, stall_timeout_s=30.0,
    )
    pipe.raw_q = SpyQueue("raw_q", 2)
    pipe.embed_q = SpyQueue("embed_q", 2)

    stats = run_with_timeout(pipe, timeout=120.0)

    assert not enqueued, (
        f"completion was signalled through bounded queue(s) {sorted(set(enqueued))}. "
        "A bounded queue can reject a put under backpressure, so the signal can be "
        "silently dropped and a consumer will wait for it forever. Signal "
        "completion out-of-band (threading.Event) instead."
    )
    assert not stats.errors
    assert stats.chunks == expected_chunk_count(shard), "chunks lost"


def test_completion_uses_out_of_band_signals(shard):
    """The positive form of the same invariant: the mechanism must exist."""
    sink = Collector()
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=3,
    )
    assert isinstance(pipe.reader_done, threading.Event), (
        "reader completion must be an Event, not a queued message")
    assert pipe.chunkers_live == 3, (
        "chunker completion must be tracked by a counter, not by counting sentinels")

    run_with_timeout(pipe, timeout=120.0)
    assert pipe.reader_done.is_set(), "reader_done must be set on completion"
    assert pipe.chunkers_live == 0, "every chunker must decrement the live counter"


def test_survives_a_long_consumer_stall(shard):
    """Termination must not depend on how long the consumer takes.

    Weaker than the invariant test above, but it exercises the real threads
    through a multi-second stall rather than asserting on structure.
    """
    gate = threading.Event()
    gate.set()
    sink = Collector()

    def gated_embed(texts):
        gate.wait(timeout=120)
        return sink.embed(texts)

    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=gated_embed, write_fn=sink.write,
        n_chunkers=3, row_group_rows=50, embed_batch=64,
        raw_queue_size=2, embed_queue_size=2, stall_timeout_s=120.0,
    )

    box = {}
    t = threading.Thread(
        target=lambda: box.setdefault("stats", pipe.run(progress_every=10**9)),
        daemon=True)
    t.start()

    time.sleep(0.5)
    gate.clear()          # stall the consumer mid-flight, queues back up
    time.sleep(6.0)
    gate.set()            # release

    t.join(timeout=120)
    assert not t.is_alive(), "pipeline did not terminate after a 6s consumer stall"
    assert box["stats"].chunks == expected_chunk_count(shard), "chunks lost across the stall"


# ------------------------------------------------- termination under load


def test_terminates_under_severe_backpressure(shard):
    """THE regression test.

    Queues at minimum size and a consumer slower than the producers, so the
    queues are full when the reader finishes and has to signal completion.
    That is exactly the state in which a sentinel gets dropped.
    """
    sink = Collector(delay=0.02)
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=3,
        row_group_rows=50,
        embed_batch=64,
        raw_queue_size=1,        # minimum -> reader blocks almost immediately
        embed_queue_size=1,      # minimum -> chunkers block almost immediately
        stall_timeout_s=30.0,
    )
    stats = run_with_timeout(pipe, timeout=120.0)
    assert not stats.errors, f"pipeline reported errors: {stats.errors}"
    assert stats.chunks > 0


def test_terminates_with_more_chunkers_than_work(shard):
    """More chunkers than row-batches: some never receive a single item.

    With the old design the reader emitted exactly `n_chunkers` sentinels; any
    imbalance in who consumed them left a thread waiting. Event-based shutdown
    has to be immune to that.
    """
    sink = Collector()
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=8,
        row_group_rows=100_000,   # one single batch for 8 chunkers
        embed_batch=512,
        stall_timeout_s=30.0,
    )
    stats = run_with_timeout(pipe, timeout=90.0)
    assert not stats.errors
    assert stats.chunks > 0


@pytest.mark.parametrize("n_chunkers", [1, 2, 4])
def test_terminates_across_chunker_counts(shard, n_chunkers):
    sink = Collector()
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=n_chunkers, row_group_rows=100, embed_batch=128,
        raw_queue_size=2, embed_queue_size=2, stall_timeout_s=30.0,
    )
    stats = run_with_timeout(pipe, timeout=90.0)
    assert not stats.errors
    assert stats.chunks > 0


# ------------------------------------------------------------ no data loss


def test_no_chunks_are_lost_under_backpressure(shard):
    """Every chunk the chunker produces must reach the writer.

    A shutdown race that drains early would lose the tail of a shard -- and
    since the manifest commits per shard, that loss would be recorded as
    success. Silent truncation is worse than a crash.
    """
    expected = expected_chunk_count(shard)
    sink = Collector(delay=0.005)
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=3, row_group_rows=50, embed_batch=64,
        raw_queue_size=1, embed_queue_size=1, stall_timeout_s=30.0,
    )
    stats = run_with_timeout(pipe, timeout=120.0)

    assert not stats.errors
    assert len(sink.texts) == expected, (
        f"lost {expected - len(sink.texts)} of {expected} chunks")
    assert stats.chunks == expected, (
        f"stats.chunks={stats.chunks} disagrees with {expected} delivered")


def test_texts_and_coords_stay_paired(shard):
    """Coordinates must line up with their text, even across 3 chunker threads.

    Queue items carry (texts, coords) together precisely so interleaving cannot
    desynchronise them. If it ever could, every vector would point at the wrong
    source span -- and nothing would error.
    """
    sink = Collector()
    pipe = ShardPipeline(
        shard_path=shard, shard_id=7,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=4, row_group_rows=100, embed_batch=128, stall_timeout_s=30.0,
    )
    run_with_timeout(pipe, timeout=90.0)

    assert len(sink.texts) == len(sink.coords)
    for text, (sid, _row, a, b) in zip(sink.texts, sink.coords):
        assert sid == 7, "shard id must be stamped on every coord"
        assert b - a == len(text), "coord span must match its text length"


def test_no_duplicate_chunks(shard):
    """A chunker consuming an item twice would duplicate work invisibly."""
    sink = Collector()
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=4, row_group_rows=50, embed_batch=64,
        raw_queue_size=2, embed_queue_size=2, stall_timeout_s=30.0,
    )
    run_with_timeout(pipe, timeout=90.0)
    assert len(set(sink.coords)) == len(sink.coords), "duplicate (shard,row,start,end)"


# ----------------------------------------------------------- backpressure


def test_queues_stay_bounded(shard):
    """Bounded queues are what stop a fast reader materialising the whole shard.

    Unbounded queues in front of a GPU turn a producer/consumer speedup into an
    out-of-memory crash.
    """
    sink = Collector(delay=0.01)
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=3, row_group_rows=50, embed_batch=64,
        raw_queue_size=2, embed_queue_size=3, stall_timeout_s=30.0,
    )

    breached = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            if pipe.raw_q.qsize() > 2 or pipe.embed_q.qsize() > 3:
                breached.append((pipe.raw_q.qsize(), pipe.embed_q.qsize()))
            time.sleep(0.005)

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    try:
        run_with_timeout(pipe, timeout=120.0)
    finally:
        stop.set()
    assert not breached, f"queue exceeded its maxsize: {breached[:3]}"


def test_backpressure_is_recorded(shard):
    """Producers blocking is healthy and must be visible in the stats.

    Without it you cannot tell "GPU is the bottleneck" from "producers are".
    """
    sink = Collector(delay=0.02)
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=3, row_group_rows=50, embed_batch=64,
        raw_queue_size=1, embed_queue_size=1, stall_timeout_s=30.0,
    )
    stats = run_with_timeout(pipe, timeout=120.0)
    assert stats.chunker_block_s > 0 or stats.reader_block_s > 0, (
        "a slow consumer must block producers; zero block time means the "
        "instrumentation is not wired up")
    assert stats.gpu_busy_s > 0
    assert stats.wall_s > 0


# ------------------------------------------------------------- error paths


def test_embed_failure_terminates_and_reports(shard):
    """A raising consumer must stop the pipeline, not wedge the producers."""
    def boom(texts):
        raise RuntimeError("simulated GPU OOM")

    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=boom, write_fn=lambda a, c: None,
        n_chunkers=3, row_group_rows=50, embed_batch=64,
        raw_queue_size=1, embed_queue_size=1, stall_timeout_s=30.0,
    )
    stats = run_with_timeout(pipe, timeout=90.0)
    assert stats.errors, "a failing embed_fn must be recorded in stats.errors"
    assert any("simulated GPU OOM" in e for e in stats.errors)


def test_chunker_failure_terminates_and_reports(shard):
    """Same for a producer: fail fast, do not strand the consumer."""
    def bad_chunker(text: str):
        raise ValueError("simulated chunker failure")

    sink = Collector()
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=bad_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=3, row_group_rows=50, embed_batch=64, stall_timeout_s=30.0,
    )
    stats = run_with_timeout(pipe, timeout=90.0)
    assert stats.errors
    assert any("simulated chunker failure" in e for e in stats.errors)


def test_missing_shard_terminates(tmp_path):
    """An unreadable shard must surface as an error, not a hang."""
    sink = Collector()
    pipe = ShardPipeline(
        shard_path=tmp_path / "does_not_exist.parquet", shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=2, stall_timeout_s=15.0,
    )
    stats = run_with_timeout(pipe, timeout=60.0)
    assert stats.errors
    assert any("reader:" in e for e in stats.errors)


def test_empty_shard_terminates_cleanly(tmp_path):
    """Zero rows: every consumer must still see completion and exit."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"text": pa.array([], type=pa.string())}), path)

    sink = Collector()
    pipe = ShardPipeline(
        shard_path=path, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=4, stall_timeout_s=15.0,
    )
    stats = run_with_timeout(pipe, timeout=60.0)
    assert not stats.errors
    assert stats.chunks == 0


def test_text_bytes_are_accounted(shard):
    """`Stats.text_bytes` must equal the bytes of chunk text actually indexed.

    The manifest carried a `bytes_text` field that was initialised to zero and
    then re-assigned to itself on every shard commit -- a no-op that looked like
    bookkeeping. The visible symptom was `bench_latency.py` printing "13,597,793
    chunks from 0.0 GB of text": not a missing number, a wrong one.

    Sizes are the easiest thing in a pipeline to leave unwired, because nothing
    fails when they are wrong.
    """
    sink = Collector()
    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=fixed_chunker, embed_fn=sink.embed, write_fn=sink.write,
        n_chunkers=2, row_group_rows=50, embed_batch=64,
        raw_queue_size=8, embed_queue_size=8, stall_timeout_s=30.0,
    )
    stats = run_with_timeout(pipe, timeout=120.0)

    assert not stats.errors
    assert stats.text_bytes == sum(len(t) for t in sink.texts), (
        "text_bytes must count exactly the text that reached the embedder"
    )
    assert stats.text_bytes > 0, "a non-empty shard must report non-zero text bytes"
