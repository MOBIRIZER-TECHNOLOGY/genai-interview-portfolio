"""
Tests for the two-stage `ScaleIndex` search and the uncovered pipeline branches.

Written for the coverage pass: `search.py` sat at 49% because only its guard
paths were exercised (by the crash-safety tests). Here the whole search flow
runs against a small, fully synthetic index — real files on disk, real memmaps,
real parquet for the lazy text fetch — with only the GPU embedder faked.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "07_rag_at_scale"))

from scale.quantize import (  # noqa: E402
    Int8Calibration, binary_encode, calibrate_int8, hamming_search, int8_encode,
    rescore,
)
from scale.search import Hit, ScaleIndex  # noqa: E402

DIM = 384
N_DOCS = 40


# ----------------------------------------------------------------- fixture


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    """A complete miniature index: parquet corpus + binary/int8/coords + manifest.

    Every chunk's vector is derived from its text via a deterministic
    vocabulary embedding, so nearest-neighbour results are semantically
    meaningful, not arbitrary.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path_factory.mktemp("scale")
    cache = root / "hf"
    cache.mkdir()
    out = root / "index"
    out.mkdir()

    topics = ["warehouse robotics dispatch auction",
              "barcode vision confidence camera",
              "telemetry ingest buffer hypertable",
              "retention policy vision frames privacy"]
    docs = [f"Document about {topics[i % 4]} number {i}. " * 6 for i in range(N_DOCS)]
    pq.write_table(pa.table({"text": docs}), cache / "shard_000.parquet")

    def embed(text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in text.lower().split():
            rng = np.random.default_rng(abs(hash(tok)) % (2**32))
            v += rng.normal(size=DIM).astype(np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    # one chunk per document: coords = (shard 0, row i, span of the text)
    vecs = np.stack([embed(d) for d in docs])
    cal = calibrate_int8(vecs)
    coords = np.array([[0, i, 0, len(docs[i])] for i in range(N_DOCS)], dtype=np.int64)

    (out / "binary.u8").write_bytes(binary_encode(vecs).tobytes())
    (out / "int8.i8").write_bytes(int8_encode(vecs, cal).tobytes())
    (out / "coords.i64").write_bytes(coords.tobytes())
    cal.save(out / "int8_calib.json")
    (out / "manifest.json").write_text(json.dumps({
        "model": "BAAI/bge-small-en-v1.5", "dim": DIM, "n_chunks": N_DOCS,
        "shards_done": ["shard_000.parquet"], "bytes_text": sum(map(len, docs)),
        "cache": str(cache),
    }), encoding="utf-8")

    return out, embed, docs


# ------------------------------------------------------------------ search


def test_open_reads_manifest_and_stats(index_dir):
    out, _, docs = index_dir
    idx = ScaleIndex.open(out)
    assert idx.n == N_DOCS and idx.dim == DIM
    st = idx.stats()
    assert st["n_chunks"] == N_DOCS
    assert st["shards_indexed"] == 1
    # 40 x 384 vectors round to 0.0 GB at every precision -- assert the exact
    # rounded arithmetic rather than an ordering that collapses at tiny scale
    assert st["binary_gb"] == round(N_DOCS * DIM / 8 / 1e9, 3)
    assert st["int8_gb"] == round(N_DOCS * DIM / 1e9, 2)
    assert st["float32_avoided_gb"] == round(N_DOCS * DIM * 4 / 1e9, 2)


def test_search_vector_two_stage(index_dir):
    out, embed, docs = index_dir
    idx = ScaleIndex.open(out)
    q = embed("telemetry buffer hypertable ingest")
    hits = idx.search_vector(q, k=5, candidates=20, fetch_text=False)

    assert len(hits) == 5
    assert all(isinstance(h, Hit) for h in hits)
    assert [h.rank for h in hits] == list(range(5))
    assert hits[0].score >= hits[-1].score
    # top hit must actually be a telemetry document
    assert "telemetry" in docs[hits[0].row]
    # coords round-trip
    assert hits[0].shard == 0 and hits[0].char_start == 0


def test_attach_text_fetches_correct_span(index_dir):
    out, embed, docs = index_dir
    idx = ScaleIndex.open(out)
    hits = idx.search_vector(embed("barcode vision confidence"), k=3,
                             candidates=15, fetch_text=True)
    for h in hits:
        assert h.text is not None, "lazy text fetch failed"
        assert h.text == docs[h.row][h.char_start:h.char_end], \
            "fetched text must match the coordinate span exactly"
    assert "barcode" in hits[0].text


def test_search_string_path_uses_query_prefix(index_dir, monkeypatch):
    """The str overload must route through encode_query (BGE prefix included)."""
    out, embed, docs = index_dir
    idx = ScaleIndex.open(out)

    seen = {}

    def fake_encode_query(query: str):
        seen["query"] = query
        return embed(query)

    monkeypatch.setattr(idx, "encode_query", fake_encode_query)
    hits = idx.search("dispatch auction robots", k=3, fetch_text=False)
    assert seen["query"] == "dispatch auction robots"
    assert len(hits) == 3


def test_candidates_clamped_to_corpus(index_dir):
    out, embed, _ = index_dir
    idx = ScaleIndex.open(out)
    hits = idx.search_vector(embed("robotics"), k=10, candidates=10_000,
                             fetch_text=False)
    assert len(hits) == 10


# -------------------------------------------------- quantize: gaps


def test_hamming_search_accepts_1d_query():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, DIM)).astype(np.float32)
    codes = binary_encode(X)
    d, i = hamming_search(codes[0], codes, k=3)     # 1-D input path
    assert d.shape == (1, 3) and i[0, 0] == 0


def test_rescore_helper_orders_candidates():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, DIM)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cal = calibrate_int8(X)
    codes = int8_encode(X, cal)
    scores, order = rescore(X[7], codes, cal, k=5)
    assert order[0] == 7, "a vector must rescore itself to the top"
    assert list(scores) == sorted(scores, reverse=True)


# -------------------------------------------------- pipeline: stall paths


def test_stall_detector_reports_starved_consumer(tmp_path):
    """A producer that never produces must trip the consumer's stall timeout.

    Covers the _get timeout branch -- the code that turned the original silent
    deadlock into a diagnosable message.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scale.pipeline import ShardPipeline

    shard = tmp_path / "s.parquet"
    pq.write_table(pa.table({"text": ["some text " * 50]}), shard)

    hold = threading.Event()          # chunk_fn blocks -> main starves

    def stuck_chunker(text):
        # long enough for main's 2s stall detector to fire, short enough that
        # the shutdown join (which waits on this thread) completes in-test
        hold.wait(timeout=6)
        return []

    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=stuck_chunker,
        embed_fn=lambda t: ("E", t), write_fn=lambda a, c: None,
        n_chunkers=1, stall_timeout_s=2.0,
    )

    box = {}
    t = threading.Thread(target=lambda: box.setdefault("stats", pipe.run()), daemon=True)
    t.start()
    t.join(timeout=30)
    hold.set()
    assert not t.is_alive(), "stall detector failed to terminate the pipeline"
    assert any("waited" in e for e in box["stats"].errors), box["stats"].errors


def test_stall_detector_reports_blocked_producer(tmp_path):
    """A consumer that never consumes must trip the producer's stall timeout."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scale.pipeline import ShardPipeline

    shard = tmp_path / "s.parquet"
    pq.write_table(pa.table({"text": ["words " * 200] * 50}), shard)

    hold = threading.Event()

    def frozen_embed(texts):
        hold.wait(timeout=6)          # embed_q backs up behind this
        return ("E", texts)

    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=lambda t: [(0, min(200, len(t)))],
        embed_fn=frozen_embed, write_fn=lambda a, c: None,
        n_chunkers=2, row_group_rows=5, embed_batch=4,
        raw_queue_size=1, embed_queue_size=1, stall_timeout_s=2.0,
    )

    box = {}
    t = threading.Thread(target=lambda: box.setdefault("stats", pipe.run()), daemon=True)
    t.start()
    t.join(timeout=40)
    hold.set()
    assert not t.is_alive()
    assert box["stats"].errors, "a wedged consumer must surface as a recorded error"


def test_stats_summary_renders(tmp_path):
    """PipelineStats.summary() is what the operator reads -- keep it printable."""
    from scale.pipeline import PipelineStats

    s = PipelineStats(chunks=1000, gpu_busy_s=8.0, gpu_wait_s=1.0,
                      write_s=0.5, wall_s=10.0)
    text = s.summary()
    assert "1,000 chunks" in text and "GPU busy 80%" in text
    assert PipelineStats().summary() == "no timing"
