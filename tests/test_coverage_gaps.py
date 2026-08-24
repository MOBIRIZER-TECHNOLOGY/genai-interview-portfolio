"""
Targeted tests for the branches the component suites left uncovered.

Each test names the line(s) it exists for. Coverage-driven tests are only worth
having when the branch is *reachable in production* — every one here is: guard
raises, ablation paths, stall detection, and the ask() orchestration.

One deletion came out of this pass instead of a test: `ShardPipeline._get` was
12 statements of dead code — orphaned by the event-based shutdown rewrite, kept
compiling, covered nothing. Dead code is not a coverage problem, it is a
maintenance liability; the fix is `git rm`, not a test.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_rag_local"))
sys.path.insert(0, str(ROOT / "07_rag_at_scale"))

from rag.chunking import chunk_markdown  # noqa: E402
from rag.pipeline import RagPipeline, RagResult  # noqa: E402
from rag.store import VectorStore  # noqa: E402
from tests.test_rag_components import CORPUS, FakeEmbedder  # noqa: E402


# ------------------------------------------------- rag/pipeline.py 46,101,109-113


@pytest.fixture()
def cpu_pipeline():
    """A real RagPipeline over the fake embedder — no GPU, no reranker."""
    emb = FakeEmbedder()
    vecs = emb.encode_passages([t for _, t in CORPUS])
    records = [
        {"id": cid, "text": t, "body": t, "source": f"{cid}.md",
         "breadcrumb": f"{cid}.md > s", "n_tokens": 10, "meta": {}}
        for cid, t in CORPUS
    ]
    store = VectorStore.build(vecs, records, "fake-embedder")
    return RagPipeline(store, emb, reranker=None, top_k=3)


def test_pipeline_truncates_without_reranker(cpu_pipeline):
    """Line 101: the no-reranker path must still honour top_k."""
    hits, r_ms, rr_ms = cpu_pipeline.retrieve("dispatch auction", mode="hybrid")
    assert len(hits) <= 3
    assert rr_ms >= 0                      # the rerank timer still runs (as ~0)


def test_pipeline_ask_orchestration(cpu_pipeline, monkeypatch):
    """Lines 109-113 + 46: ask() wiring and the total_ms property.

    generate_answer is patched at the pipeline module's import site, so the
    retrieve -> generate -> RagResult assembly runs for real.
    """
    import rag.pipeline as P
    from rag.generate import Answer

    def fake_generate(question, hits, model="m"):
        return Answer(text="ans [1]", citations=[1], valid_citations=[1],
                      invalid_citations=[], abstained=False)

    monkeypatch.setattr(P, "generate_answer", fake_generate)
    result = cpu_pipeline.ask("what is the auction period?")
    assert isinstance(result, RagResult)
    assert result.answer.grounded
    assert result.total_ms == pytest.approx(
        result.retrieve_ms + result.rerank_ms + result.generate_ms)


# ------------------------------------------------------ rag/chunking.py 130,142


def test_chunk_markdown_heading_level_jump(tmp_path):
    """Line 130: an H3 directly after H1 must pad the breadcrumb stack."""
    doc = tmp_path / "jump.md"
    doc.write_text("# Top\n\nintro text that is long enough to keep\n\n"
                   "### Deep\n\nnested body text that is long enough to keep\n",
                   encoding="utf-8")
    chunks = chunk_markdown(doc)
    deep = [c for c in chunks if "Deep" in c.breadcrumb]
    assert deep, "H3-after-H1 section lost"
    # the empty padding level is dropped from the rendered breadcrumb
    assert "Top > Deep" in deep[0].breadcrumb


def test_chunk_markdown_skips_empty_section(tmp_path):
    """Line 142: a heading with no body must not emit an empty chunk."""
    doc = tmp_path / "empty.md"
    doc.write_text("# A\n\n## Empty\n\n## Full\n\nreal content that is long "
                   "enough to survive the length gate\n", encoding="utf-8")
    chunks = chunk_markdown(doc)
    assert all(c.body.strip() for c in chunks)
    assert not any("Empty" in c.breadcrumb for c in chunks)


# ----------------------------------------------------------- rag/store.py 99


def test_store_skips_negative_indices():
    """Line 99: FAISS pads with -1 when k exceeds the index; must be skipped."""
    import faiss

    emb = FakeEmbedder()
    vecs = emb.encode_passages([t for _, t in CORPUS[:2]])
    records = [{"id": c, "text": t, "body": t, "source": "s", "breadcrumb": "b",
                "n_tokens": 1, "meta": {}} for c, t in CORPUS[:2]]
    store = VectorStore.build(vecs, records, "fake")
    # force the -1 padding path by searching the raw index beyond its size
    scores, idxs = store.index.search(vecs[:1], 5)
    assert -1 in idxs[0], "precondition: FAISS should pad with -1 here"
    store_hits = store.search(vecs[0], k=5)
    assert len(store_hits) == 2            # padding rows silently skipped


# ------------------------------------------------ scale/search.py 100,120-132,196


def test_index_refuses_short_files(tmp_path):
    """Line 100: files shorter than the manifest -> hard refusal."""
    from scale.quantize import Int8Calibration
    from scale.search import ScaleIndex

    d = tmp_path / "idx"
    d.mkdir()
    dim = 384
    (d / "binary.u8").write_bytes(b"\x00" * (5 * dim // 8))
    (d / "int8.i8").write_bytes(b"\x00" * (5 * dim))
    (d / "coords.i64").write_bytes(b"\x00" * (5 * 32))
    Int8Calibration([-1.0] * dim, [1.0] * dim, dim, 0).save(d / "int8_calib.json")
    (d / "manifest.json").write_text(json.dumps(
        {"model": "m", "dim": dim, "n_chunks": 10, "shards_done": ["x"]}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="shorter than the manifest"):
        ScaleIndex.open(d)


def test_encode_query_applies_bge_prefix(monkeypatch):
    """Lines 120-132: the lazy embedder load and the asymmetric query prefix."""
    import scale.search as S

    seen = {}

    class FakeST:
        def __init__(self, name, device=None):
            seen["model"] = name

        def half(self):
            return self

        def encode(self, texts, **kw):
            seen["texts"] = texts
            return np.ones((len(texts), 8), dtype=np.float32)

    fake_mod = type(sys)("sentence_transformers")
    fake_mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    idx = object.__new__(S.ScaleIndex)          # skip open(); unit-test the method
    idx.manifest = {"model": "BAAI/bge-small-en-v1.5"}
    idx._model = None

    v = S.ScaleIndex.encode_query(idx, "how are frames retained")
    assert seen["model"] == "BAAI/bge-small-en-v1.5"
    assert seen["texts"][0].startswith("Represent this sentence for searching")
    assert v.dtype == np.float32
    # second call must reuse the cached model (lazy-load once)
    S.ScaleIndex.encode_query(idx, "again")
    assert idx._model is not None


def test_attach_text_skips_unknown_shard(tmp_path):
    """Line 196: a hit pointing at a shard that does not exist is left textless."""
    from scale.search import Hit, ScaleIndex

    idx = object.__new__(ScaleIndex)
    idx.manifest = {"cache": str(tmp_path)}      # empty cache dir: no shards
    ScaleIndex._shard_path.cache_clear()
    hits = [Hit(rank=0, chunk_id=0, score=1.0, hamming=0,
                shard=99, row=0, char_start=0, char_end=10)]
    idx.attach_text(hits)
    assert hits[0].text is None


# ----------------------------------------- scale/pipeline.py chunker stall + print


def test_chunker_stall_when_reader_wedges(tmp_path):
    """The chunker-side stall branch: reader alive, producing nothing.

    Runs the chunker loop directly with no reader thread at all -- raw_q stays
    empty, reader_done never fires, and the stall detector must convert an
    eternal wait into a recorded error.
    """
    from scale.pipeline import ShardPipeline

    pipe = ShardPipeline(
        shard_path=tmp_path / "unused.parquet", shard_id=0,
        chunk_fn=lambda t: [], embed_fn=lambda t: t, write_fn=lambda a, c: None,
        n_chunkers=1, stall_timeout_s=2.0,
    )
    t = threading.Thread(target=pipe._chunker, daemon=True)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "chunker must not wait forever on a wedged reader"
    assert any("chunker: waited" in e for e in pipe.stats.errors)
    assert pipe.chunkers_live == 0               # finally-block bookkeeping ran


def test_progress_report_prints(tmp_path, capsys):
    """Lines 306-311: the operator-facing progress line."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scale.pipeline import ShardPipeline

    shard = tmp_path / "s.parquet"
    pq.write_table(pa.table({"text": ["words " * 100] * 200}), shard)

    pipe = ShardPipeline(
        shard_path=shard, shard_id=0,
        chunk_fn=lambda t: [(0, min(120, len(t)))],
        embed_fn=lambda texts: ("E", texts), write_fn=lambda a, c: None,
        n_chunkers=2, row_group_rows=20, embed_batch=16, stall_timeout_s=30.0,
    )
    pipe.run(progress_every=50)                  # fires several times over 200 chunks
    out = capsys.readouterr().out
    assert "GPU busy" in out and "starved" in out
