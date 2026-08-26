"""
Component tests for the project-01 RAG package — written to close the coverage
gap the first suite left.

The original suite was regression-targeted: every test pinned a bug that had
actually happened. Good tests, wrong lens for coverage — `store.py`,
`retrieve.py`, `embed.py`, `rerank.py` and `pipeline.py` sat between 0% and 42%
because no bug had happened there *yet*. That is exactly the argument for
closing the gap: the chunker was also 'fine' until it produced 42x too many
chunks.

Design choices that keep these honest and fast:

- **faiss-cpu is a real dependency, not a mock.** `VectorStore` tests exercise
  the actual index build/save/load/search path.
- **The embedder is faked with a deterministic hash-based encoder** where only
  vector *plumbing* is under test (retrieval fusion), and **real** where the
  embedding contract itself is under test (`test_embedder_*`, marked slow).
- **Ollama is monkeypatched at the httpx boundary** — the network call is the
  only thing faked; prompt construction, citation parsing and the abstain path
  all run for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_rag_local"))

from rag.chunking import Chunk, approx_tokens, chunk_corpus, chunk_markdown  # noqa: E402
from rag.generate import Answer, build_context, verify_citations  # noqa: E402
from rag.retrieve import HybridRetriever, tokenize  # noqa: E402
from rag.store import VectorStore  # noqa: E402

DIM = 32


# ------------------------------------------------------------ fake embedder


class FakeEmbedder:
    """Deterministic, vocabulary-based vectors — real cosine geometry, no GPU.

    Each token contributes a stable pseudo-random direction, so texts sharing
    words are genuinely similar. That makes retrieval *ordering* meaningful,
    which a pure-random fake would not.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.model_name = "fake-embedder"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            rng = np.random.default_rng(abs(hash(tok)) % (2**32))
            v += rng.normal(size=self.dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_passages(self, texts, **kw):
        return np.stack([self._vec(t) for t in texts])

    def encode_queries(self, texts, **kw):
        return np.stack([self._vec(t) for t in texts])


CORPUS = [
    ("d1", "The dispatch auction assigns tasks to robots every 150 milliseconds."),
    ("d2", "Error TLM-330 means a hypertable chunk write failed on tsdb-0."),
    ("d3", "Barcode reads below 0.92 confidence are retried with a new exposure."),
    ("d4", "Vision frames are blurred at the edge and kept for 14 days."),
    ("d5", "The starvation guard escalates a task after 12 lost auctions."),
]


@pytest.fixture()
def store():
    emb = FakeEmbedder()
    vecs = emb.encode_passages([t for _, t in CORPUS])
    records = [
        {"id": cid, "text": t, "body": t, "source": f"{cid}.md",
         "breadcrumb": f"{cid}.md > section", "n_tokens": approx_tokens(t), "meta": {}}
        for cid, t in CORPUS
    ]
    return VectorStore.build(vecs, records, "fake-embedder"), emb


# ------------------------------------------------------------------- store


def test_store_build_and_search(store):
    vs, emb = store
    hits = vs.search(emb.encode_queries(["hypertable chunk write failed"])[0], k=3)
    assert len(hits) == 3
    assert hits[0].chunk_id == "d2"
    assert hits[0].rank == 0
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_store_search_k_larger_than_corpus(store):
    vs, emb = store
    hits = vs.search(emb.encode_queries(["robots"])[0], k=50)
    assert len(hits) == len(CORPUS)          # clamped, not crashed


def test_store_save_load_roundtrip(tmp_path, store):
    vs, emb = store
    vs.save(tmp_path / "idx")
    # sidecar files all present
    for name in ("index.faiss", "chunks.jsonl", "meta.json"):
        assert (tmp_path / "idx" / name).exists()
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_name"] == "fake-embedder"
    assert meta["n_chunks"] == len(CORPUS)

    loaded = VectorStore.load(tmp_path / "idx")
    assert len(loaded) == len(vs)
    q = emb.encode_queries(["barcode confidence retry"])[0]
    assert [h.chunk_id for h in loaded.search(q, k=2)] == \
           [h.chunk_id for h in vs.search(q, k=2)], "search must survive persistence"


# ---------------------------------------------------------------- retrieve


def test_hybrid_search_fuses_both_arms(store):
    vs, emb = store
    r = HybridRetriever(vs, emb)
    hits = r.search("TLM-330 hypertable failure", k=3, candidates=5)
    assert hits[0].chunk_id == "d2"
    top = hits[0]
    assert top.dense_rank is not None and top.bm25_rank is not None, \
        "the right doc should be found by BOTH arms"
    assert hits[0].rrf_score >= hits[-1].rrf_score


def test_bm25_only_wins_on_rare_identifier(store):
    vs, emb = store
    r = HybridRetriever(vs, emb)
    hits = r.search_bm25_only("TLM-330", k=2)
    assert hits and hits[0].chunk_id == "d2"
    assert hits[0].dense_rank is None            # ablation arm is pure BM25


def test_dense_only_arm(store):
    vs, emb = store
    r = HybridRetriever(vs, emb)
    hits = r.search_dense_only("auction task assignment", k=3)
    assert len(hits) == 3
    assert all(h.bm25_rank is None for h in hits)


def test_rrf_depends_on_rank_not_score_magnitude(store):
    vs, emb = store
    r = HybridRetriever(vs, emb)
    hits = r.search("dispatch auction robots", k=5, candidates=5)
    # every fused score must be a sum of 1/(60+rank) terms -> bounded
    for h in hits:
        assert 0 < h.rrf_score <= 2 / 60


def test_bm25_returns_nothing_for_alien_vocabulary(store):
    vs, emb = store
    r = HybridRetriever(vs, emb)
    assert r.bm25("zzz qqq xyzzy", k=5) == []    # zero-score docs are dropped


# ---------------------------------------------------------------- chunking


def test_approx_tokens_floor():
    assert approx_tokens("") == 1                # never zero (used as divisor)
    assert approx_tokens("x" * 400) == 100


def test_chunk_markdown_splits_oversized_section(tmp_path):
    doc = tmp_path / "big.md"
    body = "word " * 900                          # ~1125 tokens >> 320 budget
    doc.write_text(f"# Title\n\n{body}", encoding="utf-8")
    chunks = chunk_markdown(doc, max_tokens=320, overlap_tokens=60)
    assert len(chunks) > 1, "oversized section must be split"
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.meta["of"] == len(chunks) for c in chunks)
    # overlap: consecutive pieces share text.
    # (This previously read `chunks[0].body[-40:] in chunks[0].body` -- a
    # tautology that passes for any input whatsoever, so the overlap property
    # was advertised as tested and was not.)
    assert len(chunks) >= 2
    tail = chunks[0].body[-40:]
    assert tail in chunks[1].body, "consecutive chunks must share the carried tail"


@pytest.mark.parametrize("body,label", [
    ("word " * 900, "one paragraph, spaces only -- no blank line to split on"),
    ("sentence. " * 400, "one paragraph of sentences"),
    ("x" * 6000, "no whitespace at all -- nothing to break on"),
    ("a, " * 1500, "commas only"),
])
def test_no_chunk_exceeds_the_embedding_window(tmp_path, body, label):
    """No chunk may exceed the budget -- the bug `_hard_split` exists to fix.

    A paragraph with no blank lines used to pass through `_split_long` whole:
    a 1,128-token chunk against a 320 budget, whose tail the 512-token embedding
    model then silently truncated. Never indexed, never retrievable, no error.

    `test_chunk_markdown_splits_oversized_section` asserts only `len(chunks) > 1`,
    which the *broken* code also satisfied once the section had several
    paragraphs. The property that actually pins the bug is this one: **every**
    chunk fits the window, including the pathological single-paragraph inputs
    above where there is no paragraph boundary to exploit.
    """
    doc = tmp_path / "big.md"
    doc.write_text(f"# Title\n\n{body}", encoding="utf-8")
    chunks = chunk_markdown(doc, max_tokens=320, overlap_tokens=60)

    assert chunks, f"{label}: produced no chunks at all"
    worst = max(approx_tokens(c.body) for c in chunks)
    # the carried overlap can push a chunk over the nominal budget, but never
    # past the embedder's 512-token window -- that is the line that matters
    assert worst <= 512, (
        f"{label}: largest chunk is {worst} tokens; anything past the 512-token "
        "embedding window is silently truncated and unretrievable"
    )


def test_chunk_markdown_nested_headings(tmp_path):
    doc = tmp_path / "nested.md"
    doc.write_text("# A\n\nintro\n\n## B\n\ndetail under B\n\n### C\n\ndeep text\n",
                   encoding="utf-8")
    chunks = chunk_markdown(doc)
    crumbs = [c.breadcrumb for c in chunks]
    assert any("A > B > C" in b for b in crumbs), f"breadcrumbs wrong: {crumbs}"


def test_chunk_corpus_walks_directory(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.md").write_text(f"# Doc {i}\n\ncontent {i} here padded out "
                                          "with enough words to pass the length gate.",
                                          encoding="utf-8")
    chunks = chunk_corpus(tmp_path)
    assert len(chunks) == 3
    assert sorted({c.source for c in chunks}) == ["0.md", "1.md", "2.md"]
    d = chunks[0].to_dict()
    assert {"id", "text", "body", "source", "breadcrumb"} <= set(d)


# ---------------------------------------------------------------- generate


def test_build_context_numbers_blocks(store):
    vs, emb = store
    r = HybridRetriever(vs, emb)
    hits = r.search("barcode", k=3, candidates=5)
    ctx, manifest = build_context(hits)
    assert "[1] source:" in ctx and "[3] source:" in ctx
    assert [m["n"] for m in manifest] == [1, 2, 3]
    assert manifest[0]["chunk_id"] == hits[0].chunk_id


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_answer_pipeline_with_mocked_ollama(monkeypatch, store):
    """Full answer() path: prompt build, citation parse, grounding verdict.

    Only the network hop is faked -- at the httpx boundary, so the request
    payload the code constructs is still exercised.
    """
    import rag.generate as G

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse({
            "message": {"content": "The threshold is 0.92 [1]."},
            "prompt_eval_count": 100, "eval_count": 12, "total_duration": 5e8,
        })

    monkeypatch.setattr(G.httpx, "post", fake_post)

    vs, emb = store
    r = HybridRetriever(vs, emb)
    hits = r.search("barcode confidence", k=2, candidates=5)
    ans = G.answer("What threshold?", hits, model="test-model")

    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["options"]["temperature"] == 0.0
    assert ans.valid_citations == [1]
    assert ans.grounded and not ans.abstained
    assert ans.prompt_tokens == 100 and ans.eval_tokens == 12
    assert ans.seconds == pytest.approx(0.5)


def test_answer_abstain_path(monkeypatch, store):
    import rag.generate as G

    monkeypatch.setattr(G.httpx, "post", lambda *a, **k: _FakeResponse(
        {"message": {"content": "NOT_FOUND: salary data is not in the corpus."}}))
    vs, emb = store
    hits = HybridRetriever(vs, emb).search("salary", k=2, candidates=5)
    ans = G.answer("What is the salary band?", hits)
    assert ans.abstained and ans.grounded
    assert ans.citations == []


def test_answer_hallucinated_citation_is_ungrounded(monkeypatch, store):
    import rag.generate as G

    monkeypatch.setattr(G.httpx, "post", lambda *a, **k: _FakeResponse(
        {"message": {"content": "A fact from [9]."}}))
    vs, emb = store
    hits = HybridRetriever(vs, emb).search("barcode", k=2, candidates=5)
    ans = G.answer("q", hits)
    assert ans.invalid_citations == [9]
    assert not ans.grounded


def test_ollama_stream_parses_ndjson(monkeypatch):
    import rag.generate as G

    class _FakeStreamResp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({"message": {"content": "Hel"}, "done": False})
            yield ""                                          # keepalive
            yield json.dumps({"message": {"content": "lo"}, "done": False})
            yield json.dumps({"done": True})

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(G.httpx, "stream", lambda *a, **k: _FakeStreamResp())
    toks = list(G.ollama_stream([{"role": "user", "content": "hi"}]))
    assert toks == ["Hel", "lo"]


# --------------------------------------------- real-model contracts (slow)
# These exercise embed.py / rerank.py / pipeline.py against the actual cached
# models. Marked slow; they are the contract tests a fake cannot provide.


@pytest.mark.slow
def test_embedder_asymmetric_prefix_changes_vector():
    """BGE is asymmetric: the same string must embed differently as a query.

    Forgetting the instruction prefix costs recall with nothing in the logs --
    this pins that the code path actually applies it.
    """
    from rag.embed import Embedder

    e = Embedder()
    text = "how long are vision frames retained"
    p = e.encode_passages([text], show=False)[0]
    q = e.encode_queries([text])[0]
    assert p.shape == (e.dim,)
    assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-3), "must be L2-normalised"
    assert float(p @ q) < 0.999, "query prefix must change the embedding"


@pytest.mark.slow
def test_reranker_orders_by_relevance(store):
    from rag.rerank import Reranker

    vs, emb = store
    hits = HybridRetriever(vs, emb).search("barcode confidence threshold",
                                          k=5, candidates=5)
    rr = Reranker()
    top = rr.rerank("what barcode confidence triggers a retry", hits, top_k=2)
    assert len(top) == 2
    assert top[0].rerank_score >= top[1].rerank_score
    assert top[0].chunk_id == "d3"
    assert rr.rerank("q", []) == []              # empty input contract


@pytest.mark.slow
def test_rag_pipeline_load_and_retrieve():
    """End-to-end load of the real project-01 index and all retrieval modes."""
    from rag.pipeline import RagPipeline

    index = ROOT / "01_rag_local" / "index"
    if not index.exists():
        pytest.skip("project 01 index not built")

    pipe = RagPipeline.load(index, use_reranker=True)
    for mode in ("hybrid", "dense", "bm25"):
        hits, r_ms, rr_ms = pipe.retrieve("what is the Rotterdam rule?", mode=mode)
        assert hits, f"no hits in mode {mode}"
        assert len(hits) <= pipe.top_k
        assert r_ms >= 0 and rr_ms >= 0
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        pipe.retrieve("q", mode="quantum")


# --------------------------------------------------------------- project 04 data


def test_voice_eval_leakage_is_measured_not_assumed():
    """The voice corpus has train/eval text overlap -- it must stay *quantified*.

    24 of 60 held-out sentences appear verbatim in training, and every eval speaker
    is also a training speaker. That was measured (the unleaked subset actually
    scores *better*: 2.4% WER vs 2.6%, 98.5% domain-term recall vs 91.4%), so the
    headline is not inflated by it.

    This test does not forbid the overlap -- it pins the fact that it exists and is
    known, so a future regeneration cannot quietly change the amount and leave the
    README's audit describing a corpus that no longer exists. If it fires, re-run
    the leaked/unleaked split before touching the number.
    """
    data = ROOT / "04_lora_voice" / "data"
    if not (data / "train.jsonl").exists():
        pytest.skip("voice corpus not generated (make_dataset.py)")

    def texts(name):
        return [json.loads(l)["text"]
                for l in (data / name).read_text(encoding="utf-8").splitlines() if l.strip()]

    tr, ev = texts("train.jsonl"), texts("eval.jsonl")
    overlap = sum(1 for t in ev if t in set(tr))
    share = overlap / len(ev)
    assert 0.30 <= share <= 0.50, (
        f"{overlap}/{len(ev)} eval sentences ({share:.0%}) also appear in training. "
        "The README documents ~40% and reports the leaked/unleaked split measured at "
        "that level. A different figure means the audit needs re-running, not editing."
    )
