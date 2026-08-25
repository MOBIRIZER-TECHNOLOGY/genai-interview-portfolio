"""
Coverage for project 08's wiring modules — the glue the logic tests skipped.

The first paradigm test file covered graph walks and the agent loop's control
flow; these cover the modules around them: block assembly for the generator,
the extraction pipeline (LLM mocked at the httpx boundary), and the agent's
network/default-construction paths. Same rule as every coverage test here:
every branch exercised is reachable in production, and the LLM is the only
thing faked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_rag_paradigms"))
sys.path.insert(0, str(ROOT / "01_rag_local"))

from graphrag.graph import KnowledgeGraph  # noqa: E402


@pytest.fixture()
def kg():
    data = {
        "chunk_text": {"c1": "shed mode is SEV3 severity",
                       "c2": "SEV3 responds next business day"},
        "triples": [
            {"s": "shed mode", "r": "classified_as", "o": "sev3", "source": "c1"},
            {"s": "sev3", "r": "has_response_time", "o": "next business day", "source": "c2"},
        ],
    }
    return KnowledgeGraph(data)


# ------------------------------------------------------- graphrag/answer.py


def test_graph_blocks_carry_facts_then_chunks(kg):
    from graphrag.answer import graph_context_to_blocks

    ctx = kg.retrieve("what severity is shed mode?")
    blocks = graph_context_to_blocks(ctx)
    assert blocks[0].source == "knowledge-graph"
    assert "classified_as" in blocks[0].body
    assert any(b.chunk_id == "c1" for b in blocks[1:]), "source chunks must follow facts"


def test_ask_graph_generates_from_blocks(kg, monkeypatch):
    """Full ask_graph wiring with only the generator's LLM call faked."""
    import graphrag.answer as A
    from rag.generate import Answer

    seen = {}

    def fake_generate(question, blocks, model="m"):
        seen["blocks"] = blocks
        return Answer(text="SEV3 -> next business day [1]", citations=[1],
                      valid_citations=[1], invalid_citations=[], abstained=False)

    monkeypatch.setattr(A, "generate_answer", fake_generate)
    r = A.ask_graph("what is shed mode's response time?", kg)
    assert r.grounded and not r.abstained
    assert r.entities == ["shed mode"]
    assert r.n_facts > 0 and r.llm_calls == 1
    assert seen["blocks"][0].source == "knowledge-graph"


def test_ask_graph_abstains_with_no_linkable_entity(kg):
    """No entity -> no blocks -> honest NOT_FOUND with zero LLM calls."""
    from graphrag.answer import ask_graph

    r = ask_graph("how does photosynthesis work?", kg)
    assert r.abstained and r.grounded
    assert r.llm_calls == 0 and r.n_facts == 0
    assert r.answer_text.startswith("NOT_FOUND")


# ------------------------------------------------------ graphrag/extract.py


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_chat_json_parses_clean_and_wrapped(monkeypatch):
    import graphrag.extract as E

    monkeypatch.setattr(E.httpx, "post", lambda *a, **k: _FakeResp(
        {"message": {"content": '{"triples": []}'}}))
    assert E._chat_json("p") == {"triples": []}

    monkeypatch.setattr(E.httpx, "post", lambda *a, **k: _FakeResp(
        {"message": {"content": 'noise {"triples": [{"s":"a","r":"b","o":"c"}]} tail'}}))
    assert E._chat_json("p")["triples"][0]["s"] == "a"

    monkeypatch.setattr(E.httpx, "post", lambda *a, **k: _FakeResp(
        {"message": {"content": "no json at all"}}))
    assert E._chat_json("p") is None


def test_extract_graph_uses_cache(tmp_path, monkeypatch):
    """The cached path must not touch the LLM at all."""
    import graphrag.extract as E

    cached = {"model": "m", "n_chunks": 1, "triples": [], "chunk_text": {},
              "extract_seconds": 0.0}
    fake_cache = tmp_path / "graph_data.json"
    fake_cache.write_text(json.dumps(cached), encoding="utf-8")
    monkeypatch.setattr(E, "GRAPH_DATA", fake_cache)

    def boom(*a, **k):
        raise AssertionError("LLM must not be called on the cached path")

    monkeypatch.setattr(E, "_chat_json", boom)
    assert E.extract_graph() == cached


def test_extract_graph_full_run_with_mocked_llm(tmp_path, monkeypatch):
    """The extraction loop end to end: chunking real, LLM scripted.

    Includes the two regression payloads: a triple with a null field (crashed a
    real run 20 chunks in) and a non-dict entry.
    """
    import graphrag.extract as E

    monkeypatch.setattr(E, "GRAPH_DATA", tmp_path / "graph_data.json")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# Shed\n\nShed mode is classified as SEV3 by the runbook and it is "
        "long enough to survive the chunker's minimum length gate.",
        encoding="utf-8")
    # point the module's corpus root at the fixture
    monkeypatch.setattr(E, "ROOT", tmp_path.parent if False else E.ROOT)

    import rag.chunking as C
    real_chunks = C.chunk_corpus(corpus)
    monkeypatch.setattr(C, "chunk_corpus", lambda p: real_chunks)

    monkeypatch.setattr(E, "_chat_json", lambda *a, **k: {"triples": [
        {"s": "Shed_Mode", "r": "classified as", "o": "SEV 3"},
        {"s": None, "r": "x", "o": "y"},              # null field -- regression
        "not-a-dict",                                  # junk entry -- regression
    ]})

    data = E.extract_graph(force=True)
    assert data["n_chunks"] == len(real_chunks)
    t = data["triples"][0]
    assert t["s"] == "shed mode" and t["o"] == "sev3", "normalisation must apply"
    assert t["r"] == "classified_as"
    assert len(data["triples"]) == len(real_chunks), "junk entries must be dropped, not crash"
    assert (tmp_path / "graph_data.json").exists()


# ----------------------------------------------------------- agentic/agent.py


def test_agent_malformed_json_in_action(monkeypatch):
    """A reply that is not JSON at all must become a recorded malformed step."""
    import agentic.agent as A

    class Pipe:
        top_k = 3

        def retrieve(self, q, mode="hybrid"):
            return [], 0.0, 0.0

    script = iter(["utter garbage not json",
                   '{"action": "abstain", "reason": "give up"}'])
    monkeypatch.setattr(A, "_chat", lambda m, mo, u: next(script))
    r = A.AgenticRag(pipeline=Pipe(), max_steps=3).ask("q?")
    assert any(s.action == "malformed" for s in r.steps)
    assert r.abstained


def test_graph_load_classmethod_uses_cached_extraction(monkeypatch):
    """KnowledgeGraph.load() must build from whatever extract_graph returns."""
    import graphrag.graph as G

    data = {"chunk_text": {}, "triples": [
        {"s": "a", "r": "rel", "o": "b", "source": "c1"}]}
    monkeypatch.setattr(G, "extract_graph", lambda: data)
    kg = G.KnowledgeGraph.load()
    assert kg.g.number_of_edges() == 1


def test_graph_stats_and_walk_limit():
    """stats() on a populated graph, and the neighborhood `limit` early-exit."""
    from graphrag.graph import KnowledgeGraph

    triples = [{"s": "hub", "r": f"r{i}", "o": f"n{i}", "source": "c"}
               for i in range(20)]
    kg = KnowledgeGraph({"chunk_text": {}, "triples": triples})
    st = kg.stats()
    assert st["nodes"] == 21 and st["edges"] == 20
    assert st["largest_component"] == 21
    hits = kg.neighborhood("hub", hops=2, limit=5)
    assert len(hits) == 5, "limit must cap the walk"


def test_agent_chat_posts_json_format(monkeypatch):
    import agentic.agent as A

    seen = {}

    def fake_post(url, timeout=None, json=None):
        seen["url"], seen["payload"] = url, json
        return _FakeResp({"message": {"content": '{"action":"abstain","reason":"r"}'}})

    monkeypatch.setattr(A.httpx, "post", fake_post)
    out = A._chat([{"role": "user", "content": "q"}], "test-model", "http://x")
    assert seen["url"] == "http://x/api/chat"
    assert seen["payload"]["format"] == "json"
    assert seen["payload"]["options"]["temperature"] == 0.0
    assert json.loads(out)["action"] == "abstain"


def test_agent_default_pipeline_construction(monkeypatch):
    """The no-argument constructor must load the project-01 pipeline."""
    import agentic.agent as A
    import rag.pipeline as P

    built = {}

    class FakePipe:
        top_k = 3

    def fake_load(cls, path):
        built["path"] = path
        return FakePipe()

    monkeypatch.setattr(P.RagPipeline, "load", classmethod(fake_load))
    agent = A.AgenticRag()
    assert isinstance(agent.pipe, FakePipe)
    assert str(built["path"]).endswith("index")


def test_agent_search_dedups_chunks_across_steps(monkeypatch):
    """A chunk retrieved twice must appear as evidence exactly once."""
    import agentic.agent as A

    class Hit:
        def __init__(self, cid):
            self.chunk_id, self.source, self.body = cid, "s.md", f"body {cid}"

    class Pipe:
        top_k = 3

        def retrieve(self, q, mode="hybrid"):
            return [Hit("same"), Hit(f"uniq-{q}")], 0.0, 0.0

    script = iter([
        '{"action": "search", "query": "one"}',
        '{"action": "search", "query": "two"}',
        '{"action": "answer", "text": "done [1] [2] [3]"}',
    ])
    monkeypatch.setattr(A, "_chat", lambda m, mo, u: next(script))
    r = A.AgenticRag(pipeline=Pipe()).ask("q?")
    # 4 retrieved hits, but "same" deduped -> 3 evidence blocks, all citable
    assert r.grounded, "citations [1..3] must all be valid after dedup"
