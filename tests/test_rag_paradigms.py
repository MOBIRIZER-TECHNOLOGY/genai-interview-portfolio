"""
Deterministic tests for project 08 — GraphRAG and the agentic loop.

The LLM-dependent halves (extraction quality, agent reasoning) are measured by
`evaluate_paradigms.py`; what belongs HERE is everything that must never flake:
entity normalisation, graph walks, action parsing, budget enforcement, and
grounding verification. Both real bugs found while building project 08 are
pinned:

- the extractor oscillated between "shed mode" and "shed_mode", producing two
  disconnected nodes -- the crucial classified_as edge existed but was
  unreachable, and retrieval returned NOT_FOUND with no error
- the extractor emitted null fields inside otherwise-valid triples, crashing
  the run 20 chunks in
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_rag_paradigms"))
sys.path.insert(0, str(ROOT / "01_rag_local"))

from graphrag.extract import normalize_entity  # noqa: E402
from graphrag.graph import KnowledgeGraph  # noqa: E402

from agentic.agent import AgenticRag, _parse_action  # noqa: E402


# ------------------------------------------------------ entity resolution


def test_underscore_and_space_variants_unify():
    """THE GraphRAG regression: 'shed_mode' vs 'shed mode' were two nodes.

    The classified_as edge existed in the data but was unreachable from the
    linked entity -- a silent retrieval failure with a correct-looking graph.
    """
    assert normalize_entity("shed_mode") == normalize_entity("shed mode")
    assert normalize_entity("SHED  MODE") == "shed mode"


def test_possessive_links_to_the_base_entity():
    """The second normalisation regression: "shed mode's" must not become "shed modes".

    Order matters and that is the whole bug. Stripping quote characters first
    turns "shed mode's severity" into "shed modes severity" -- a token matching
    no node, so the hop silently returns nothing. The possessive rule therefore
    runs BEFORE quote-stripping in `normalize_entity`, and this pins that order.

    Both `extract.py` and project 08's README claimed this was "found by test,
    also pinned" while no such test existed -- the code fix was real, the
    pinning was not. Written when a documentation audit caught the gap.
    """
    assert normalize_entity("shed mode's") == normalize_entity("shed mode")
    assert normalize_entity("SEV3's") == "sev3"
    # the trailing-possessive rule must not eat a legitimate terminal s
    assert normalize_entity("vision frames") == "vision frames"


@pytest.mark.parametrize("raw,expected", [
    ("TLM 330", "tlm-330"),           # identifier folding
    ("`sev2`", "sev2"),               # markdown backticks stripped
    ("Sev 3", "sev3"),                # spaced severity folded
    ("Atlas-Dispatch", "atlas-dispatch"),
    ("shed mode's", "shed mode"),     # possessive stripped, base entity intact
])
def test_normalize_entity_canonical_forms(raw, expected):
    assert normalize_entity(raw) == expected


# ------------------------------------------------------------ graph walks


@pytest.fixture()
def kg():
    """A tiny graph with a known two-hop chain -- no LLM, no cache file."""
    data = {
        "chunk_text": {"c1": "shed mode is SEV3", "c2": "SEV3 responds next business day",
                       "c3": "tlm-101 is fixed by restarting ntp-relay"},
        "triples": [
            {"s": "shed_mode", "r": "classified_as", "o": "sev3", "source": "c1"},
            {"s": "sev3", "r": "has_response_time", "o": "next business day", "source": "c2"},
            {"s": "tlm-101", "r": "fixed_by", "o": "restart ntp-relay", "source": "c3"},
        ],
    }
    return KnowledgeGraph(data)


def test_build_renormalises_cached_triples(kg):
    """An improved normaliser must heal a cached extraction at build time."""
    assert "shed mode" in kg.g
    assert "shed_mode" not in kg.g


def test_two_hop_walk_reaches_the_join(kg):
    hits = kg.neighborhood("shed mode", hops=2)
    facts = [h.fact for h in hits]
    assert any("classified_as" in f for f in facts), "hop 1 missing"
    assert any("next business day" in f for f in facts), "hop 2 missing -- the join failed"
    # hop ordering: direct edges before second-hop edges
    assert hits[0].hops <= hits[-1].hops


def test_walk_respects_hop_limit(kg):
    one_hop = kg.neighborhood("shed mode", hops=1)
    assert all(h.hops == 1 for h in one_hop)
    assert not any("next business day" in h.fact for h in one_hop)


def test_unknown_entity_walks_nowhere(kg):
    assert kg.neighborhood("nonexistent", hops=2) == []


def test_entity_linking_longest_match_wins(kg):
    ents = kg.link_entities("what severity is shed mode classified as?")
    assert "shed mode" in ents
    # a fragment of a longer matched entity must not appear separately
    assert "shed" not in ents


def test_retrieve_collects_facts_and_source_chunks(kg):
    ctx = kg.retrieve("what is the response time for shed mode's severity?")
    assert ctx.query_entities
    assert ctx.facts
    ids = {c["chunk_id"] for c in ctx.chunks}
    assert "c1" in ids, "source chunk of the hop-1 fact must be returned"


def test_retrieve_with_no_linkable_entity(kg):
    ctx = kg.retrieve("how does photosynthesis work?")
    assert ctx.query_entities == [] and ctx.facts == [] and ctx.chunks == []


# ------------------------------------------------------ agent action parse


@pytest.mark.parametrize("raw,expected_action", [
    ('{"action": "search", "query": "tlm-101"}', "search"),
    ('{"action": "answer", "text": "x [1]"}', "answer"),
    ('{"action": "abstain", "reason": "not in corpus"}', "abstain"),
    ('Sure! Here you go: {"action": "search", "query": "q"}', "search"),  # prose wrapper
    ("total garbage", "malformed"),
    ('{"action": "delete_everything"}', "malformed"),                     # unknown verb
    ('["not", "a", "dict"]', "malformed"),
])
def test_parse_action(raw, expected_action):
    assert _parse_action(raw)["action"] == expected_action


# ---------------------------------------------------------- agent loop
# The LLM is faked with a scripted sequence, so the loop's control flow --
# budget, dedup, grounding, abstention -- is tested deterministically.


class _FakeHit:
    def __init__(self, cid, body):
        self.chunk_id, self.source, self.body = cid, f"{cid}.md", body


class _FakePipeline:
    top_k = 3

    def retrieve(self, query, mode="hybrid"):
        return [_FakeHit(f"h::{query[:12]}", f"evidence for {query}")], 0.0, 0.0


def make_agent(script: list[str], **kw) -> AgenticRag:
    agent = AgenticRag(pipeline=_FakePipeline(), **kw)
    it = iter(script)

    import agentic.agent as A

    # patch the module-level chat used inside ask()
    agent._script = it
    original = A._chat
    A._chat = lambda messages, model, url: next(it)
    agent._restore = lambda: setattr(A, "_chat", original)
    return agent


def test_agent_search_then_answer():
    agent = make_agent([
        '{"action": "search", "query": "most pages"}',
        '{"action": "answer", "text": "TLM-101, fixed by ntp-relay [1]"}',
    ])
    try:
        r = agent.ask("what fixes the incident with most pages?")
    finally:
        agent._restore()
    assert r.answer.startswith("TLM-101")
    assert r.grounded and not r.abstained
    assert r.searches == 1 and r.llm_calls == 2
    assert "answered" in r.stop_reason


def test_agent_hallucinated_citation_marked_ungrounded():
    agent = make_agent([
        '{"action": "search", "query": "q"}',
        '{"action": "answer", "text": "a fact from [9]"}',   # only 1 block exists
    ])
    try:
        r = agent.ask("q?")
    finally:
        agent._restore()
    assert not r.grounded
    assert "UNGROUNDED" in r.stop_reason


def test_agent_budget_exhaustion_abstains():
    """An agent that only ever searches must land on NOT_FOUND, not hang."""
    agent = make_agent([f'{{"action": "search", "query": "q{i}"}}' for i in range(9)],
                       max_steps=3)
    try:
        r = agent.ask("q?")
    finally:
        agent._restore()
    assert r.abstained
    assert "budget exhausted" in r.stop_reason
    assert r.llm_calls == 3


def test_agent_repeated_query_burns_step_visibly():
    agent = make_agent([
        '{"action": "search", "query": "same thing"}',
        '{"action": "search", "query": "SAME THING"}',       # case-folded repeat
        '{"action": "answer", "text": "done [1]"}',
    ])
    try:
        r = agent.ask("q?")
    finally:
        agent._restore()
    assert r.searches == 1, "the repeat must not re-run retrieval"
    assert any("repeat-ignored" in s.action for s in r.steps)
    assert r.grounded


def test_agent_abstain_path():
    agent = make_agent(['{"action": "abstain", "reason": "no such data"}'])
    try:
        r = agent.ask("what is the salary band?")
    finally:
        agent._restore()
    assert r.abstained and r.grounded
    assert r.answer.startswith("NOT_FOUND")


def test_agent_evidence_numbering_is_stable_across_searches():
    """Block numbers must never shift once assigned, or citations rot."""
    agent = make_agent([
        '{"action": "search", "query": "first"}',
        '{"action": "search", "query": "second"}',
        '{"action": "answer", "text": "uses [1] and [2]"}',
    ])
    try:
        r = agent.ask("q?")
    finally:
        agent._restore()
    assert r.grounded
    assert r.searches == 2


def test_trajectory_is_auditable():
    agent = make_agent([
        '{"action": "search", "query": "alpha"}',
        '{"action": "answer", "text": "done [1]"}',
    ])
    try:
        r = agent.ask("q?")
    finally:
        agent._restore()
    tj = r.trajectory()
    assert "search: alpha" in tj and "answer:" in tj and "->" in tj
