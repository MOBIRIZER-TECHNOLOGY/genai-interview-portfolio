"""
The knowledge graph: build from triples, retrieve by entity walk.

    from graphrag.graph import KnowledgeGraph
    kg = KnowledgeGraph.load()
    result = kg.retrieve("what is the response time for the severity shed mode triggers?")

## How retrieval differs from vector search

Vector RAG asks "which chunks *resemble* this question?". Graph retrieval asks
"which entities does this question *mention*, and what do we know within a few
hops of them?" — a fundamentally different failure profile:

- multi-hop relational questions: the graph wins, because the join is stored
- fuzzy/paraphrase questions naming no entity: the graph has nothing to anchor
  on and vector similarity wins
- entity questions phrased with exotic vocabulary: the graph still wins, because
  entity matching is lexical, not semantic

## Deterministic entity linking, on purpose

Query entities are found by matching question n-grams against graph node names —
no LLM call. Three reasons: it is *testable* (an LLM linker makes every retrieval
test flaky), it is *fast* (~1 ms vs ~500 ms), and its failure mode is honest — a
miss returns nothing, rather than hallucinating a plausible entity. The cost is
recall on synonyms ("camera footage" will not match the node "vision frames"),
and the eval measures exactly that cost rather than hiding it.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "01_rag_local"))

from graphrag.extract import extract_graph, normalize_entity  # noqa: E402

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "what", "which", "who", "how", "when",
    "for", "of", "to", "in", "on", "at", "and", "or", "that", "this", "it",
    "do", "does", "should", "can", "i", "we", "long", "many", "much", "mean",
}


@dataclass
class GraphHit:
    """One retrieved fact, auditable back to its source chunk."""
    fact: str                    # "shed mode --triggers--> sev3"
    source: str                  # chunk id the triple was extracted from
    hops: int                    # distance from a query entity (1 = direct)


@dataclass
class GraphContext:
    """What graph retrieval hands to the generator."""
    query_entities: list[str]
    facts: list[GraphHit]
    chunks: list[dict] = field(default_factory=list)   # project-01-shaped blocks


class KnowledgeGraph:
    def __init__(self, data: dict):
        self.chunk_text: dict[str, str] = data["chunk_text"]
        self.g = nx.MultiDiGraph()
        for t in data["triples"]:
            # re-normalise at build time so an improved normaliser heals a
            # cached extraction instead of forcing a re-extract
            self.g.add_edge(normalize_entity(t["s"]), normalize_entity(t["o"]),
                            relation=t["r"], source=t["source"])
        self.n_triples = len(data["triples"])

    @classmethod
    def load(cls) -> "KnowledgeGraph":
        return cls(extract_graph())

    # ------------------------------------------------------ entity linking

    def link_entities(self, question: str, max_ngram: int = 4) -> list[str]:
        """Match question n-grams against node names. Longest match wins.

        Sorting candidates by length before matching means "manual inspection
        lane" is linked as one entity rather than shadowed by a shorter
        "inspection" node.
        """
        q = normalize_entity(question)
        words = [w for w in re.findall(r"[a-z0-9.-]+", q)]
        ngrams = set()
        for n in range(max_ngram, 0, -1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if n == 1 and (gram in STOPWORDS or len(gram) < 3):
                    continue
                ngrams.add(gram)

        hits = [node for node in self.g.nodes if node in ngrams]
        # prefer longer (more specific) entities, then drop ones contained in a
        # longer hit ("sev" when "sev2" matched)
        hits.sort(key=len, reverse=True)
        out: list[str] = []
        for h in hits:
            if not any(h in longer and h != longer for longer in out):
                out.append(h)
        return out

    # ---------------------------------------------------------- retrieval

    def neighborhood(self, entity: str, hops: int = 2, limit: int = 30) -> list[GraphHit]:
        """Every edge within `hops` of the entity, breadth-first, closest first."""
        if entity not in self.g:
            return []
        seen_edges: set[tuple] = set()
        out: list[GraphHit] = []
        frontier = {entity}
        visited = {entity}

        for depth in range(1, hops + 1):
            nxt: set[str] = set()
            for node in frontier:
                # both directions: what it points at, and what points at it
                for u, v, attrs in list(self.g.out_edges(node, data=True)) + \
                                   list(self.g.in_edges(node, data=True)):
                    key = (u, attrs["relation"], v)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    out.append(GraphHit(
                        fact=f"{u} --{attrs['relation']}--> {v}",
                        source=attrs["source"], hops=depth))
                    nxt.update((u, v))
            frontier = nxt - visited
            visited |= nxt
            if len(out) >= limit:
                break
        return out[:limit]

    def retrieve(self, question: str, hops: int = 2, max_facts: int = 24,
                 max_chunks: int = 4) -> GraphContext:
        """Entity-anchored retrieval: link -> walk -> collect facts + source chunks.

        The generator receives BOTH the walked facts (the relational skeleton)
        and the source chunks they came from (the prose with numbers and
        qualifiers) -- triples alone drop too much nuance to answer from.
        """
        entities = self.link_entities(question)
        facts: list[GraphHit] = []
        for e in entities:
            facts.extend(self.neighborhood(e, hops=hops))
        # closest hops first, stable across entities
        facts.sort(key=lambda h: h.hops)
        facts = facts[:max_facts]

        # source chunks, deduped, ordered by how many retrieved facts cite them
        counts: dict[str, int] = {}
        for h in facts:
            counts[h.source] = counts.get(h.source, 0) + 1
        chunk_ids = sorted(counts, key=counts.get, reverse=True)[:max_chunks]

        chunks = [{
            "chunk_id": cid,
            "source": cid.split("::")[0] + ".md",
            "breadcrumb": cid,
            "body": self.chunk_text[cid],
            "rerank_score": None,
        } for cid in chunk_ids if cid in self.chunk_text]

        return GraphContext(query_entities=entities, facts=facts, chunks=chunks)

    # -------------------------------------------------------------- stats

    def stats(self) -> dict:
        comps = list(nx.weakly_connected_components(self.g))
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "triples": self.n_triples,
            "components": len(comps),
            "largest_component": max(len(c) for c in comps) if comps else 0,
        }
