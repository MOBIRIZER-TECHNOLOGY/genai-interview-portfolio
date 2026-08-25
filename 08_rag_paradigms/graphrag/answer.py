"""
GraphRAG answering: walked facts + source chunks -> grounded, cited answer.

    python -m graphrag.answer "what is the response time for the severity shed mode triggers?"

Reuses project 01's generator wholesale — numbered blocks, mandatory [n]
citations, mechanical verification, NOT_FOUND abstention. The paradigms differ
in *what they retrieve*, not in generation discipline; holding the generator
constant is what makes the three-way comparison a retrieval experiment instead
of a prompt-engineering one.

The graph facts are prepended as block [1] (a synthetic "knowledge graph"
block), source chunks follow as [2..n]. The model may cite either; the facts
block carries the relational skeleton, the chunks carry the prose and numbers.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "01_rag_local"))
sys.path.insert(0, str(ROOT / "08_rag_paradigms"))

from rag.generate import DEFAULT_MODEL, answer as generate_answer  # noqa: E402

from graphrag.graph import GraphContext, KnowledgeGraph  # noqa: E402


@dataclass
class _Block:
    """Duck-typed to what rag.generate.build_context expects of a hit."""
    chunk_id: str
    source: str
    breadcrumb: str
    body: str
    rerank_score: float | None = None


def graph_context_to_blocks(ctx: GraphContext) -> list[_Block]:
    blocks: list[_Block] = []
    if ctx.facts:
        fact_lines = "\n".join(f"- {h.fact}   (from {h.source}, {h.hops} hop)"
                               for h in ctx.facts)
        blocks.append(_Block(
            chunk_id="graph::facts",
            source="knowledge-graph",
            breadcrumb=f"knowledge graph walk from: {', '.join(ctx.query_entities) or '(no entities linked)'}",
            body=fact_lines,
        ))
    for c in ctx.chunks:
        blocks.append(_Block(chunk_id=c["chunk_id"], source=c["source"],
                             breadcrumb=c["breadcrumb"], body=c["body"]))
    return blocks


@dataclass
class GraphAnswer:
    question: str
    entities: list[str]
    n_facts: int
    answer_text: str
    grounded: bool
    abstained: bool
    retrieve_ms: float
    generate_ms: float
    llm_calls: int = 1                     # generation only; linking is lexical


def ask_graph(question: str, kg: KnowledgeGraph | None = None,
              model: str = DEFAULT_MODEL) -> GraphAnswer:
    kg = kg or KnowledgeGraph.load()

    t0 = time.perf_counter()
    ctx = kg.retrieve(question)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    blocks = graph_context_to_blocks(ctx)
    if not blocks:
        # no entity linked and therefore nothing to walk: abstain honestly
        # rather than generating from an empty context
        return GraphAnswer(question, [], 0,
                           "NOT_FOUND: no known entity in the question matched the graph.",
                           grounded=True, abstained=True,
                           retrieve_ms=retrieve_ms, generate_ms=0.0, llm_calls=0)

    t0 = time.perf_counter()
    ans = generate_answer(question, blocks, model=model)
    generate_ms = (time.perf_counter() - t0) * 1000

    return GraphAnswer(question, ctx.query_entities, len(ctx.facts),
                       ans.text, ans.grounded, ans.abstained,
                       retrieve_ms, generate_ms)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?",
                    default="what is the response time for the severity shed mode triggers?")
    args = ap.parse_args()

    kg = KnowledgeGraph.load()
    print("graph:", kg.stats())
    r = ask_graph(args.question, kg)
    print(f"\nQ: {r.question}")
    print(f"entities linked: {r.entities}")
    print(f"facts walked   : {r.n_facts}")
    print(f"\n{r.answer_text}")
    print(f"\ngrounded={r.grounded} abstained={r.abstained} "
          f"| retrieve {r.retrieve_ms:.0f} ms, generate {r.generate_ms:.0f} ms")
