"""
Agentic RAG: the model decides what to search, reads the results, and searches
again — a retrieve-reason-retrieve loop instead of one shot.

    python -m agentic.agent "what action fixes the incident type that causes the most pages?"

## What single-shot RAG cannot do, structurally

One-shot retrieval embeds the *question*. For a compositional question the
answer's location depends on an intermediate fact the question never states:

    "What action fixes the incident type that causes the most pages?"
      hop 1: most pages -> TLM-101 clock skew storms (41%)     [runbook]
      hop 2: TLM-101 -> restart ntp-relay                      [runbook table]

The right second search ("TLM-101 fix") can only be *formulated after reading*
the first result. No amount of better embedding fixes an unknown query.

## Design: a typed action loop, not a framework

Each turn the model sees the question plus all evidence gathered so far, and
must emit exactly one JSON action:

    {"action": "search", "query": "..."}     -> hybrid+rerank over project 01
    {"action": "answer", "text": "... [n]"}  -> final, with block citations
    {"action": "abstain", "reason": "..."}   -> corpus does not contain it

Deliberate constraints, each of which is a lesson from this repo:

- **Bounded steps** (default 4). An agent that can loop is an agent that can
  loop forever; the budget converts "stuck" into "abstain with a trajectory".
- **Evidence blocks are numbered across the whole trajectory** and the final
  answer's citations are verified with project 01's *mechanical* checker.
  Agency changes retrieval, not grounding discipline.
- **Repeated-query detection**: issuing a search you already ran burns a step
  and gains nothing; the loop injects a warning rather than silently spinning.
- **The trajectory is a first-class artifact.** Every eval row records what the
  agent searched and why it stopped — an agent you cannot audit is an agent you
  cannot debug.

## The honest cost

Every step is a full LLM call. A 2-hop answer costs ~3 generations plus 2
retrievals against ~1 generation for single-shot. The eval measures whether the
extra hops actually buy accuracy on multi-hop questions — and whether they
*waste* latency on the simple ones, which is the half people forget to check.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "01_rag_local"))

from rag.generate import verify_citations  # noqa: E402

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"

SYSTEM = """You answer questions about the Atlas platform using a search tool.

Each turn, reply with EXACTLY ONE JSON object, nothing else:
  {"action": "search", "query": "<short keyword query>"}
  {"action": "answer", "text": "<answer citing evidence blocks like [2]>"}
  {"action": "abstain", "reason": "<why the corpus cannot answer this>"}

Strategy:
- Break multi-part questions into hops: search for the intermediate fact first,
  READ the evidence, then search for what it points to.
- Prefer short keyword queries over full sentences.
- Answer ONLY from the numbered evidence blocks, citing them like [2].
- Before answering, re-read the question: your answer must address the FULL
  question asked, not an intermediate fact you found along the way. If the
  question asks for an action/value about X and you have only identified X,
  search for X's action/value first.
- If evidence is missing after searching, abstain -- never guess.
- Do not repeat a search you have already run."""


@dataclass
class Step:
    action: str
    detail: str                    # the query, or the answer/abstain text
    n_new_blocks: int = 0


@dataclass
class AgentResult:
    question: str
    answer: str
    abstained: bool
    grounded: bool
    steps: list[Step] = field(default_factory=list)
    llm_calls: int = 0
    searches: int = 0
    total_ms: float = 0.0
    stop_reason: str = ""

    def trajectory(self) -> str:
        lines = [f"Q: {self.question}"]
        for i, s in enumerate(self.steps, 1):
            extra = f"  (+{s.n_new_blocks} blocks)" if s.action == "search" else ""
            lines.append(f"  {i}. {s.action}: {s.detail[:90]}{extra}")
        lines.append(f"  -> {self.stop_reason}")
        return "\n".join(lines)


def _chat(messages: list[dict], model: str, url: str) -> str:
    r = httpx.post(f"{url}/api/chat", timeout=180, json={
        "model": model, "messages": messages, "stream": False,
        "format": "json", "options": {"temperature": 0.0},
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


def _parse_action(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"action": "malformed", "raw": text[:200]}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"action": "malformed", "raw": text[:200]}
    if not isinstance(obj, dict) or obj.get("action") not in ("search", "answer", "abstain"):
        return {"action": "malformed", "raw": str(obj)[:200]}
    return obj


class AgenticRag:
    def __init__(self, pipeline=None, model: str = MODEL, url: str = OLLAMA_URL,
                 max_steps: int = 4, per_search_k: int = 3):
        if pipeline is None:
            from rag.pipeline import RagPipeline

            pipeline = RagPipeline.load(ROOT / "01_rag_local" / "index")
        self.pipe = pipeline
        self.model = model
        self.url = url
        self.max_steps = max_steps
        self.per_search_k = per_search_k

    # ------------------------------------------------------------- search

    def _search(self, query: str) -> list:
        self.pipe.top_k = self.per_search_k
        hits, _, _ = self.pipe.retrieve(query, mode="hybrid")
        return hits

    # ---------------------------------------------------------------- run

    def ask(self, question: str) -> AgentResult:
        t_start = time.perf_counter()
        result = AgentResult(question=question, answer="", abstained=False,
                             grounded=False)
        evidence: list = []                       # accumulated blocks, stable numbering
        seen_chunks: set[str] = set()
        seen_queries: set[str] = set()

        def evidence_text() -> str:
            if not evidence:
                return "(no evidence gathered yet -- you must search first)"
            return "\n\n".join(
                f"[{i}] ({h.source}) {h.body}" for i, h in enumerate(evidence, 1))

        for step_no in range(1, self.max_steps + 1):
            budget = self.max_steps - step_no
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                    f"Question: {question}\n\n"
                    f"Evidence blocks so far:\n{evidence_text()}\n\n"
                    f"Actions remaining after this one: {budget}. "
                    + ("You MUST answer or abstain now." if budget == 0 else "")},
            ]
            raw = _chat(messages, self.model, self.url)
            result.llm_calls += 1
            act = _parse_action(raw)

            if act["action"] == "search":
                q = str(act.get("query", "")).strip()
                norm = q.lower()
                if norm in seen_queries:
                    # burning steps on a repeat is the classic small-model loop;
                    # surface it in the transcript instead of silently spinning
                    result.steps.append(Step("search(repeat-ignored)", q))
                    continue
                seen_queries.add(norm)
                hits = self._search(q)
                new = [h for h in hits if h.chunk_id not in seen_chunks]
                for h in new:
                    seen_chunks.add(h.chunk_id)
                    evidence.append(h)
                result.searches += 1
                result.steps.append(Step("search", q, n_new_blocks=len(new)))
                continue

            if act["action"] == "answer":
                text = str(act.get("text", "")).strip()
                cited, valid, invalid = verify_citations(text, len(evidence))
                result.answer = text
                result.grounded = bool(cited) and not invalid
                result.steps.append(Step("answer", text))
                result.stop_reason = "answered" + ("" if result.grounded
                                                  else " (UNGROUNDED)")
                break

            if act["action"] == "abstain":
                reason = str(act.get("reason", "")).strip()
                result.answer = f"NOT_FOUND: {reason}"
                result.abstained = True
                result.grounded = True
                result.steps.append(Step("abstain", reason))
                result.stop_reason = "abstained"
                break

            result.steps.append(Step("malformed", act.get("raw", "")))
        else:
            result.answer = "NOT_FOUND: step budget exhausted before an answer."
            result.abstained = True
            result.grounded = True
            result.stop_reason = f"budget exhausted ({self.max_steps} steps)"

        result.total_ms = (time.perf_counter() - t_start) * 1000
        return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?",
                    default="what action fixes the incident type that causes the most pages?")
    ap.add_argument("--max-steps", type=int, default=4)
    args = ap.parse_args()

    agent = AgenticRag(max_steps=args.max_steps)
    r = agent.ask(args.question)
    print(r.trajectory())
    print(f"\nANSWER: {r.answer}")
    print(f"grounded={r.grounded} | {r.llm_calls} LLM calls, {r.searches} searches, "
          f"{r.total_ms:.0f} ms")
