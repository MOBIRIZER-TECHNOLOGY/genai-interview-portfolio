"""
The three-way experiment: vector RAG vs GraphRAG vs Agentic RAG, same corpus,
same questions, same generator discipline.

    python evaluate_paradigms.py
    python evaluate_paradigms.py --subset multihop     # just the hard ones

## Why this file is the point of project 08

Anyone can implement three retrieval paradigms. The question that matters is
**when does each win**, and that is only answerable with all three running over
the *same* corpus against the *same* labelled questions. Two question sets:

- **standard** — project 01's 17 answerable + 3 unanswerable. Mostly single-hop
  lookups; the home turf of vector RAG.
- **multihop** — 7 hand-written compositional questions whose answer requires
  joining two facts from different chunks. The home turf of graphs and agents.
  Each was verified by hand against the corpus, with the hop chain documented.

Scoring is project 01's: `must_contain` strings for fact recall (deterministic,
no judge), plus abstention correctness, latency, and **LLM calls** — because the
paradigms differ most in what they cost, and a comparison that omits cost always
flatters the expensive one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "01_rag_local"))
sys.path.insert(0, str(HERE))

# Multi-hop questions. `chain` documents the join each requires; every gold
# answer was verified against the corpus by hand.
MULTIHOP = [
    {
        "id": "m1",
        "question": "What is the response time for the severity level that shed mode is classified as?",
        "chain": "shed mode -> SEV3 -> next business day",
        "must_contain": ["next business day"],
    },
    {
        "id": "m2",
        "question": "What action fixes the incident type that causes the most pages?",
        "chain": "most pages (41%) -> TLM-101 clock skew -> restart ntp-relay",
        "must_contain": ["ntp-relay"],
    },
    {
        "id": "m3",
        "question": "Who can access the data class with the longest retention period?",
        "chain": "longest retention (10 years) -> operator audit log -> security team",
        "must_contain": ["security team"],
    },
    {
        "id": "m4",
        "question": "What is the p50 latency of the model that reads Code-128 barcodes?",
        "chain": "Code-128 -> nw-barcode-ocr-v2 -> 7 ms",
        "must_contain": ["7 ms"],
    },
    {
        "id": "m5",
        "question": "How many robots were in the replay scenario recorded at the cell where Atlas was first deployed?",
        "chain": "first deployed -> Rotterdam -> peak-friday.yaml -> 214 robots",
        "must_contain": ["214"],
    },
    {
        "id": "m6",
        "question": "What minimum metric applies to the vision model type with 18 ms p50 latency?",
        "chain": "18 ms -> pallet detect -> mAP@0.5 >= 0.94",
        "must_contain": ["0.94"],
    },
    {
        "id": "m7",
        "question": "Where does a pallet go after the maximum number of failed barcode retries?",
        "chain": "3 failures -> manual inspection lane",
        "must_contain": ["manual inspection"],
    },
]


def load_standard() -> list[dict]:
    qa = json.loads((ROOT / "01_rag_local" / "eval" / "qa_set.json").read_text(encoding="utf-8"))
    return qa["questions"]


# ------------------------------------------------------------- paradigms


def make_vector():
    from rag.pipeline import RagPipeline

    pipe = RagPipeline.load(ROOT / "01_rag_local" / "index")

    def run(q: str) -> dict:
        t0 = time.perf_counter()
        r = pipe.ask(q)
        return {"answer": r.answer.text, "abstained": r.answer.abstained,
                "ms": (time.perf_counter() - t0) * 1000, "llm_calls": 1}
    return run


def make_graph():
    from graphrag.answer import ask_graph
    from graphrag.graph import KnowledgeGraph

    kg = KnowledgeGraph.load()

    def run(q: str) -> dict:
        t0 = time.perf_counter()
        r = ask_graph(q, kg)
        return {"answer": r.answer_text, "abstained": r.abstained,
                "ms": (time.perf_counter() - t0) * 1000, "llm_calls": r.llm_calls}
    return run


def make_agentic():
    from agentic.agent import AgenticRag

    agent = AgenticRag()

    def run(q: str) -> dict:
        r = agent.ask(q)
        return {"answer": r.answer, "abstained": r.abstained,
                "ms": r.total_ms, "llm_calls": r.llm_calls,
                "trajectory": r.trajectory()}
    return run


# ------------------------------------------------------------------ eval


def score(rows: list[dict], runner, name: str, verbose: bool) -> dict:
    facts, abstains, lat, calls = [], [], [], []
    details = []
    for q in rows:
        out = runner(q["question"])
        text_lc = out["answer"].lower()
        if q.get("type") == "unanswerable" or not q.get("must_contain"):
            ok = out["abstained"]
            abstains.append(1.0 if ok else 0.0)
        else:
            ok = all(s.lower() in text_lc for s in q["must_contain"]) and not out["abstained"]
            facts.append(1.0 if ok else 0.0)
        lat.append(out["ms"])
        calls.append(out["llm_calls"])
        details.append({"id": q.get("id", "?"), "pass": ok,
                        "answer": out["answer"][:160],
                        "ms": round(out["ms"]), "llm_calls": out["llm_calls"],
                        **({"trajectory": out["trajectory"]} if "trajectory" in out else {})})
        if verbose:
            mark = "PASS" if ok else "FAIL"
            print(f"    [{mark}] {name} {q.get('id','?')}: {out['answer'][:90]}")
    return {
        "fact_recall": statistics.mean(facts) if facts else None,
        "abstain_correct": statistics.mean(abstains) if abstains else None,
        "p50_ms": statistics.median(lat),
        "mean_llm_calls": statistics.mean(calls),
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["standard", "multihop", "both"], default="both")
    ap.add_argument("--paradigms", nargs="+", default=["vector", "graph", "agentic"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(HERE / "paradigm_results.json"))
    args = ap.parse_args()

    sets = {}
    if args.subset in ("standard", "both"):
        sets["standard"] = load_standard()
    if args.subset in ("multihop", "both"):
        sets["multihop"] = MULTIHOP

    makers = {"vector": make_vector, "graph": make_graph, "agentic": make_agentic}
    runners = {name: makers[name]() for name in args.paradigms}

    results: dict = {}
    for set_name, rows in sets.items():
        print(f"\n## {set_name} ({len(rows)} questions)\n")
        results[set_name] = {}
        for pname, runner in runners.items():
            t0 = time.perf_counter()
            s = score(rows, runner, pname, args.verbose)
            s["wall_s"] = round(time.perf_counter() - t0, 1)
            results[set_name][pname] = s
            fr = f"{s['fact_recall']:.0%}" if s["fact_recall"] is not None else "  -"
            ab = f"{s['abstain_correct']:.0%}" if s["abstain_correct"] is not None else "  -"
            print(f"  {pname:<8} fact recall {fr:>5} | abstain {ab:>5} | "
                  f"p50 {s['p50_ms']:>6.0f} ms | {s['mean_llm_calls']:.1f} LLM calls/q")

    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nresults -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
