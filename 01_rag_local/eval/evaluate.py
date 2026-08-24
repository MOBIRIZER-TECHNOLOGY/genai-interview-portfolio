"""
Evaluation harness. This is the part that turns a demo into engineering.

    python eval/evaluate.py --retrieval-only      # fast, no LLM (seconds)
    python eval/evaluate.py                       # full, includes generation

It measures two separate things, and keeping them separate is the whole point:

RETRIEVAL  (does the right document reach the context window?)
  recall@k  - fraction of questions where a gold document appears in the top k
  MRR       - 1/rank of the first gold document, averaged. Rewards ranking it #1,
              not merely including it.
  Reported for every combination of {dense, bm25, hybrid} x {rerank on, off}, so
  you can point at a number when you claim hybrid + reranking is worth it.

GENERATION (given good context, does the answer stay true?)
  fact recall    - the required strings appear in the answer
  citation valid - every [n] the model emitted maps to a real context block
  abstain rate   - on the 'unanswerable' questions, does it say NOT_FOUND
                   instead of inventing something

Why measure abstention explicitly: the dangerous failure of RAG is not a missing
answer, it is a fluent wrong one. A system that scores 90% on answerable
questions and 0% on abstention is worse in production than one that scores 80%
and 100%, because the first one lies without warning.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from rag.generate import DEFAULT_MODEL  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402


# ------------------------------------------------------------------ metrics


def recall_at_k(hits: list, gold: list[str], k: int) -> float:
    sources = {h.source for h in hits[:k]}
    return 1.0 if any(g in sources for g in gold) else 0.0


def reciprocal_rank(hits: list, gold: list[str]) -> float:
    for i, h in enumerate(hits, start=1):
        if h.source in gold:
            return 1.0 / i
    return 0.0


# -------------------------------------------------------------- retrieval


def eval_retrieval(pipe: RagPipeline, questions: list[dict], k: int) -> dict:
    """Sweep {dense, bm25, hybrid} x {rerank off, on} over the answerable set."""
    answerable = [q for q in questions if q["gold_sources"]]
    rows = {}

    original_reranker = pipe.reranker
    for use_rr in (False, True):
        pipe.reranker = original_reranker if use_rr else None
        if use_rr and original_reranker is None:
            continue
        for mode in ("dense", "bm25", "hybrid"):
            r1, r3, rr, lat = [], [], [], []
            for q in answerable:
                t0 = time.perf_counter()
                hits, _, _ = pipe.retrieve(q["question"], mode=mode)
                lat.append((time.perf_counter() - t0) * 1000)
                r1.append(recall_at_k(hits, q["gold_sources"], 1))
                r3.append(recall_at_k(hits, q["gold_sources"], min(3, k)))
                rr.append(reciprocal_rank(hits, q["gold_sources"]))
            label = f"{mode}{' + rerank' if use_rr else ''}"
            rows[label] = {
                "recall@1": statistics.mean(r1),
                f"recall@{min(3, k)}": statistics.mean(r3),
                "mrr": statistics.mean(rr),
                "p50_ms": statistics.median(lat),
                "n": len(answerable),
            }
    pipe.reranker = original_reranker
    return rows


# -------------------------------------------------------------- generation


def eval_generation(pipe: RagPipeline, questions: list[dict], mode: str) -> dict:
    answerable, unanswerable, details = [], [], []

    for q in questions:
        result = pipe.ask(q["question"], mode=mode)
        a = result.answer
        text_lc = a.text.lower()

        if q["type"] == "unanswerable":
            ok = a.abstained
            unanswerable.append(1.0 if ok else 0.0)
            details.append(
                {
                    "id": q["id"], "type": q["type"], "abstained": a.abstained,
                    "pass": ok, "answer": a.text[:200], "ms": round(result.total_ms),
                }
            )
            continue

        missing = [s for s in q["must_contain"] if s.lower() not in text_lc]
        gold_retrieved = any(h.source in q["gold_sources"] for h in result.hits)
        ok = not missing and not a.abstained
        answerable.append(
            {
                "fact": 1.0 if ok else 0.0,
                "cite": 1.0 if not a.invalid_citations and a.citations else 0.0,
                "retrieved": 1.0 if gold_retrieved else 0.0,
                "ms": result.total_ms,
            }
        )
        details.append(
            {
                "id": q["id"], "type": q["type"], "pass": ok, "missing": missing,
                "gold_retrieved": gold_retrieved, "citations": a.citations,
                "invalid_citations": a.invalid_citations,
                "answer": a.text[:200], "ms": round(result.total_ms),
            }
        )

    return {
        "fact_recall": statistics.mean(a["fact"] for a in answerable) if answerable else 0.0,
        "citation_valid": statistics.mean(a["cite"] for a in answerable) if answerable else 0.0,
        "gold_in_context": statistics.mean(a["retrieved"] for a in answerable) if answerable else 0.0,
        "abstain_correct": statistics.mean(unanswerable) if unanswerable else 0.0,
        "p50_latency_ms": statistics.median([a["ms"] for a in answerable]) if answerable else 0.0,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "details": details,
    }


# -------------------------------------------------------------------- main


def print_table(rows: dict) -> None:
    cols = list(next(iter(rows.values())).keys())
    width = max(len(k) for k in rows) + 2
    print(f"{'retriever':<{width}}" + "".join(f"{c:>13}" for c in cols))
    print("-" * (width + 13 * len(cols)))
    for name, vals in rows.items():
        cells = "".join(
            f"{v:>13.3f}" if isinstance(v, float) else f"{v:>13}" for v in vals.values()
        )
        print(f"{name:<{width}}{cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(HERE.parent / "index"))
    ap.add_argument("--qa", default=str(HERE / "qa_set.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM (fast)")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    qa = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    questions = qa["questions"]
    pipe = RagPipeline.load(args.index, llm_model=args.model, top_k=args.top_k)

    print("=" * 78)
    print(f"  RAG evaluation  |  {len(pipe.store)} chunks  |  top_k={args.top_k}")
    print("=" * 78)

    print("\n## Retrieval quality (answerable questions only)\n")
    retrieval = eval_retrieval(pipe, questions, args.top_k)
    print_table(retrieval)
    print(
        "\n  recall@1 = gold doc ranked first.  mrr = 1/rank of first gold doc.\n"
        "  Compare the rows: this is the evidence that hybrid + reranking earns its cost."
    )

    results = {"retrieval": retrieval, "config": {"top_k": args.top_k, "mode": args.mode}}

    if not args.retrieval_only:
        print(f"\n## Generation quality  (mode={args.mode}, model={args.model})\n")
        gen = eval_generation(pipe, questions, args.mode)
        for key in ("gold_in_context", "fact_recall", "citation_valid", "abstain_correct"):
            print(f"  {key:<18} {gen[key]:.1%}")
        print(f"  {'p50 latency':<18} {gen['p50_latency_ms']:.0f} ms")
        print(f"  n = {gen['n_answerable']} answerable + {gen['n_unanswerable']} unanswerable")

        failures = [d for d in gen["details"] if not d["pass"]]
        if failures:
            print(f"\n  {len(failures)} failure(s):")
            for d in failures:
                why = (
                    "did not abstain" if d["type"] == "unanswerable"
                    else f"missing {d.get('missing')}"
                    + ("" if d.get("gold_retrieved") else "  [RETRIEVAL MISS]")
                )
                print(f"    {d['id']} ({d['type']}): {why}")
                print(f"        {d['answer'][:120]}")
        else:
            print("\n  All questions passed.")
        results["generation"] = gen

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
