"""
Step 2 - ask a question.

    python ask.py "what is the Rotterdam rule?"
    python ask.py "what does TLM-330 mean?" --show-context
    python ask.py "how long are vision frames kept?" --mode dense --no-rerank
    python ask.py --compare "who is on call for a SEV1?"     # all 3 retrieval modes

`--compare` is the interesting one: it runs dense-only, BM25-only and hybrid
side by side so you can see which documents each arm finds. That contrast is the
clearest way to explain in an interview *why* hybrid retrieval exists.
"""

import argparse
from pathlib import Path

from rag.generate import DEFAULT_MODEL
from rag.pipeline import RagPipeline

HERE = Path(__file__).parent


def show_hits(title: str, hits: list) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for i, h in enumerate(hits, 1):
        bits = [f"rrf={h.rrf_score:.4f}"]
        if h.dense_rank is not None:
            bits.append(f"dense#{h.dense_rank}")
        if h.bm25_rank is not None:
            bits.append(f"bm25#{h.bm25_rank}")
        if h.rerank_score is not None:
            bits.append(f"rerank={h.rerank_score:+.2f}")
        print(f"  [{i}] {h.breadcrumb}")
        print(f"      {'  '.join(bits)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="What is the Rotterdam rule?")
    ap.add_argument("--index", default=str(HERE / "index"))
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for generation")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--top-k", type=int, default=4, help="context blocks sent to the LLM")
    ap.add_argument("--candidates", type=int, default=20, help="retrieved before reranking")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--compare", action="store_true", help="compare retrieval modes, no generation")
    args = ap.parse_args()

    index = Path(args.index)
    if not index.exists():
        raise SystemExit(f"no index at {index.resolve()} -- run:  python ingest.py")

    pipe = RagPipeline.load(
        index,
        use_reranker=not args.no_rerank,
        llm_model=args.model,
        candidates=args.candidates,
        top_k=args.top_k,
    )
    print(f"Index: {len(pipe.store)} chunks | embedder: {pipe.store.model_name} | "
          f"reranker: {'on' if pipe.reranker else 'off'}")
    print(f"\nQ: {args.question}")

    if args.compare:
        for mode in ("dense", "bm25", "hybrid"):
            hits, r_ms, rr_ms = pipe.retrieve(args.question, mode=mode)
            show_hits(f"{mode.upper()}  (retrieve {r_ms:.0f} ms, rerank {rr_ms:.0f} ms)", hits)
        return

    result = pipe.ask(args.question, mode=args.mode)

    show_hits(f"Retrieved context ({args.mode})", result.hits)

    if args.show_context:
        for i, h in enumerate(result.hits, 1):
            print(f"\n--- block [{i}] {h.source} " + "-" * 30)
            print(h.body)

    a = result.answer
    print("\n" + "=" * 70)
    print(a.text)
    print("=" * 70)

    status = "ABSTAINED" if a.abstained else ("grounded" if a.grounded else "UNGROUNDED")
    print(f"\ncitations: {a.citations or 'none'}  ->  {status}")
    if a.invalid_citations:
        print(f"  !! hallucinated block numbers: {a.invalid_citations}")
    for c in a.valid_citations:
        m = a.contexts[c - 1]
        print(f"  [{c}] {m['source']}  ({m['breadcrumb']})")

    print(
        f"\ntiming: retrieve {result.retrieve_ms:.0f} ms | rerank {result.rerank_ms:.0f} ms | "
        f"generate {result.generate_ms:.0f} ms | total {result.total_ms:.0f} ms"
    )
    print(f"tokens: {a.prompt_tokens} prompt -> {a.eval_tokens} generated  ({a.model})")


if __name__ == "__main__":
    main()
