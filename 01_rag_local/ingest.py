"""
Step 1 - build the index.

    python ingest.py                      # corpus/ -> index/
    python ingest.py --show-chunks 3      # also print the first 3 chunks

Chunks the corpus, embeds every chunk on the GPU, writes a FAISS index plus a
JSONL of chunk metadata to `index/`. Takes a couple of seconds on this corpus;
re-run it any time you edit the documents.
"""

import argparse
import time
from pathlib import Path

from rag.chunking import chunk_corpus
from rag.embed import Embedder, DEFAULT_MODEL
from rag.store import VectorStore

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "corpus"))
    ap.add_argument("--index", default=str(HERE / "index"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=320, help="target chunk size")
    ap.add_argument("--overlap", type=int, default=60, help="overlap between split sections")
    ap.add_argument("--show-chunks", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    docs = sorted(corpus.glob("*.md"))
    if not docs:
        raise SystemExit(f"no .md files in {corpus.resolve()}")

    print(f"Corpus: {len(docs)} documents in {corpus.resolve()}")

    t0 = time.perf_counter()
    chunks = chunk_corpus(corpus, max_tokens=args.max_tokens, overlap_tokens=args.overlap)
    t_chunk = time.perf_counter() - t0

    sizes = [c.n_tokens for c in chunks]
    print(
        f"Chunked:  {len(chunks)} chunks in {t_chunk*1000:.0f} ms  "
        f"(tokens: min {min(sizes)}, median {sorted(sizes)[len(sizes)//2]}, max {max(sizes)})"
    )

    for c in chunks[: args.show_chunks]:
        print("\n" + "-" * 70)
        print(f"{c.id}  |  {c.breadcrumb}  |  ~{c.n_tokens} tokens")
        print("-" * 70)
        print(c.body[:400] + ("..." if len(c.body) > 400 else ""))

    print(f"\nEmbedding with {args.model} ...")
    embedder = Embedder(args.model)
    print(f"  device={embedder.device}  dim={embedder.dim}")

    t0 = time.perf_counter()
    vectors = embedder.encode_passages([c.text for c in chunks])
    t_embed = time.perf_counter() - t0
    print(f"  {len(vectors)} vectors in {t_embed:.2f}s  ({len(vectors)/t_embed:.0f} chunks/s)")

    store = VectorStore.build(vectors, [c.to_dict() for c in chunks], args.model)
    out = Path(args.index)
    store.save(out)

    size_mb = sum(f.stat().st_size for f in out.iterdir()) / 1024**2
    print(f"\nIndex written to {out.resolve()}  ({size_mb:.2f} MB)")
    print("Next:  python ask.py \"what is the Rotterdam rule?\"")


if __name__ == "__main__":
    main()
