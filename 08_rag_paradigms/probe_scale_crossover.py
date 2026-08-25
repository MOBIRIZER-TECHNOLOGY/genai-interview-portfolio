"""
Test the claim this project rests on: that vector RAG wins multi-hop **because
the corpus is small**, and stops winning as it grows.

    python probe_scale_crossover.py
    python probe_scale_crossover.py --sizes 30 1000 10000 100000 --k 4

## Why this exists

`evaluate_paradigms.py` measures that vector RAG scores 100% on multi-hop while
graph and agentic score 29%. The README explains it with a *mechanism*: at 30
chunks, top-4 retrieval pulls **both** hop chunks into context at once, so the
generator does the join in-context and no graph is needed. It then makes a
prediction: at project-07 scale, top-4 of 13.6 M cannot do that, and the
machinery starts to matter.

That prediction was never tested. It is the most load-bearing sentence in the
project and it was an argument, not a measurement.

## Method

Take the same 7 multi-hop questions and the Atlas corpus, then **dilute** the
corpus with real distractor passages drawn from project 07's FineWeb-Edu shards.
At each corpus size, embed, retrieve top-k for each question, and ask one thing:

    are BOTH hop chunks still in the retrieved set?

That is co-retrieval, the mechanism the whole explanation depends on. It needs no
LLM and no generation, which is the point -- it isolates retrieval from
answering, so the number cannot be confounded by the generator being clever.

The hop markers below were derived from each question's documented `chain` and
verified against the corpus by hand. A hop is "retrieved" when a retrieved chunk
contains its marker text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "01_rag_local"))

CORPUS = ROOT / "01_rag_local" / "corpus"
FINEWEB = Path("C:/genai-data/hf")

# (question id, hop-1 marker, hop-2 marker) -- the two facts that must meet.
# Markers are lowercase substrings; a hop counts as retrieved when some
# retrieved chunk contains it.
HOPS = [
    ("m1", "shed mode", "next business day"),
    ("m2", "41%", "ntp-relay"),
    ("m3", "10 years", "security team"),
    ("m4", "code-128", "7 ms"),
    ("m5", "rotterdam", "214"),
    ("m6", "18 ms", "0.94"),
    ("m7", "3 failures", "manual inspection"),
]

QUESTIONS = {
    "m1": "What is the response time for the severity level that shed mode is classified as?",
    "m2": "What action fixes the incident type that causes the most pages?",
    "m3": "Who can access the data class with the longest retention period?",
    "m4": "How fast is the model that reads Code-128 barcodes?",
    "m5": "How many robots run the scenario used by the cell that was first deployed?",
    "m6": "What accuracy is required of the model that runs in 18 ms?",
    "m7": "Where does a pallet go after 3 failures?",
}


def load_distractors(n: int) -> list[str]:
    """Real passages from project 07's corpus -- not lorem ipsum.

    Synthetic filler would understate the difficulty: random text is trivially
    far from every query in embedding space, so it would never displace a hop
    chunk and the crossover would never appear. Real web text is the honest
    distractor because some of it genuinely resembles the question.
    """
    import pyarrow.parquet as pq

    shards = sorted(FINEWEB.rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"no FineWeb shards under {FINEWEB} -- see project 07")
    out: list[str] = []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            for t in batch.column("text").to_pylist():
                if t and len(t) > 200:
                    out.append(t[:1200])
                    if len(out) >= n:
                        return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[30, 1000, 10000, 100000])
    ap.add_argument("--k", type=int, default=4, help="top-k, matching the pipeline")
    ap.add_argument("--out", default=str(HERE / "scale_crossover.json"))
    args = ap.parse_args()

    from rag.chunking import chunk_corpus
    from sentence_transformers import SentenceTransformer

    atlas = [c.text for c in chunk_corpus(CORPUS)]
    bodies = [c.body for c in chunk_corpus(CORPUS)]
    print(f"Atlas corpus: {len(atlas)} chunks")

    biggest = max(args.sizes)
    n_distract = max(0, biggest - len(atlas))
    print(f"loading {n_distract:,} FineWeb distractor passages ...")
    distractors = load_distractors(n_distract)

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    print("embedding ...")
    atlas_v = model.encode(atlas, batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    dist_v = model.encode(distractors, batch_size=256, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True) \
        if distractors else np.zeros((0, atlas_v.shape[1]), dtype=atlas_v.dtype)

    qids = [h[0] for h in HOPS]
    q_v = model.encode([f"Represent this sentence for searching relevant passages: {QUESTIONS[q]}"
                        for q in qids], convert_to_numpy=True, normalize_embeddings=True)

    rows = []
    print(f"\n{'corpus':>10} {'both hops':>11} {'hop1 only':>10} {'neither':>8}   per-question")
    print("-" * 78)
    for size in args.sizes:
        take = max(0, size - len(atlas))
        vecs = np.vstack([atlas_v, dist_v[:take]]) if take else atlas_v
        texts = bodies + distractors[:take]

        both = hop1 = neither = 0
        detail = []
        for i, (qid, m1, m2) in enumerate(HOPS):
            scores = vecs @ q_v[i]
            top = np.argpartition(-scores, min(args.k, len(scores) - 1))[:args.k]
            got = " ".join(texts[j].lower() for j in top)
            a, b = m1 in got, m2 in got
            detail.append({"q": qid, "hop1": a, "hop2": b})
            if a and b:
                both += 1
            elif a or b:
                hop1 += 1
            else:
                neither += 1
        marks = "".join("B" if d["hop1"] and d["hop2"] else
                        ("." if d["hop1"] or d["hop2"] else "x") for d in detail)
        print(f"{size:>10,} {both}/{len(HOPS):>9} {hop1:>10} {neither:>8}   {marks}")
        rows.append({"corpus_size": size, "k": args.k, "both_hops": both,
                     "one_hop": hop1, "neither": neither, "detail": detail})

    Path(args.out).write_text(json.dumps(
        {"k": args.k, "n_atlas": len(atlas), "rows": rows}, indent=1), encoding="utf-8")
    print(f"\nB = both hops retrieved (the join is free) · . = one hop · x = neither")
    print(f"results -> {args.out}")

    first, last = rows[0], rows[-1]
    print(f"\nco-retrieval at {first['corpus_size']:,} chunks: "
          f"{first['both_hops']}/{len(HOPS)}   "
          f"at {last['corpus_size']:,}: {last['both_hops']}/{len(HOPS)}")
    if last["both_hops"] < first["both_hops"]:
        print("The mechanism degrades with scale, as the README predicts -- which is\n"
              "the measured case for graph/agentic machinery at larger corpora.")
    else:
        print("Co-retrieval did NOT degrade over this range. The README's scale\n"
              "argument is not supported by this experiment and needs revising.")


if __name__ == "__main__":
    main()
