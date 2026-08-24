# 🎤 Interview notes — RAG at scale

---

## The 60-second pitch

> "RAG over a real 200 GB corpus — FineWeb-Edu — on one consumer machine. The
> whole design is a precision cascade: 1-bit binary codes for the full scan, int8
> on disk to rescore the top few hundred. That's a 32× memory reduction, and I
> measured what it costs: 0.985 recall@10 against exact float32 search, with a
> quality ratio of 1.0000.
>
> Two things I'd want to talk about. First, I nearly got the measurement wrong
> twice — validating on synthetic vectors said the technique doesn't work
> (recall 0.18 vs 0.62 on real embeddings), and leaving the query in the corpus
> capped recall at exactly 0.9 in a way that looked plausible. Second, my sizing
> was wrong: 200 GB of compressed parquet is ~500 GB of text, so 480 M chunks
> rather than the 143 M I'd projected. Binary index goes from 7 GB to 23 GB —
> still fits in 61 GB of RAM, which is the whole reason the cascade exists."

---

## Core questions

### "How do you fit hundreds of millions of vectors on one machine?"

You don't store them at float32. 480 M × 384 dims × 4 bytes is **715 GB**.

The cascade:

| precision | bytes/vector | 480 M vectors | where it lives |
|---|---:|---:|---|
| float32 | 1536 | 715 GB | never materialised |
| int8 | 384 | 184 GB | disk, memmapped |
| **binary** | **48** | **23 GB** | **RAM** |

Search is two stages. Stage 1 scans **all** the binary codes — Hamming distance
is XOR plus a popcount lookup, so it's memory-bandwidth work over 23 GB rather
than float math over 715 GB. That produces ~500 candidates. Stage 2 reads only
those 500 rows from the int8 memmap (192 KB), decodes, and does exact dot
products to fix the ordering.

Stage 1 only has to get the true neighbours *into* the candidate set. Stage 2
does the ranking. That division is why 1-bit precision is survivable.

### "Why does binarising at zero not destroy the embedding?"

The model is trained to produce L2-normalised vectors where **direction** carries
the meaning, and each dimension is roughly zero-centred. Taking `> 0` keeps the
sign — the dominant bit — and discards magnitude. Recall drops from 1.0 to 0.62
on its own, which sounds fatal until you remember stage 1 isn't ranking, it's
filtering.

And this is exactly where I'd flag the trap: **it doesn't work on random
vectors.** Isotropic Gaussians on the unit sphere give binary-only recall of
0.18. Every dimension is independent, so the sign pattern carries almost no
neighbour information. Real embeddings are strongly anisotropic and clustered.
I have both numbers in the repo because validating on synthetic data would have
led me to throw the technique away.

### "int8 or float16 for the rescore stage?"

Measured: int8 gives 0.985 recall@10, float16 gives 0.999. Looks like float16
wins — until you look at the **quality ratio**, the mean similarity of what you
retrieved over the mean similarity of the exact top-10. Both are **1.0000**.

The 1.4% of documents int8 "misses" are near-ties: cosine 0.8112 versus 0.8109.
No user can tell those apart. So int8 is the right call, and it halves the disk —
184 GB instead of 368 GB.

The general lesson: **recall against exact search is a proxy, not the goal.** On
real corpora with many near-duplicates, chasing the last point of recall buys
nothing a user perceives. Reporting a quality measure alongside recall is what
lets you make that trade honestly.

### "Walk me through a bug you found in your own measurement."

Recall plateaued at exactly 0.899 no matter how many candidates I added. Binary
alone was 0.62, so the candidates clearly *were* improving — but the ceiling
never moved.

The cause: my query vectors were sampled *from* the corpus and were still
indexed. A vector's nearest neighbour is itself, so it always took one of the 10
slots. With k=10 that caps recall at exactly 9/10.

Two things gave it away: the plateau, and the suspiciously round number. Excluding
self from both the ground truth and the results took the same configuration from
0.899 to 0.985.

What makes this worth telling: 0.899 is *plausible*. It's not obviously broken.
If I'd shipped it I'd have spent a week tuning candidate depth against a ceiling
that had nothing to do with candidate depth.

### "How does latency scale, and what breaks first?"

Four costs that scale completely differently:

| stage | scales with |
|---|---|
| embed the query | nothing — constant |
| **binary scan** | **O(n)** — the one that grows |
| int8 rescore | candidate depth, not n |
| fetch text | k |

So the question isn't "how fast is the index", it's **"at what n does the linear
scan blow the latency budget"** — everything else is constant. `bench_latency.py`
measures the curve by subsampling and projects it. Subsampling is legitimate here
precisely *because* stage 1 is a full linear scan: cost is proportional to row
count and independent of which rows. That would not be valid for HNSW, where
you'd have to rebuild the graph at each size.

When the flat scan stops fitting, the answer is IVF or HNSW — partition the space
so a query touches a fraction of it. The trade is that recall becomes a
**tunable** (`nprobe`, `efSearch`) rather than a guarantee, so you've swapped a
known cost for a new error budget you now have to measure.

### "Why not just use Pinecone / Qdrant / FAISS?"

In production, you should — and Qdrant and FAISS implement exactly this cascade.
Building it by hand here was to make the memory arithmetic visible, because
that's what lets you *configure* them correctly. If you don't know why binary
plus rescore works, you can't reason about `nprobe`, quantisation settings, or
why your recall dropped after an index rebuild.

I'd also push back gently on the premise: at 480 M vectors on one box, a managed
service's network round-trip alone can exceed the entire local search time.

### "You said 200 GB but got 480 M chunks. Explain."

That was my sizing error and it's a good one to own. I estimated from 200 GB of
*text*: 50 B tokens ÷ 350 tokens/chunk ≈ 143 M chunks.

But the 200 GB is **compressed parquet**. Measured, each 2.05 GB shard yields
~5.2 M chunks, so 93 shards is ~480 M — roughly 500 GB of raw text. Everything
downstream moves: binary index 7 GB → 23 GB, int8 184 GB, embedding time 8 h →
~43 h at the measured end-to-end rate.

Two lessons. **Measure one unit before extrapolating** — one shard would have
caught this in 20 minutes. And **compression ratio is not a detail** when your
capacity plan is denominated in bytes on disk.

### "Your embedding throughput dropped from 4,900/s to 3,100/s. Why?"

4,900 chunks/s is the pure GPU number, measured in isolation. 3,100/s is
end-to-end. The gap is parquet decode, chunking and disk writes sitting on the
critical path, single-threaded, not overlapped with the GPU — so the GPU idles
while Python parses.

The fix is a producer/consumer split: worker threads decode and chunk into a
queue, the main thread does nothing but feed the GPU. That should recover most of
the difference and cut the ~43 h materially. I know it's the bottleneck because
GPU utilisation sits around 60–95% rather than pinned.

### "Which modern techniques did you implement, and when are they wrong?"

That second half is the real question. Everything has a regime.

- **HyDE** — generate a hypothetical answer, embed *that*. Turns a hard
  question↔passage comparison into an easy passage↔passage one. **Wrong for
  identifier lookups**: asked about the auction, my model generated "the robot
  with the *highest* bid score" when Atlas awards the *lowest*. For `TLM-330`
  it'd invent a code and drag retrieval toward it.
- **Multi-query** — several paraphrases, fuse with RRF. A recall technique that
  costs precision; pairs with a reranker.
- **Decomposition** — split multi-hop questions. Only handles *parallel*
  decomposition; sequential multi-hop needs an agent loop.
- **Routing** — pick strategy per query. Adds an LLM call to the critical path;
  at high QPS you'd distil it into a fine-tuned 0.5B or logistic regression on
  the query embedding.
- **Contextual retrieval** — ~35% fewer retrieval failures, but one LLM call *per
  chunk*. At 480 M chunks that's months. It's for thousands-to-millions of
  chunks, or a high-value subset.
- **Semantic chunking** — cut on topic shift, thresholded by *percentile* of
  observed distances (absolute thresholds don't transfer across document styles).
  3–4× the embedding cost.
- **ColBERT** — one vector per token, MaxSim scoring. Ranks well and is
  interpretable. **Cannot be the primary index**: 480 M chunks → 5.2 TB even at
  2-bit compression, versus 7.4 GB chunk-level. Its place is second-stage
  reranking over candidates the binary index already returned.
- **Matryoshka** — truncate dims and renormalise. Only valid on models *trained*
  with the MRL objective.

### "How do you evaluate this?"

Two layers, and keeping them separate is the point.

**Deterministic first** — recall@k, MRR, mechanical citation verification
(project 01). Free, reproducible, cannot hallucinate. Do these before reaching
for a judge.

**LLM judge for what they can't see** — RAGAS faithfulness, answer relevance,
context precision, context recall. The value is in the pairs: low faithfulness
with *high* context recall means the model ignored good context (fix the prompt);
with *low* recall it means retrieval starved it (fix retrieval). Same symptom,
opposite fix.

And I'd state the caveat unprompted: a judge is an instrument with its own error
rate — biased toward verbose answers, capped by its own competence, inconsistent
near boundaries. Use it for *relative* comparison between systems, not as truth.

---

## Numbers to have ready

- **Storage:** `n × dim × bytes`. Binary is `dim/8` bytes — 32× smaller than fp32.
- **Measured cascade:** binary-only 0.624 → +int8 rescore@500 **0.985**, quality **1.0000**.
- **Synthetic control:** 0.180 — the number that would have killed the project.
- **Embedding:** 4,900 chunks/s pure GPU (fp16, batch 512), 3,100/s end-to-end.
- **This corpus:** ~5.2 M chunks per 2.05 GB parquet shard.

---

## Questions to ask *them*

- "What's your recall target, and how do you measure it — against exact search,
  or against human relevance labels?"
- "Where does your index live in the memory hierarchy, and what happens when it
  stops fitting?"
- "Do you re-measure recall after an index rebuild or an embedding model change?"
- "How do you decide chunk size, and when did you last revisit it?"

---

## Related projects

- **[01_rag_local](../01_rag_local/)** — hybrid retrieval, reranking, grounding
- **[06_local_gpu_inference](../06_local_gpu_inference/)** — the same
  which-resource-are-you-short-of discipline on the inference side
