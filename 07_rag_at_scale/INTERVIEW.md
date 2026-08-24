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
> capped recall at exactly 0.9 in a way that looked plausible. Second, the thing
> that actually bit me was a chunker bug producing 42× too many chunks, which
> also made my capacity estimate garbage until I found it. Post-fix the 82 shards
> I downloaded are ~279 M chunks — a 13 GB binary index, which fits in 61 GB of
> RAM. That fit is the whole reason the cascade exists.""

---

## Core questions

### "How do you fit hundreds of millions of vectors on one machine?"

You don't store them at float32. 279 M × 384 dims × 4 bytes is **428 GB**.

The cascade:

| precision | bytes/vector | 279 M vectors | where it lives |
|---|---:|---:|---|
| float32 | 1536 | 428 GB | never materialised |
| int8 | 384 | 107 GB | disk, memmapped |
| **binary** | **48** | **13 GB** | **RAM** |

Search is two stages. Stage 1 scans **all** the binary codes — Hamming distance
is XOR plus a popcount lookup, so it's memory-bandwidth work over 13 GB rather
than float math over 428 GB. That produces ~500 candidates. Stage 2 reads only
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
107 GB instead of 214 GB.

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

### "Tell me about a concurrency bug you wrote."

I added a producer/consumer pipeline so the GPU stopped waiting on Python.
It worked — the instrumentation showed **GPU busy 100%, starved 0%** for a whole
shard. Then it aborted after 3.25 M successfully embedded chunks:
`chunker: waited >900s for input`.

Completion was signalled by sentinels pushed through the **bounded** queues:

```python
try: self.raw_q.put(SENTINEL, timeout=30)
except queue.Full: pass
```

Under sustained backpressure that `put` times out and the sentinel is silently
dropped. The consumer then waits forever for a message that no longer exists.

The fix is a principle, not a patch: **never signal completion through a channel
that can drop messages.** `threading.Event` plus a live-producer counter;
consumers exit on "producer finished AND queue empty", checked in that order. An
Event cannot be lost to backpressure.

Two things I'd draw out. First, the stall detector I'd added earlier is what
turned this from a silent hang into a diagnosable message — an earlier version
of the same pipeline deadlocked at **0.00 CPU seconds** with no output at all,
and finding that needed a CPU-time sample rather than a traceback. Second, the
blast radius was set by commit granularity: because the manifest commits per
shard, 3.25 M chunks of correct work were discarded. Cheaper commits would have
bounded the loss.

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

I'd also push back gently on the premise: at 279 M vectors on one box, a managed
service's network round-trip alone can exceed the entire local search time.

### "Your capacity estimate was wrong twice. What happened?"

Worth owning both, because they're different kinds of wrong.

**First, compression.** I estimated from 200 GB of *text*, but the 200 GB is
compressed parquet — a 2.15 GB shard holds ~3.7 GB of raw text.

**Second, and much worse, a bug.** My chunker produced **42× too many chunks**:
mean 178 characters against a 1400-character target. For a document shorter than
the chunk size, `end` immediately hit the text end, the separator search was
skipped, and `pos` advanced *one character per iteration* — so a 500-char
document became ~180 near-identical chunks. On real text, 2,000 documents went
from 371,937 chunks to 9,937 once fixed.

Nothing about it looked broken. Chunks existed, contained text, embedded fine.
It surfaced only as "this shard is taking suspiciously long". Had it shipped:
42× the compute, and an index full of near-duplicates crowding out real results.

The lesson is the same for both: **measure one unit before extrapolating.** One
shard, twenty minutes, would have caught the compression ratio *and* the bug.

### "Describe a time your diagnosis was wrong."

Indexing was slower than the GPU should allow, and a download was running
concurrently. Obvious story: two jobs, one disk, the download is stealing I/O
bandwidth and starving the producers. I wrote it in the README.

Then I stopped the download to confirm — and throughput **fell**, to 248
chunks/s, with the GPU still pinned at 99%. `nvidia-smi` showed 2662/3090 MHz
and no throttle reasons. The GPU had been the bottleneck the entire time;
sustained rate alone is 632 chunks/s = 191k tokens/s.

The wrong diagnosis was *plausible*, which is what made it dangerous — it
explained the symptom and would have sent me optimising I/O. The measurement
that killed it took ninety seconds. I'd rather run the cheap experiment than
ship the plausible story.

### "How long does indexing actually take, and what limits it?"

Measured with nothing else running: **632 chunks/s × 302 tokens = 191k tokens/s**,
GPU-bound at 98–100% (2662/3090 MHz, no throttle reasons set).

The 82 shards I downloaded are ~71 billion tokens, so a full index is **103
hours**. That's not a tuning problem — it's what a 5070 Ti does with BGE-small at
that token volume. The honest planning unit is shards: one shard ≈ 3.4 M chunks ≈
1.6 hours.

I did add a producer/consumer pipeline, and it worked — the instrumentation shows
**GPU busy 100%, starved 0%** across a whole shard, where single-threaded it
idled. But that only removed the *Python* bottleneck; the GPU ceiling was always
the real one.

The one lever I'd try next is batch size, because it has a sharp cliff on real
302-token chunks:

```
batch 128:  191 chunks/s
batch 256:  196 chunks/s
batch 512:  783 chunks/s   <- 4x
```

The build already uses 512; 1024 is untested and is where I'd look before
accepting 103 hours.

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
  chunk*. At 279 M chunks that's months. It's for thousands-to-millions of
  chunks, or a high-value subset.
- **Semantic chunking** — cut on topic shift, thresholded by *percentile* of
  observed distances (absolute thresholds don't transfer across document styles).
  3–4× the embedding cost.
- **ColBERT** — one vector per token, MaxSim scoring. Ranks well and is
  interpretable. **Cannot be the primary index**: 279 M chunks → 3.1 TB even at
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
