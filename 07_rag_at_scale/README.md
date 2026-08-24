# 🗄️ Project 07 — RAG at 200 GB: quantisation, latency, and modern retrieval

Build a retrieval system over a **real 200 GB corpus** on one consumer machine —
and measure what actually breaks first.

> **In one sentence:** binary quantisation gives a **32× memory reduction** at
> **0.985 recall@10 and a quality ratio of 1.0000**, which is the only reason
> hundreds of millions of vectors fit in 61 GB of RAM.

---

## 🧠 The idea (for non-experts)

Project 01 does RAG over 8 documents. Everything is easy at that size: hold all
30 chunks in memory, compare the query against every one, done.

At 200 GB none of that works, and the reasons are worth stating precisely:

| what | at 8 documents | at 200 GB |
|---|---|---|
| corpus | fits in RAM | **streamed**, never loaded |
| vectors (float32) | 46 KB | **~715 GB** — will not fit anywhere |
| search | compare against all 30 | comparing against all is the bottleneck |
| chunk text | stored with the vector | storing it re-materialises the corpus |

So the engineering is entirely about **what you refuse to keep**.

---

## ✅ The core result: the precision cascade

The whole design rests on one measured claim. From `validate_quantization.py`,
on **real embeddings** (20,000 wikitext passages, BGE-small, 384 dims):

| stage | recall@10 | quality ratio | ms/query |
|---|---:|---:|---:|
| binary only (no rescore) | 0.624 | — | 1.57 |
| binary → int8 rescore, 100 cand | 0.960 | 0.9994 | 1.68 |
| **binary → int8 rescore, 500 cand** | **0.985** | **1.0000** | **1.85** |
| binary → float16 rescore, 500 cand | 0.999 | 1.0000 | 1.90 |
| binary → float16 rescore, 1000 cand | 1.000 | 1.0000 | 2.39 |

**Memory: binary 1.0 MB · int8 7.7 MB · float32 30.7 MB — a 32× reduction.**

Two decisions fall straight out of this table:

- **int8 beats float16 for rescoring**, despite 0.985 vs 0.999 recall, because the
  *quality ratio is identical at 1.0000*. The 1.4% of "missed" documents are
  near-ties — cosine 0.8112 vs 0.8109 — that no user could distinguish. int8
  costs half the disk (55 GB vs 110 GB at 143M vectors).
- **500 candidates is the knee.** 100 → 500 buys 2.5 points; 500 → 1000 buys 0.1.

### 🔍 Two traps that produce the wrong answer

**Trap 1 — validating on synthetic vectors.** The same test on isotropic Gaussian
vectors gives binary-only recall of **0.180** versus **0.624** on real
embeddings. Random vectors are the worst case for binary quantisation: every
dimension is independent, so the sign pattern carries almost no neighbour
information. Real embeddings are strongly anisotropic. *If you validate this
technique on `np.random.normal` you will conclude it doesn't work and throw away
a 32× win.* The control is in the script (`--include-synthetic`).

**Trap 2 — leaving the query in the corpus.** My first run showed recall
plateauing at exactly 0.899 no matter how many candidates I added. The cause: the
query vector was also indexed, so it was always its own nearest neighbour and
permanently occupied one of the 10 slots — capping recall at 0.9. Both the
plateau and the suspiciously round number were the clue. Excluding self took the
same configuration from 0.899 to 0.985.

Neither number is wrong in a way that looks wrong. That is what makes them worth
writing down.

---

## 🐛 The chunker bug that cost 42× the compute

The first build run produced **15 million chunks from one shard** and was still
going. That number was wrong by a factor of 42, and the bug is worth studying
because nothing about it looked broken.

`chunk_text` advanced with `pos = max(pos + 1, end - overlap_chars)`. For a
document **shorter than the chunk size**, `end` immediately equals `len(text)`,
the separator search is skipped (`if end < n`), so `end` never moves again — and
`pos` then advances by **one character per iteration**. A 500-character document
yielded ~180 near-identical chunks instead of 1.

Measured on real FineWeb text, before and after the fix:

| | before | after |
|---|---:|---:|
| chunks from 2,000 docs (10.1 MB) | 371,937 | **9,937** |
| mean chars per chunk | 178 | **1,209** (target 1400) |
| vs. the expected count at stride 1160 | 42.68× | **1.14×** ← the overlap |
| short doc (500 chars) | ~180 chunks | **1 chunk** |

Two things made it hard to see: the output was *plausible* — chunks existed, had
text in them, and embedded fine — and the failure only triggers on short
documents, so any test with a few long paragraphs passes. It only showed up as
"this shard is taking suspiciously long".

The cost if it had shipped: 42× the embedding compute, an index full of
near-duplicate chunks that would crowd out real results, and a corpus-size
estimate off by an order of magnitude.

---

## 📏 The corpus is bigger than "200 GB" suggests

**200 GB of compressed parquet is ~344 GB of raw text.** FineWeb-Edu shards are
~2.15 GB each and yield **~3.6 M chunks** apiece after the fix:

| | initial estimate | measured (post-fix) |
|---|---:|---:|
| chunks from 93 shards | 143 M | **~316 M** (3.4 M/shard measured) |
| binary index (RAM) | 6.9 GB | **16.1 GB** (fits in 61 GB) |
| int8 rescore (disk) | 55 GB | **129 GB** (fits in 1.6 TB) |

The lesson: **measure one unit before extrapolating.** One shard would have
caught both the compression ratio and the chunker bug in twenty minutes.

---

## ⚡ Producer/consumer threading

The single-threaded build reached **3,100 chunks/s** against a **4,900 chunks/s**
GPU ceiling — the GPU idled roughly a third of the run while Python decoded
parquet and sliced strings:

```
read parquet -> chunk text -> EMBED ON GPU -> quantise -> write
[------- CPU, GIL-bound -------]  [-- GPU --]  [-- CPU --]
```

`scale/pipeline.py` overlaps them:

```
[reader]       parquet row-batches      -> raw_q
[chunker x3]   text -> (texts, coords)  -> embed_q
[main]         embed on GPU, quantise, AND write
```

### The deadlock I had to remove

The first version also had a **writer thread**. It deadlocked: the main thread
blocked on a full `write_q` while the writer was stuck, and because no queue
operation had a timeout, the process sat at **0.00 CPU seconds** indefinitely
with no error and no log line. Diagnosing it took a CPU-time sample, not a
traceback — a hung Python process tells you nothing on its own.

Two fixes, and the second is the more important lesson:

1. **Deleted the writer thread.** It bought almost nothing — appending ~1 MB to
   the OS buffer is sub-millisecond and the expensive `fsync` happens once per
   shard — while adding a whole deadlock surface. The real win is overlapping
   *read and chunk* with the GPU, which is preserved.
2. **Every blocking queue call now has a timeout and checks a shutdown event.**
   A stall surfaces as `"chunker: blocked >900s on a full queue"` instead of
   silence.

The general rule: **a concurrent stage that isn't on the critical path is pure
risk.** Add threads where the time actually goes, nowhere else.

**Why threads work here despite the GIL:** chunking is pure Python and does hold
it, but `pyarrow` releases the GIL during parquet decode, HF `tokenizers` is Rust
and releases it, and every CUDA op releases it. So while the GPU works, the
interpreter is free and the chunkers run in that window.

Every queue is **bounded**. An unbounded queue in front of a GPU turns a
speedup into an OOM, because readers happily materialise the whole shard in RAM.
`maxsize` makes a fast producer block instead — which is the backpressure you
want.

The run reports **GPU-busy percentage** and where each stage blocked, so the
bottleneck is visible rather than inferred.

### The verdict: it worked

The pipeline's own instrumentation, every progress line of a full shard:

```
  250,719 chunks   294/s   GPU busy 100%  starved 0%
  501,688 chunks   418/s   GPU busy 100%  starved 0%
1,000,869 chunks   607/s   GPU busy 100%  starved 0%
1,500,073 chunks   706/s   GPU busy 100%  starved 0%
2,250,320 chunks   818/s   GPU busy 100%  starved 0%
```

**GPU busy 100%, starved 0%, for the entire run.** Producers never starved the
GPU once. (The rising chunks/s is the cumulative average recovering from model
load and int8 calibration at startup; instantaneous rate is higher.)

### A hypothesis I had to discard

Mid-build I assumed the concurrent download was stealing disk bandwidth and
throttling the producers. I stopped the download to confirm — and throughput
**fell** to 248 chunks/s with the GPU still pinned at 99%.

Disk contention was never the limiter. The GPU was, the whole time. Sustained
rate with nothing else running is **632 chunks/s**, and `nvidia-smi` shows
2662/3090 MHz with no throttle reasons set — genuinely saturated.

Stating this because the wrong diagnosis was *plausible*: two jobs, one disk, an
obvious story. The measurement that killed it took ninety seconds.

### ⚠️ Do not compare chunks/s across the chunker fix

The obvious metric is misleading here, and it is worth being explicit about:

| | pre-fix | post-fix |
|---|---:|---:|
| chunks/s | 3,100 | 1,185 |
| mean chars per chunk | 178 | 1,209 |
| **text throughput** | **552 KB/s** | **1,432 KB/s** |

Chunks/s went *down* by 2.6× and real throughput went *up* by 2.6×, because each
post-fix chunk carries ~7× more text and therefore ~7× more GPU work. The old
number was inflated by embedding near-duplicate 178-character fragments.

**Measure the unit that matters.** For an indexing pipeline that is bytes of
corpus per second, or tokens per second — not chunks, whose definition your own
code controls.

---

## 🧵 The concurrency bug that cost a shard

The first threaded run embedded **3.25 M chunks successfully** and then aborted:

```
ERROR chunker: waited >900s for input
aborting: shard failed, manifest not committed
```

Completion was signalled by sentinels pushed through the **bounded** queues:

```python
try: self.raw_q.put(SENTINEL, timeout=30)
except queue.Full: pass          # <- silently drops the shutdown signal
```

Under sustained backpressure that `put` times out and the sentinel is dropped.
A chunker then waits forever for a message that no longer exists. The stall
detector caught it — working exactly as designed — but the shard was already
lost, because the manifest only commits at shard boundaries.

**Fix: never signal completion through a channel that can drop messages.**
`threading.Event` for "reader finished" plus a live-chunker counter. Consumers
exit on *"producer finished AND queue empty"*, checked in that order. An Event
cannot be lost to backpressure.

---

## ⏱️ What full-corpus indexing actually costs

Measured, with nothing else running:

```
632 chunks/s x 302 tokens = 191k tokens/s   (GPU-bound, 98-100%)
```

The 82 downloaded shards hold ~283 GB of raw text ≈ **71 billion tokens**.
At 191k tokens/s that is **103 hours**. Not a bug, not a tuning problem — that
is what a 5070 Ti does with BGE-small at this token volume.

| shards | chunks | binary index | time |
|---:|---:|---:|---:|
| 1 | 3.4 M | 0.16 GB | 1.6 h |
| 3 | 10.8 M | 0.52 GB | 4.7 h |
| 10 | 36 M | 1.73 GB | 15.8 h |
| 82 (all downloaded) | ~295 M | 14.2 GB | **103 h** |

The lever that would actually move this is batch size — measured on real
302-token chunks there is a **4x cliff**:

```
batch 128:  191 chunks/s
batch 256:  196 chunks/s
batch 512:  783 chunks/s   <- 4x
```

The build already uses 512. Worth testing 1024 before accepting the 103 h figure.

---

## 🛟 Crash safety: the resume-duplication bug

The first version had a second bug I only found by killing it: the data files are
append-only and the manifest is the commit point, but nothing reconciled them.
Killing a run mid-shard left ~15 M vectors on disk that the manifest didn't know
about — so resuming would re-process that shard and **append its vectors a second
time**. Silent duplicates, no error, nothing in the logs.

`truncate_uncommitted()` now rolls the three files back to the manifest's row
count on startup, and `save_manifest()` writes atomically via `os.replace` (a
torn manifest would make that truncation compute the wrong offset and corrupt the
index).

---

## 📁 What's in this project

```
07_rag_at_scale/
├── download_corpus.py        resumable 200 GB fetch (FineWeb-Edu), outside OneDrive
├── build_index.py            streaming chunk -> embed -> quantise -> append
├── validate_quantization.py  the recall/quality measurement above
├── bench_latency.py          latency vs index size, concurrency, 200 GB projection
├── scale/
│   ├── quantize.py           binary + int8 codecs, calibration, Hamming search
│   └── search.py             two-stage search over the memmapped index
└── techniques/
    ├── query_side.py         HyDE, multi-query, decomposition, routing
    ├── indexing.py           semantic chunking, contextual retrieval, late chunking, Matryoshka
    ├── late_interaction.py   ColBERT MaxSim scoring
    └── evaluation.py         RAGAS-style faithfulness / relevance / precision / recall
```

**Storage layout** (in `C:\genai-data`, deliberately outside OneDrive):

```
binary.u8        [n, 48]   uint8   loaded to RAM  -- stage 1 scans this
int8.i8          [n, 384]  int8    memmapped      -- stage 2 reads ~500 rows
coords.i64       [n, 4]    int64   (shard, row, char_start, char_end)
int8_calib.json  frozen global quantisation range
manifest.json    shard bookkeeping -- drives resume
```

`coords` is the design decision people miss: chunks store **32 bytes of
coordinates**, not their text. Text is read back from the parquet only for the
handful of chunks that reach an answer. Storing text alongside vectors would
re-materialise all 500 GB.

---

## 🚀 How to run it

```powershell
..\activate.ps1

# 1. download (resumable, ~6 h at 9.5 MB/s; safe to interrupt)
python download_corpus.py --target-gb 200
python download_corpus.py --status

# 2. index -- runs WHILE the download continues, processes whatever has landed
python build_index.py                 # or --max-shards 4 for a slice
python build_index.py --status

# 3. the core measurement (no index needed, ~2 min)
python validate_quantization.py --include-synthetic

# 4. latency, scaling curve and 200 GB projection
python bench_latency.py
```

Steps 1 and 2 are both resumable and can run concurrently — the indexer records
completed shards in `manifest.json` and skips them next time.

### Resumability is not optional at this size

The download **did** fail, at 157 GB / 78 shards, with a `ConnectionError` from
the Hub:

```
ConnectionError: Network error: Request middleware error:
  error sending request for url (.../xet-read-token/87f09149...)
```

Six hours of transfer is long enough that a transient network fault is not an
edge case, it is the expected path. Re-running picked up at **84% and skipped
every completed shard**, costing nothing.

Two details that made that work, both easy to get wrong:

- **Shard-level granularity.** `snapshot_download` verifies each file
  independently, so a partially-written shard is re-fetched and a complete one
  is not. A single 200 GB stream would have had to restart from zero.
- **A misleading exit code.** The failure was reported as `exit code 0`, because
  the shell pipe's status is what gets reported, not the Python process's. Check
  the actual output, not just the status.

---

## ▶️ Resuming after a shutdown

**Everything here survives a hard power-off.** Nothing needs cleaning up by hand.

### Current state as of the last session

| | |
|---|---|
| Corpus | **82 / 93 shards, 164 GB** at `C:\genai-data\hf` (download stopped deliberately) |
| Index | **0 chunks committed** — a build was mid-shard-1 when the machine went down |
| Uncommitted rows | ~2.4 M on disk, rolled back automatically on next run |
| Everything else | measured, committed, and independent of the index |

### To carry on

```powershell
cd 07_rag_at_scale
..ctivate.ps1

python build_index.py --status          # confirm what is committed
python build_index.py --max-shards 3    # ~4.7 h, rolls back partial work first
```

The first thing `build_index.py` prints is how many uncommitted rows it
discarded. That is the crash-safety machinery doing its job: the data files are
append-only, the manifest is the commit point, and anything past the last
committed shard is truncated before new work starts.

To finish the corpus download (11 shards left) — it self-retries now:

```powershell
python download_corpus.py --target-gb 200
```

### What does NOT need the index

The result that carries this project is already measured and committed:

```powershell
python validate_quantization.py --include-synthetic   # ~2 min, no index needed
```

`quantization_results.json` is in the repo. The 0.985 recall / 1.0000 quality /
32x reduction numbers do not improve with more shards — a bigger index buys a
bigger number in the README, not a stronger claim.

## 📉 Measured latency — and why the flat scan loses

One committed shard: **3,401,375 chunks**, binary index 0.163 GB, int8 1.31 GB,
float32 avoided 5.22 GB (32x). Indexed in 50.3 min at 1,127 chunks/s with
**GPU busy 100%, starved 0%** throughout.

| vectors | binary ms | rescore ms | p50 ms | QPS/thread |
|---:|---:|---:|---:|---:|
| 100,000 | 8.9 | 0.24 | 9.1 | 109 |
| 500,000 | 44.6 | 0.33 | 44.9 | 22.3 |
| 1,000,000 | 88.5 | 0.34 | 88.8 | 11.3 |
| 2,000,000 | 180.2 | 0.39 | 180.7 | 5.5 |
| **3,401,375** | **306.3** | **0.38** | **306.7** | **3.3** |

### The cascade is validated. The flat scan is not.

**Rescore cost is flat.** 0.38 ms at 3.4 M vectors, identical to 0.24 ms at
100 k, and still under 3 ms at 2000 candidates. It depends only on candidate
depth, never on corpus size — exactly the design intent. Reading ~500 rows from a
1.31 GB memmap costs essentially nothing.

**The binary scan is ruthlessly O(n)**: ~90 ms per million vectors, and it is
306 ms of a 307 ms query. Extrapolating to the corrected corpus size (~316 M
chunks):

```
306 ms x 93  ->  ~28 SECONDS per query
```

**So this architecture does not reach 200 GB, and the benchmark is what proves
it.** A flat scan is right up to roughly 10 M vectors and wrong past it. This is
the measured argument for IVF or HNSW: partition the space so a query touches a
fraction of it. The trade is that recall stops being a guarantee and becomes a
tunable (`nprobe`, `efSearch`) — a new error budget you then have to measure,
which is exactly what `validate_quantization.py` is for.

Reporting this rather than quietly benchmarking 100 k vectors and claiming 9 ms
is the whole point of building the harness.

### Concurrency: memory-bandwidth bound

| workers | QPS | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|
| 1 | 3.3 | 306 | 317 | 321 |
| 4 | 11.8 | 336 | 359 | 376 |
| 8 | 19.9 | 393 | 421 | 435 |

8 threads deliver **6x the QPS, not 8x**, and p50 degrades 306 → 393 ms. numpy
releases the GIL inside the XOR/popcount so threads genuinely scale, but they
contend for the same memory bandwidth — which is the real ceiling, and another
reason the answer is to touch *less* of the index rather than scan it faster.

### The surprise: text fetch dominates end-to-end

| stage | ms |
|---|---:|
| query embedding | 8 |
| search (binary + rescore) | 307 |
| **fetch text for 5 hits** | **~1,600–2,200** |
| **end-to-end** | **1,894–2,553** |

Storing 32-byte coordinates instead of chunk text saves ~344 GB — but
`attach_text` has to scan parquet row-batches to locate rows, so retrieving five
snippets costs longer than the entire search. The storage win is real and so is
the retrieval cost; a production fix needs a row-offset index into each parquet
file, or the chunk text in a key-value store beside the vectors.

Worth stating plainly: this cost was invisible until the end-to-end number was
measured separately from the search number.

---

## 🧪 Modern techniques

Each module is independently runnable and documents **when the technique is wrong**,
which is usually the more useful half.

### Query-side — `techniques/query_side.py`

| technique | what it fixes | measured cost |
|---|---|---|
| **routing** | picks a strategy per query instead of one for all | 0.05–0.1 s |
| **HyDE** | question↔passage vocabulary gap | ~0.6 s |
| **multi-query** | unlucky phrasing missing the answer | ~1 s |
| **decomposition** | multi-hop questions matching neither half | ~1 s |

Live output from `route()`:

```
"what does TLM-330 mean"                        -> lookup      (BM25-weighted, no HyDE)
"how does the dispatch auction decide a winner" -> conceptual  (dense-weighted, HyDE on)
"how does the bid formula relate to SEV2"       -> comparison  (decompose first)
```

**HyDE's failure mode, observed live.** Asked how the auction picks a winner, it
generated *"the robot with the highest bid score"* — Atlas awards the **lowest**
bid. The hypothetical is confidently wrong, which is exactly why the router turns
HyDE **off** for identifier lookups: for `TLM-330` a hallucinated passage drags
retrieval toward a code that doesn't exist. HyDE helps conceptual questions
because it only needs the right *shape*, not the right facts.

### Indexing-side — `techniques/indexing.py`

- **Semantic chunking** — cut where the topic changes, using a *percentile* of
  the observed sentence distances rather than an absolute threshold (absolute
  thresholds don't transfer between document styles). Costs 3–4× the embedding
  work, which is why the 200 GB path uses structural chunking.
- **Contextual retrieval** (Anthropic) — prepend an LLM-written sentence situating
  each chunk. ~35% fewer retrieval failures, but one LLM call **per chunk**: at
  316 M chunks that is months of compute. Project 01's heading breadcrumb is the
  free structural approximation.
- **Late chunking** — embed the whole document, *then* pool per chunk, so each
  chunk vector carries context it never had in isolation. Usually *cheaper* than
  normal chunking (one pass per document). Needs a long-context embedding model.
- **Matryoshka** — truncate 768→256→64 dims and renormalise. Only valid on a model
  trained with the MRL objective; slicing a normal embedder destroys it.

### Late interaction — `techniques/late_interaction.py`

ColBERT MaxSim: one vector per **token**, scored as
`Σ_i max_j (q_i · d_j)`. Working output:

```
score  6.800  "Barcode reads below 0.92 confidence are re-attempted..."   <- correct
      'confidence' -> 'confidence'  0.832
score  5.561  "nw-barcode-ocr-v2 degrades on shrink-wrapped pallets..."
score  3.792  "Vision frames are blurred at the edge..."
```

Interpretable, and it ranks correctly. **But it cannot be the primary index:**

| | chunk-level binary | token-level ColBERT (2-bit) |
|---|---:|---:|
| 316 M chunks | **15.2 GB** | **3,540 GB** |

700× the storage. Its place in a 200 GB system is as a *second-stage reranker*
over the few hundred candidates the binary index already returned.

### Evaluation — `techniques/evaluation.py`

RAGAS-style: faithfulness, answer relevance, context precision, context recall.
The diagnostic value is in the **pairs** — low faithfulness with *high* context
recall means the model ignored good context; with *low* recall it means retrieval
starved it and the model guessed. Same symptom, opposite fix.

An LLM judge is an instrument with its own error rate: biased toward verbose
answers, capped by its own competence. Prefer deterministic metrics where they
exist — project 01's mechanical citation verification is free and cannot
hallucinate.

---

## 🖥️ Tech stack

- **Corpus:** FineWeb-Edu `sample/100BT` (286 GB available; 200 GB targeted)
- **Embeddings:** BGE-small-en-v1.5, fp16, batch 512 — **4,900 chunks/s** on GPU
- **Quantisation:** 1-bit binary (packed) + int8 with frozen global percentile calibration
- **Search:** numpy Hamming via 256-entry popcount LUT, then memmapped int8 rescore
- **Validated on:** RTX 5070 Ti 16 GB, 61.6 GB RAM, 1.6 TB free, Python 3.12

---

## ❓ FAQ

**Why not FAISS / a vector database?**
FAISS is in `01_rag_local` where it belongs. Here the point is that the *storage
format* is the design, and writing it by hand makes the memory arithmetic
visible. In production you would use FAISS IVF-PQ or Qdrant — both implement
exactly this cascade, and knowing why is what lets you configure them.

**When does a flat binary scan stop working?**
It is O(n): every query touches all 23 GB. `bench_latency.py` measures the curve
and projects it. Past a few hundred million vectors you need IVF or HNSW to touch
a *fraction* of the index — at which point recall becomes a tunable
(`nprobe`/`efSearch`) rather than a guarantee, and you have a new error budget to
measure.

**Why store int8 on disk rather than in RAM?**
It is only read for ~500 rows per query — 192 KB. The OS page cache handles that
better than we would, and the RAM is better spent on the binary index that gets
scanned in full.

**Could you skip binary and just use int8?**
Yes, and recall would be near-perfect — but int8 is 184 GB, which does not fit in
61 GB of RAM. The binary stage exists precisely so the full scan happens over
something that fits.

**Is 500 candidates right for my corpus?**
Re-run `validate_quantization.py` on your own embeddings. The knee moves with
embedding model, dimensionality and how clustered your data is.

---

## Related projects

- **[01_rag_local](../01_rag_local/)** — the same pipeline at 30 chunks, with
  hybrid retrieval, reranking and grounded generation
- **[05_mcp_server](../05_mcp_server/)** — exposes retrieval as authenticated tools
- **[06_local_gpu_inference](../06_local_gpu_inference/)** — the same
  "know which resource you're short of" discipline, applied to inference
