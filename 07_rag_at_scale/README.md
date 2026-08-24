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
| chunks from 93 shards | 143 M | **~335 M** |
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

### Where the bottleneck moved

Sampling GPU utilisation every 2 s during a threaded build, *while the 200 GB
download was still running*:

```
100%  16%  100%  4%  6%  33%  58%  100%  100%  100%  38%  100%     (mean ~63%)
```

The GPU now *reaches* 100% — it never did single-threaded — but it still starves
in bursts. The producers are **disk-I/O bound**, competing with the concurrent
download writing 129 GB. That is contention between two jobs I chose to run at
once, not a flaw in the pipeline: indexing reads 2 GB parquet files while the
downloader writes at ~9.5 MB/s to the same volume.

The honest reading: threading fixed the *GIL/serialisation* bottleneck and
exposed an *I/O* one. Run the download to completion first and the same code
should sit much closer to the GPU ceiling.

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
  480 M chunks that is months of compute. Project 01's heading breadcrumb is the
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
| 480 M chunks | **7.4 GB** | **5,208 GB** |

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
