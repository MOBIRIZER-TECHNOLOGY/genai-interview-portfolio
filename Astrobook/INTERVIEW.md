# 🎤 Interview notes — end-to-end: building and deploying a custom model

Every number here was measured on this machine and can be re-derived from
`build/`, `runs/*/run_metrics.json`, `runs/qlora-nf4/validation.json` and
`build/grounding.json`. Re-run before an interview so you are quoting live
results, not remembered ones.

The other `INTERVIEW.md` files in this repo go deep on one technique each. This
one is the **lifecycle** — the question "walk me through taking a custom model
from raw data to production," which is the question a senior role actually turns
on. Depth on LoRA itself lives in `02_lora_text/INTERVIEW.md`; depth on
retrieval in `01_rag_local/` and `07_rag_at_scale/`.

**Read this first, and say it out loud in the interview:** this system is *not
deployed*. It is a finished research artifact with a measured deployment plan and
nine explicit launch gates, none of which are met. Claiming a production
deployment you do not have is the fastest way to lose a senior interview, because
the follow-up questions are all operational. Saying "here is what I built, here
is what I measured, here is exactly what is between it and production, and here
is why I have not shipped it" is a *stronger* answer than most real deployments
produce.

---

## The 90-second pitch

> "I took 52 PDFs of Jyotisha literature — about 7,400 pages, no labels, no
> existing dataset — and turned them into a domain-adapted 4B model with
> retrieval, end to end on one consumer GPU.
>
> The pipeline is twelve stages: extract and chunk the PDFs, generate 14,517
> instruction pairs with a local teacher model, split by book so nothing leaks,
> train LoRA and QLoRA as a controlled A/B, measure both against the untuned
> base, merge the winner into a standalone 8 GB model, build a hybrid retrieval
> index over the same chunks, and prove with a needle test that the model is
> actually reading the retrieved passages.
>
> The measured result: the adapter cuts held-out perplexity from 16.91 to 4.42
> and eliminates markdown preamble entirely, 70% to 0%. QLoRA saved 54.7% of VRAM
> for a 14.5% speed cost with an eval-loss difference of 0.0107 — inside noise.
>
> The part I am most confident about is what it *doesn't* do. Fine-tuning
> installed the voice and installed no facts, exactly as predicted, so retrieval
> supplies the facts. And when I reviewed my own retrieval for deployment I found
> that 68% of the corpus was never reaching the embedder — the chunks were sized
> for training and the embedder caps at 512 tokens. That bug was invisible in
> every metric I had, because the lexical half of the hybrid index was covering
> for it."

That last paragraph is the one that gets you hired. Lead with the result, close
with the defect you found in your own work and how you found it.

---

## The whiteboard question: "walk me through it end to end"

Draw this. Do not narrate it as a list — the point is that each stage's output
constrains the next, and most real failures live in the seams, not the boxes.

```
  52 PDFs ──▶ extract ──▶ generate ──▶ split ──▶ train ──▶ evaluate
              1,904        14,517       by        LoRA vs    vs untuned
              chunks       pairs        book      QLoRA      base
                 │                                              │
                 │                                              ▼
                 └────────────▶ index ──▶ retrieve ──▶ serve ◀── merge
                                hybrid     RRF          vLLM     8.04 GB
```

The single most important thing on this diagram is the line running from
**extract** to **index**. One extraction feeds both training and retrieval — and
that shared input is where my worst bug lived, because the two consumers want
different chunk sizes and only one of them said so.

### What can go wrong at each stage, in one line each

| Stage | The silent failure |
|---|---|
| Extract | Chunks slice mid-sentence and every downstream artifact is subtly worse. Mine was **54% pinned at the size cap** before I fixed the splitter. |
| Generate | The teacher model rambles and never closes its JSON, so you get zero usable pairs and blame the prompt. |
| Split | Row-level split leaks translator idiom across the boundary; your eval reports a number the model did not earn. |
| Train | The adapter is not attached, or the model silently pages to system RAM and trains 37× slower without ever raising OOM. |
| Evaluate | You measure loss, conclude "it works", and ship something confidently wrong on facts. |
| Merge | You trust `merge_and_unload()` and never check that the weights actually moved. |
| Index | The embedder truncates and you never see it, because BM25 covers the gap. |
| Serve | You serve in the quantization you *trained* in and give up 20% of throughput for memory you have. |

If you can speak to all eight of those, you have demonstrably done this before.

---

## Stage 1 — data: "where does training data come from when you have none?"

This is the question most candidates handle worst, because the tutorials all
start from a HuggingFace dataset that already exists.

**Answer: you synthesise it from the corpus with a teacher model, and you
constrain the teacher hard.** My generation prompt does four things that matter:

1. **Grounds every pair in one passage** — "answerable ENTIRELY from that
   passage, never use outside knowledge."
2. **Spreads across four question types** — definitional, rule-application,
   comparative, explanatory — so the adapter does not learn one shape.
3. **Forces stand-alone questions.** This is the subtle one. The obvious failure
   is a question like *"What does the passage say about the 3rd house?"* At
   training time there is no passage — so the model learns to answer questions
   that reference context it will never see. The prompt gives explicit BAD/GOOD
   examples.
4. **Provides an escape hatch** — a `grounded: false` flag for passages too
   tabular or too garbled to support a real question. A short honest batch beats
   a padded one, and 407 pairs were dropped by QC.

### "How did you choose the teacher model?"

**By measurement, not preference** — and the measurement inverted the obvious
choice:

| Teacher | Raw tok/s | Time per chunk (8 pairs) | Valid pairs |
|---|---:|---:|---:|
| `qwen2.5:7b` | 138 | 17.7 s (hit the token cap) | **0** |
| `qwen3:14b` | 75 | **14.6 s** | **8** |

The 7B is nearly twice as fast per token and produced **nothing usable** — it
rambles and never closes its JSON. The 14B is slower per token and faster in
practice because it is concise and terminates. *Raw throughput is not throughput.*

The related answer, if they ask about scaling generation: I tried
`OLLAMA_NUM_PARALLEL=4` and throughput **fell** from 4.2 to 1.1 chunks/min. At
97% VRAM the KV-cache slots thrash. I reverted it. Full corpus took 456.9 minutes,
resumable, appending per chunk.

---

## Stage 2 — the split: the question that separates seniors from juniors

**"How do you split a corpus like this?"**

Not by row. Chunks from one book share vocabulary, translator idiom, and
phrasing. A random row split puts question A from *Saravali* in train and
question B from the same page in test, and your eval measures memorisation.

I split **by whole book**: 38 books train, 5 books held out entirely — 1 of them
for val, the other 4 for test. `test_split_has_no_book_leakage` asserts zero
overlap, and I re-verified it during this review: train ∩ test = 0 books.

### The follow-up most candidates have no answer to

**"Fine — but is your validation set independent of your test set?"**

Mine was not, and I found it auditing my own split. The first version of
`03_split.py` held out 5 whole books from train — correct — and then split *each
held-out book's rows* 50/50 into val and test. Train was clean. Val and test
shared **232 of their 235 source chunks**: different questions, generated by the
same teacher from the same passages.

Why it matters: any decision made on val — checkpoint selection, early stopping,
a hyperparameter sweep — is then effectively made on test, and the "held-out"
number stops being independent of anything. In this run the damage was zero,
because I trained a fixed 2 epochs and took the final checkpoint without ever
selecting on val. But the method was wrong, and the moment anyone added early
stopping it would have started quietly inflating the headline.

**The fix, and why it was cheap.** Deal the held-out *books* between val and test
instead of splitting each book's rows — greedy largest-first into two bins,
balancing row counts, with the bin spanning more books becoming test because test
carries the headline claims. Val is now one book (915 pairs), test is four (907
pairs), and they share zero chunks. Crucially the **train set is byte-identical
before and after** — same 12,695 examples, same set hash — so the already-trained
adapter stays valid and nothing needed retraining.

Two details worth saying out loud, because they are what an interviewer is
actually probing for:

- **The suite had two leakage tests and neither caught it.** `test_split_has_no_book_leakage`
  checked train↔test, and another checked train↔val. Nothing checked val↔test.
  The new `test_split_val_and_test_use_different_books` asserts both book- and
  chunk-level independence, and I confirmed it *fails* on the old split before
  regenerating — a test you have not seen fail is not evidence.
- **I had to check the fix did not invalidate the model.** Any wider hold-out
  would have moved books the adapter had already trained on into the eval sets.
  Keeping the partition inside the books already held out is what made this a
  data fix rather than a retrain.

"Val and test must be independent *of each other*, not just of train" is a
distinction most candidates never make, and finding it in your own work is worth
more than never having had the bug.

---

## Stage 3 — training: the parts specific to doing it for real

Rank 32 (not the usual 8 — teaching a domain idiom, not a response format),
`alpha = 2r = 64`, **all seven linear modules** (target module choice matters more
than rank; q/v-only is the common under-configuration), seq len 1024, lr 1e-4,
effective batch 16, 2 epochs, `assistant_only_loss=True`.

For the mechanism — why `B` is zero-initialised, why the scale is `α/r`, what NF4
and double quantisation are — see `02_lora_text/INTERVIEW.md`. Two things here are
specific to the end-to-end story:

### "How do you know your comparison is valid?"

**Pre-registered predictions.** Before any GPU time, I wrote down five predictions
with pass thresholds, and `06_compare.py` checks them mechanically so I cannot
rationalise afterwards. Three held, two missed:

| Prediction | Result | |
|---|---|---|
| VRAM saving 50–60% | 54.7% | ✅ |
| Speed cost 25–40% | **14.5%** | ❌ below the band |
| Eval-loss Δ < 0.15 | 0.0107 | ✅ |
| Adapter size identical | **50% different** | ❌ |
| Trainable params identical | 0 diff | ✅ |

And the detail to tell: **my checker had a bug that hid a miss.** It originally
tested a one-sided ceiling (`≤ 60%`), so 14.5% reported HOLDS despite missing the
stated band by half. A prediction that is wrong in a direction you *like* is still
wrong, and a scorecard lenient enough to pass it is not measuring anything. I
fixed the checker to enforce two-sided bands.

The adapter-size miss is a good control-variable story: 264 MB vs 132 MB, an exact
2:1 ratio — same 504 tensors, same 66,060,288 parameters, different storage dtype
(peft writes fp32; the QLoRA path leaves them in bf16 compute dtype). Anyone
comparing file sizes would conclude "QLoRA produces smaller adapters." It does
not. That is precisely what a control exists to catch.

### "What was the worst bug you hit?"

**A model that trained fine, converged fine, and was 37× too slow.**

Phase 6 put Qwen3-8B on a 16 GB card. §5 predicted bf16 would OOM. It did not —
on Windows, WDDM lets the GPU oversubscribe VRAM by **paging to system RAM over
PCIe**. The model loaded, reported success, showed a moving progress bar and a GPU
pinned at 100%, and ran at **177.7 s/step**. In 4-bit the same model runs at
**4.74 s/step**.

| | 8B bf16 | 8B QLoRA nf4 |
|---|---:|---:|
| Spilled to system RAM | ~11 GB | 0 |
| sec/step | 177.7 | **4.74** |
| 100 steps | 4.9 hours | **7.9 min** |

**An OOM is a better failure than this**, because an OOM is immediate and
unambiguous. This one is only detectable by noticing that a 500-step run reports
an ETA of 24 hours. The lesson, and this is the sentence to say: *do not use OOM
as your fit test — use step time and the spill counter.*

It is also the honest case **for** QLoRA. At 4B it is a trade: half the memory for
15% more time. At 8B it is not a trade at all — it is the difference between a
model you can train and one you cannot.

---

## Stage 4 — evaluation: "how do you know it works?"

Three layers, and being able to name what each one does *not* prove is the whole
answer.

**Layer 1 — held-out loss.** Same loaded weights, adapter toggled with peft's
`disable_adapter()`, so nothing but the adapter can differ. n = 250 on the 5
held-out books:

| | BASE | TUNED | |
|---|---:|---:|---|
| Assistant-token NLL | 2.8277 | **1.4871** | −1.3407 |
| Perplexity | 16.91 | **4.42** | **3.8× better** |

Proves the fine-tuning unambiguously worked. Proves **nothing about facts.**

**Layer 2 — mechanical style probes.** Regex counts, no judge, no API key, n = 20:

| | BASE | TUNED |
|---|---:|---:|
| Markdown headers | 70% | **0%** |
| Hedging preamble | 45% | **0%** |
| Doctrinal framing | 45% | **60%** |
| Sanskrit terminology | 70% | **35%** ← got *worse* |
| Mean answer length | 120 words | 82 words |

Format training succeeded completely — both bad-behaviour probes read exactly
zero. And one metric moved the wrong way. My working hypothesis is that the base
sprays Sanskrit terms as padding across 120 words while the adapter uses one only
where needed across 82 — but I recorded it as **an open question, not an
explanation**, because it is equally possible the adapter lost vocabulary. Say that out loud. "I don't know yet, here are the two
candidate explanations and the experiment that would separate them" beats a
confident story every time.

**Layer 3 — what is still unmeasured.** Factual accuracy. `07_demo.py` caught the
model answering *"the 2nd, 5th, 7th, 8th, 10th, or 11th house lords"* where
Phaladeepika says *"the ascendant lord, 7th lord, 5th lord, Jupiter, the planet
aspecting the 5th, the planet occupying the 5th."* Fluent, correctly shaped,
confidently wrong.

**Low loss means the model predicts corpus-style text well. It does not mean the
facts are right. Those are different quantities and only the first is
demonstrated.** That sentence is the single most useful thing in this document.

---

## Stage 5 — merge: turning an adapter into a shippable artifact

**"You trained against a 4-bit base. What do you ship?"**

You cannot merge into NF4. A 4-bit weight is a quantised integer plus a shared
scale; adding a bf16 delta to it is not a defined operation. So merging requires
loading the base in **full precision**, and the output is bf16 at full size —
8.04 GB from a 132 MB adapter.

This surprises people: QLoRA lets you *train* in 4 bits, and the merged artifact
still comes out full size. Re-quantise afterwards if you want it small, and note
that re-quantisation happens **after** training, so the adapter never adapted to
it.

**And verify the merge rather than trusting it.** `11_merge.py` snapshots real
layers before merging and checks `W_new == W + (B @ A) · (α/r)` numerically on the
actual 66M-parameter adapter, asserting both that the error is bf16 rounding *and*
that the weights actually moved. A merge that silently no-ops produces a model
that loads, generates, and is just the base — the exact failure `moved > 0`
catches.

**The deployment counter-argument, worth raising yourself:** merging is not
obviously the right call. vLLM can hot-swap LoRA adapters against a shared base,
so keeping the 132 MB adapter as the shipped artifact means one base model serving
many tenants or many domain adapters. Merge when you have one model and want the
simplest possible serving path; keep the adapter when you expect more than one.

---

## Stage 6 — retrieval: the seam, and the bug that lived in it

**"Your model has the voice but not the facts. Now what?"**

LoRA for voice, RAG for facts — and they compose in exactly one direction. The
adapter decides *how* to answer; the retrieved passage decides *what is true*.
`10_rag.py` runs the four-way ablation so the contribution of each is visible
rather than asserted: base alone rambles and invents sources; base + RAG has the
facts and the wrong register; adapter alone has the register and invented facts;
adapter + RAG is the product.

### The retrieval design, and why hybrid

Dense embeddings (`bge-small-en-v1.5`, 384-dim) fused with BM25 via **Reciprocal
Rank Fusion**. Both, because each covers the other's blind spot: this corpus is
full of rare Sanskrit terms — *Arudha*, *Prishtodaya*, *Ashtakavarga* — that
embedding models were never trained on and blur into nearby concepts, while BM25
matches them exactly on high IDF. Conversely BM25 fails on paraphrase ("what
happens when Saturn is in the 7th" against a passage reading "Sani in Kalatra
Bhava"), which is where dense wins.

RRF scores by `1/(k + rank)` from each list rather than by raw score — which
sidesteps having to normalise a cosine similarity against a BM25 score, two
quantities on entirely different scales.

### The bug — tell this story in full

Reviewing my own system for deployment, I measured what share of the corpus was
actually reaching the embedder:

```
bge-small-en-v1.5 max_seq_length          512
chunk tokens          p50 1,658   p95 2,142   max 4,400
chunks over the limit         1,878 / 1,904  =  98.6%
corpus tokens embedded                31.7%   (68.3% dropped)
```

**Two-thirds of the corpus was invisible to the dense retriever.** The cause is a
seam, not a mistake in either component: `01_extract.py` packs chunks toward 1,800
tokens because that is the right size for *generating training pairs*, and
`09_index.py` embeds those same chunks with a model that hard-caps at 512.

Then it got worse downstream. `chat.py --rag` truncates each retrieved passage to
its first 900 characters of a median 6,700-character chunk. Measured across all
907 held-out test rows, resolved to their true source chunk:

```
share of the reference answer's content words present in ...
  the whole source chunk           59.3%
  first 2,600 chars (10_rag.py)    40.7%
  first   900 chars (chat.py)      20.8%
```

So the ceiling on a grounded answer was set by the slice, not by the retriever.

**Why it stayed hidden — this is the interesting part.** BM25 reads the whole
chunk, so hybrid fusion kept returning the right book (17/20 on the ablation) and
retrieval *looked* healthy. A partial failure in one half of a redundant system
was masked by the other half. That is the general lesson: redundancy improves
robustness and destroys observability, so you have to measure each component
alone, not just the ensemble.

**The fix** is a parameter, not a rewrite: `pack()` already receives clean
verse/paragraph/sentence units, so emit a second `retrieval_chunks.jsonl` at ~350
tokens with overlap from the same `split_units()` output, and index that.
Training keeps the 1,800-token chunks.

### "How do you prove the model is actually using retrieval?"

Not with a judge model and not with vibes. **A needle test.** Insert a fact that
cannot exist in Jyotisha literature or in any pretraining corpus — an invented
graha named *Zorvaxa*, a sage named *Quenlith*, a yoga requiring exactly 847
degrees of separation — bury it in the middle of the retrieved passages, and ask
about it.

Result: **3/3 needles reported back, 0 control leaks.** The control is what makes
it valid: the same questions asked with no passages supplied, confirming the model
cannot produce the proof tokens from priors.

And a design detail worth mentioning, because it is a mistake I made and caught:
the proof token must not appear in the question. An earlier version checked
whether the model said "Zorvaxa" — a word the question itself contained, so the
model could echo it without reading anything. The control correctly flagged the
test as invalid. Now the question asks what Zorvaxa *grants* and the proof token
is "glassblowing."

Lexical grounding on real questions, n = 20: **47.9%** of the answer's content
words appear in the retrieved passages, against **13.5%** for the same model
answering the same questions with no passages — a **+34.4 point lift**. Retrieval
finds the right book on 17 of 20.

**And a sampling lesson worth telling.** That measurement first ran at n = 5 and
reported +13.1 points; a later run at n = 4 reported +48.4. Both bracket the n = 20
answer of +34.4 so wildly that neither was worth quoting — and the reason the
sample was that small is that the sampler took one question per held-out book, so
the book count silently capped it. Two of my scripts had that same defect, and
both only became visible when a split change moved the book count. *If a sample
size is derived from your data's shape rather than stated, it will change under
you without saying so.*

---

## Stage 7 — serving: the decisions and the measurements behind them

### "You trained in 4-bit. Do you serve in 4-bit?"

**No, and I measured why.** Same merged model, same prompts, greedy decode:

| | tok/s | peak VRAM |
|---|---:|---:|
| NF4 + double quant | 26.3 | **2.76 GB** |
| bf16 | **33.0** | 8.10 GB |

bitsandbytes dequantises on every forward pass, and batch-1 decode is exactly
where that hurts. NF4 was the correct call at *training* time, where it bought
54.7% of VRAM and made an 8B trainable at all. At serving time on a card with room
to spare it costs 20% of throughput to save memory nobody needs.

**Quantization is not a property of the model, it is a property of a phase.**

Both numbers are single-stream through an unbatched HF `generate` loop, so neither
is a serving figure — and saying that is the point. A real runtime with continuous
batching and paged attention changes the shape entirely.

### "What runtime, and why?"

vLLM, serving the merged bf16 model behind an OpenAI-compatible endpoint, with
the retriever in the gateway process. The reasoning:

- **Continuous batching** is the thing HF `generate` cannot do. Per-request
  throughput barely moves; aggregate throughput moves by an order of magnitude.
- **PagedAttention** keeps KV cache fragmentation from capping concurrency.
- **Retrieval stays in-process** — 13 ms warm on CPU, 979 MB RSS. A network hop to
  a vector database would cost more than it returns.

### "Why not a vector database?"

1,904 chunks × 384 dimensions is a NumPy matrix multiply. Measured end to end,
including the query encode, it is **13 ms on CPU**. Pinecone or pgvector would add
a network hop, an operational dependency, and a bill, to replace a line of NumPy
that is not the bottleneck.

Say where the answer flips: past roughly a million chunks, when the index no
longer fits comfortably in RAM, or when you need filtered search and multi-tenant
isolation. `07_rag_at_scale/` in this repo is where I took that seriously.

Being able to argue *against* infrastructure is a senior signal. Most candidates
only demonstrate they know the tool exists.

---

## Stage 8 — deployment and operations

### The artifacts, and where each one lives

| Artifact | Size | Where it belongs |
|---|---:|---|
| Merged model | 8.04 GB | Private model registry with a SHA and a model card |
| Adapter | 132 MB | Same — it is the cheaper artifact if you go multi-adapter |
| Embeddings + BM25 | 14 MB | Versioned build artifact, mounted read-only |
| Chunks | 12 MB | Same version stamp as the index |

**The rule that matters: version the index with the corpus hash *and* the
embedding model name.** An index built from different chunks must never be able to
pair silently with the wrong code — that failure produces plausible answers from
the wrong passages, which is worse than a crash.

Model and index are **mounted, not baked into the image**, so an 8 GB layer is not
rebuilt on every code change.

### "What do you monitor?"

Not just latency and error rate. For a RAG system the operationally important
signals are:

- **Retrieved chunk ids and scores, logged per answer.** Without this you cannot
  audit a bad answer after the fact, and "the model said something wrong" is
  unactionable.
- **Abstention rate** — how often it says "the excerpts don't cover this." A
  sudden drop means it started filling gaps from memory.
- **Score distribution drift** on the top-k. Falling RRF scores mean the questions
  have moved away from the corpus.
- **Token throughput and queue depth**, which set your timeout, rather than a
  guessed timeout setting your behaviour.

### "How would you roll back?"

Model version and index version are separate, immutable, and independently
pinned — so a bad retrieval rebuild rolls back without touching the model, and a
bad adapter rolls back without re-embedding 1,904 chunks. Coupling those two is
the mistake that turns a five-minute rollback into a rebuild.

### The gates I set, and have not met

Nine, each a number rather than a judgement: embedded-token share above 95%
(now 31.7%), answer-word survival above 55% (now 20.8%), faithfulness judged on
150 held-out questions (never run), abstention measured (unmeasured),
personal-prediction refusal verified (unmeasured), corpus licence posture decided
(open), weights out of the git tree (12 GB currently exposed), tests green in CI
(pytest not currently installed), load-tested at target concurrency (single-stream
only).

**Having written gates you have not met is a better answer than a vague claim that
it is production-ready.** It shows you know what production means.

---

## The landmine questions

### "What about the copyright of your training data?"

Do not fumble this one — it is increasingly the first question a serious company
asks, and "I didn't think about it" ends the conversation.

My corpus mixes public-domain Sanskrit doctrine with clearly in-copyright modern
work: named authors' books, two commercial software manuals, and — the subtle
one — the modern English **translations** of classical texts. The underlying verse
is centuries old; the translation is a fresh copyrightable work with its own term.

Two distinct exposures: the service returns **literal excerpts**, and the adapter
was **trained on Q&A derived from** those books. Private study on your own machine
is one posture; a public endpoint reproducing paragraphs of a living translator's
work is another.

The postures, cheapest first: keep it internal and authenticated; split the index
by licence and quote only the public-domain subset while citing the rest by name;
drop excerpt display entirely and return answer-plus-citation; or ship the
pipeline and let users bring their own corpus. It is a decision that has to be
made *before* building, because it changes what you build — and I encode it as a
per-chunk field so the serving path enforces it rather than relying on memory.

### "What are the safety considerations?"

Domain-specific and concrete. An astrology service gets asked about illness, death
timing, pregnancy, money, and marriage. The system prompt constrains the model to
describe what texts assert and forbids personal prediction — *"the text holds that
Mars in the 4th indicates X"*, never *"Mars in your 4th will cause X"* — and that
constraint is enforced in the **training data itself**, not only at inference, so
every one of the 14,517 pairs is framed doctrinally.

Then the honest part: **nothing yet measures compliance.** An adversarial eval on
real-shaped queries is one of my nine gates. A guardrail you have not measured is
a hope.

### "How much does it cost to run?"

The merged model is 8.04 GB in bf16, so anything with 16 GB of VRAM hosts it with
room for KV cache. An L4 is roughly $0.70/hr always-on and is the cheapest GPU
that serves a 4B well under vLLM; serverless per-second GPU is better for spiky
traffic, with cold start as the tax. Training cost was one consumer GPU and 88
minutes of wall clock for the 4B arms.

Pricing a 4B deployment like a 70B one is a common and expensive mistake.

### "What would you do differently?"

Four things, in order:

1. **Size chunks per consumer, from the start.** One extraction, two chunk files.
   Assuming one artifact serves both training and retrieval cost me 68% of my
   corpus for weeks.
2. **Hold out different books for val and test**, not different rows of the same
   books.
3. **Measure each component of the hybrid alone**, not just the ensemble — the
   redundancy is exactly what hid the failure.
4. **Decide the licence posture before extraction**, because it determines whether
   a book belongs in the corpus at all.

---

## Rapid-fire numbers to memorise

| | |
|---|---|
| Corpus | 52 PDFs, 44 usable, ~7,400 pages, ~2.97M tokens |
| Chunks | 1,904 (target 1,800 tokens) |
| Pairs | 14,517 generated, 407 QC-rejected |
| Splits | 12,695 train / 915 val / 907 test — 38 books train, 1 val book, 4 test books |
| Base | Qwen3-4B-Instruct-2507 |
| LoRA | r=32, α=64, all 7 linear modules, 66,060,288 trainable |
| QLoRA vs LoRA | −54.7% VRAM, +14.5% time, +0.0107 eval loss |
| 8B on 16 GB | 177.7 s/step bf16 (11 GB spilled) vs 4.74 s/step NF4 = **37×** |
| Validation | perplexity 16.91 → **4.42** (3.8×), markdown 75% → **0%** |
| Merged model | 8.04 GB, from a 132 MB adapter |
| Retrieval | 13 ms CPU warm, hybrid dense + BM25 + RRF |
| Grounding | 3/3 needles, 0 control leaks, +34.4 pt lexical lift (n=20), right book 17/20 |
| Serving | 26.3 tok/s NF4 @ 2.76 GB · 33.0 tok/s bf16 @ 8.10 GB |
| The bug | 98.6% of chunks truncated at 512; **68.3% of corpus never embedded** |

---

## Three stories to have loaded

Interviews are won on specifics. Have these three ready in 60 seconds each — each
one demonstrates a different competency.

**1. The truncation bug — "how do you find a bug that no metric shows?"**
Reviewing my own retrieval for deployment, I measured token counts against the
embedder's limit rather than assuming they fit. 98.6% of chunks were over. It was
invisible because BM25 covered for the dense half, and the fix was a parameter.
*Competency: you audit your own systems, and you know redundancy hides failure.*

**2. The 37× slowdown — "tell me about a debugging session."**
An 8B model that loaded, trained, and converged, at 177 s/step, because Windows
paged 11 GB to system RAM instead of raising OOM. Found by noticing an ETA, not by
an error. *Competency: you distrust green lights, and you know the platform.*

**3. The prediction checker that hid its own miss — "tell me about a process
failure."**
I pre-registered five predictions so I could not rationalise afterwards, then
wrote a checker with a one-sided threshold that passed a prediction missing its
band by half — in the direction I liked. *Competency: you audit the instrument,
not just the measurement.*

---

## Questions to ask them

These signal seniority because each one implies you have hit the problem:

- "When you fine-tune, what does your eval look like — and what does it *not*
  cover?"
- "How do you version a retrieval index against the model that was trained on the
  same corpus?"
- "What is your abstention story? When the retrieval comes back empty, what does
  the user see?"
- "Do you merge adapters or serve them against a shared base — and what drove that
  call?"
- "Who decides whether a document is allowed into the corpus, and at what point in
  the pipeline is that enforced?"
