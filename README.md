# 🧠 Sr GenAI Developer — Interview Prep Workspace

Eight self-contained projects covering the ground a senior GenAI interview actually
walks over: **retrieval, fine-tuning across three modalities, local GPU
economics, and agent tooling.** Everything runs on one consumer GPU with no API
keys and no data leaving the machine.

Every project has **three things**: runnable code, a `README.md` that teaches the
concept from scratch, and an `INTERVIEW.md` with the questions you'll be asked
and answers you can defend.

---

## The projects

| # | Project | What it demonstrates | Headline result | Validated |
|---|---|---|---|---|
| **01** | [RAG (local)](01_rag_local/) | Hybrid retrieval, cross-encoder reranking, grounded generation with verified citations, and a real eval harness | reranking took recall@1 **0.88 → 0.94**; **100%** abstention on unanswerable questions; **368 ms** warm p50 | via tests + 08 |
| **02** | [LoRA — text](02_lora_text/) | Fine-tuning a 0.5B LLM for structured extraction; bf16 vs QLoRA | exact match **0% → 84.2%** in **50 s**, 34 MB adapter; QLoRA *loses* | ✅ retrained |
| **03** | [LoRA — image](03_lora_image/) | Teaching Stable Diffusion a brand-new visual concept, measured with CLIP | concept fidelity **+40% relative**, 12 MB adapter, 11 min | ✅ + ablation |
| **04** | [LoRA — voice](04_lora_voice/) | Domain adaptation of Whisper using TTS-synthesised training data | WER **52.1% → 2.1%**, domain terms **1% → 96%**, 80 s; **cross-engine holdout 0.9% WER** | ✅ retrained |
| **05** | [MCP server](05_mcp_server/) | Tools/resources/prompts exposing projects 01+02, plus a full **OAuth 2.1** server with PKCE and per-tool scopes | PKCE rejects stolen codes, refresh rotates, revocation takes effect | ✅ + 10 tests |
| **06** | [Local GPU inference](06_local_gpu_inference/) | Quantisation and batching: memory, speed **and** the quality you pay for it | batching **~30×**; int4 costs **+9.2% perplexity**; decoding is **overhead**-bound, not bandwidth-bound | ⚠️ see below |
| **07** | [RAG at scale](07_rag_at_scale/) | Real FineWeb-Edu corpus, binary+int8 precision cascade, measured latency scaling, and modern retrieval techniques | **13.6 M chunks**; **32×** memory reduction at **0.985 recall@10**; flat scan O(n) at 91 ms/M — the measured case for IVF/HNSW | ✅ within 2% |
| **08** | [RAG paradigms](08_rag_paradigms/) | GraphRAG + Agentic RAG vs the vector baseline — same corpus, same questions, cost included | **vector won everything, incl. multihop (100% vs 29%)** | ⚠️ see below |
| **tests** | [pytest + DeepEval](tests/) · [INTERVIEW](tests/INTERVIEW.md) | **171 deterministic tests at 99% coverage**, mutation-verified, + judge-calibrated LLM gates | the DeepEval layer **found a real hallucination** that carried a valid citation | 16-bug ledger |

They share one fictional domain — the **"Atlas"** warehouse-robotics platform —
on purpose. Project 05 calls projects 01 and 02 as tools, so the set reads as one
system rather than eight disconnected demos. That's a much stronger portfolio story.

---

---

## 🔬 The validation campaign — every project re-run from scratch

Months after building them, all eight projects were re-run end to end: models
retrained, benchmarks re-measured, servers restarted. **The point was not to
confirm the numbers — it was to find out which ones were wrong.**

Five of the eight turned up something real, and every finding is now in the
project's own README rather than buried here.

| project | reproduced? | what validating it found |
|---|---|---|
| **02** LoRA text | memory & adapter size **exact**; accuracy ±1 example | **A headline number was inflated.** `action` scored 95.8% against an **86.7% lookup baseline** — 4 of 5 components had exactly one action. Dataset rebuilt so actions key on the *symptom*; baseline fell to **31.7%** and the model reached **100%**. Making the task harder improved the result. |
| **03** LoRA image | VRAM & adapter **exact**; fidelity **+40.4%** vs +40% | **The headline lesson was confounded.** "Attempt 1 failed from caption dilution" — but it also used half the rank and half the steps. The controlled ablation confirms it (**41% of the fidelity gain**), and found caption dilution is *indistinguishable from turning the adapter down*. |
| **04** LoRA voice | every base number **exact**; SAPI holdout **better** (0.9% vs 1.5%) | **Closed a gap the README admitted to.** General-English WER **1.9% → 3.8%** — small absolutely, a *doubling* relatively. The damage is **formatting drift**, not lost hearing: "half past eight" → "half-past-eight", hyphenation learned from `CON-401`. |
| **05** MCP + OAuth | full flow verified live | **PKCE had no test at all.** Porting the demo to pytest exposed that `_verify_pkce` — documented as "the single most important function in this file" — is **never called**. The SDK enforces PKCE; the repo's function was decorative. |
| **06** GPU inference | memory & perplexity **exact**; speed **not** | **A 36% run-to-run noise floor**, larger than every difference between fp32/fp16/bf16/int4 combined. It flipped the sign of a published comparison. The benchmark now takes `--repeats` and reports medians + spread; each claim is marked robust or unsupported. |
| **07** RAG at scale | **within 2%** across a 136× size range | **A number that was wrong rather than missing.** `bytes_text` re-assigned itself on every commit, so a 13.6 M-chunk index reported "0.0 GB of text". Fixed and pinned. |
| **08** RAG paradigms | headline **exact** (100% vs 29%) | **The right answer for the wrong reason.** The scale argument ("top-4 of 30 is 13% of the corpus") was never tested. Co-retrieval survives a **10,000× larger corpus** — the crossover is driven by **distractor competition**, not size. |

### What the campaign says about measurement itself

Three patterns worth more than any single number:

1. **What reproduces exactly vs what doesn't is not random.** Memory, adapter
   size, perplexity, recall and quality reproduced *to the digit* everywhere.
   Wall-clock speed did not, anywhere. Deterministic computation over fixed
   inputs is reproducible; timing on a shared consumer GPU is a distribution.
   Quote the first to three decimals, never the second.

2. **The same machine gave 2% reproducibility in project 07 and 36% in project
   06** — because 07's harness already took repeats and reported percentiles,
   and its workload was bandwidth-bound and stable, while 06 took one sample of
   an overhead-bound workload. Methodology explained the gap, not hardware.

3. **Mechanisms reproduce better than numbers — and are worth more.** Project
   02's bf16-vs-QLoRA gap sat entirely in `error_code` before *and* after a
   complete dataset rebuild. Project 03's caption effect survived a controlled
   re-test. An explanation that survives a re-run is defensible in a way a
   percentage never is.

**Nothing was quietly corrected.** Each project keeps the original number, the
re-run number, and the reason they differ, because the reasoning is the part an
interviewer can probe.

---

## Start here

```powershell
# one-time setup, ~15 min  (see SETUP.md)
pip install uv
uv venv --python 3.12 $HOME\.venvs\genai
# ... torch + libraries, see SETUP.md

.\activate.ps1
python 00_shared\check_env.py
```

Then work through the projects in order. **01 → 02 → 05** is the tightest path if
you're short on time: retrieval, fine-tuning, and the agent layer that ties them
together.

---

## 🎯 How to use this for interview prep

**Two weeks out —** run every project end-to-end yourself. Don't read the
results; regenerate them. The numbers in each README came from this machine and
you should be able to say "I measured this" and mean it.

That is exactly what [the validation campaign](#-the-validation-campaign--every-project-re-run-from-scratch)
above was, and it is worth doing for a reason beyond confidence: **re-running
your own work is the cheapest way to find the claim you cannot defend.** Five of
eight projects had one. An interviewer who asks "did you run that twice?" is
asking whether you know which of your numbers are stable — and after this, you
do.

**One week out —** read the `INTERVIEW.md` files. Each one has a 60-second pitch,
the questions that follow, and the trade-offs behind each answer. Practise the
pitches out loud.

**The day before —** re-run the eval harnesses so your numbers are fresh:

```powershell
cd 01_rag_local      ; python eval\evaluate.py
cd ..\02_lora_text   ; python evaluate.py
cd ..\06_local_gpu_inference ; python benchmark.py
```

And the cheapest check of all — 88 seconds, no GPU, no Ollama, and it fails if
any number in these READMEs has drifted from what the code actually does:

```powershell
pytest tests/ -m "not llm and not gpu"
```

### What actually separates senior from mid in these interviews

Every project here is built around the same five habits, because these are what
get probed:

1. **You measured it.** Not "reranking improves quality" but "reranking took
   recall@1 from 0.88 to 0.94 for 40 ms, here's the harness". A senior candidate
   brings numbers; a mid-level one brings opinions.

2. **You know when the technique is wrong.** Project 02 runs QLoRA and it *loses*
   — slower, more VRAM, 10 points worse — and explains exactly why (at 0.5B the
   memory is activations, not weights). Knowing when *not* to reach for the
   fashionable method is the clearest seniority signal there is.

3. **You read your own results honestly.** Project 01's hybrid retrieval ties with
   dense-only, and the README says so, and says why (the corpus is too small to
   discriminate). Project 03 documents a failed first attempt and the diagnosis.
   Interviewers trust candidates who volunteer their limitations.

4. **You separate the failure modes.** Retrieval quality vs generation quality.
   Tool correctness vs tool selection. Concept fidelity vs prompt adherence.
   Every project measures both halves, because the fixes are different.

5. **You verify your own claims.** This one was learned the hard way, twice.
   A documentation audit found two places claiming a bug was "pinned by test"
   when no such test existed — and chasing one turned up a **live bug** (chunks
   silently truncated past the embedder's window). Then re-running all eight
   projects found **five more claims that did not survive contact with a second
   measurement**, including a headline number resting on a lookup table and a
   36% noise floor underneath a published speed comparison.

   So claims here are now compiled, not asserted: `tests/test_doc_drift.py`
   checks counts, coverage and every cited test against the code, and CI goes
   red when prose drifts. **A claim you have not re-tested is a claim you cannot
   defend — and the fastest way to find your weakest one is to re-run your own
   work.**

---

## Cross-cutting themes worth being able to discuss

**Fine-tune or RAG?** RAG adds *knowledge*, fine-tuning changes *behaviour*.
Project 01 is the knowledge half, 02 the behaviour half, and the strongest
production systems use both. Getting this backwards is the most common
architectural mistake in applied GenAI.

**LoRA is one idea across every modality.** Projects 02, 03 and 04 apply the same
low-rank adapter trick to a language model, a diffusion UNet and an ASR
encoder-decoder. The target modules change (`q,k,v,o` + MLP / cross-attention /
both stacks); the mathematics doesn't.

**Know which resource you're short of.** Project 06 makes it concrete — and
overturns the received wisdom while doing it. "Decoding is memory-bandwidth
bound" is the textbook line, but tripling the weights (0.5B → 1.5B) left decode
speed unchanged: at this scale, in this runtime, you're **overhead**-bound. That
one fact explains why int4 bought memory but zero speed, why int8 was 5× slower,
and why batching gave 32×. It's the same lesson as project 02, where QLoRA lost
because the memory was activations, not weights. Optimise the resource you're
actually short of.

**Grounding is a discipline, not a prompt.** Abstain paths, numbered citations,
and *mechanical verification* of those citations. Project 01 builds it, and
project 05 proves it survives being wrapped in a tool call.

---

## Layout

```
learning/
├── README.md            you are here
├── SETUP.md             one-time environment setup + troubleshooting
├── activate.ps1         activate the shared venv
├── 00_shared/
│   ├── check_env.py     verify GPU, CUDA build, libraries, Ollama
│   └── gpu.py           VRAM + timing helpers
├── 01_rag_local/
├── 02_lora_text/
├── 03_lora_image/
├── 04_lora_voice/
├── 05_mcp_server/
├── 06_local_gpu_inference/
├── 07_rag_at_scale/
├── 08_rag_paradigms/
└── tests/               171 deterministic tests (99% coverage) + DeepEval gates
    ├── README.md        the two layers, judge calibration, the 16-bug ledger
    └── INTERVIEW.md     testing a non-deterministic system, and what isn't tested
```

Each project directory:

```
<project>/
├── README.md          the tutorial: concept, results, how to run, tuning knobs, FAQ
├── INTERVIEW.md       pitch, likely questions, answers, trade-offs
├── requirements.txt   what this project alone needs
├── make_dataset.py    generate the demo data (self-contained, no downloads)
├── train_lora.py      / ingest.py / server.py — the main artifact
└── evaluate.py        the measurement harness
```

---

## 📊 Project 07: what got measured, and what it proved

The scale-RAG project finished its measurement arc. The short version:

**The precision cascade works.** Binary (1-bit) codes for the full scan, int8 on
disk to rescore the top 500: **32× memory reduction at 0.985 recall@10 with a
quality ratio of 1.0000** against exact float32 search. At 13.6 M indexed chunks
the rescore stage costs **0.45 ms** — a 136× larger index moved it a fifth of a
millisecond, confirming it scales with candidate depth and never with corpus
size.

**The flat scan doesn't.** Measured O(n) at **91.3 ms per million vectors**,
perfectly linear from 100 k to 13.6 M (a 3.4 M-scale run predicted the 13.6 M
number to within 3%). Projected to the full corpus: **~28.8 s per query** — so
the architecture provably does not reach 200 GB, and the benchmark exists
precisely to prove that rather than hide it. That is the measured argument for
IVF/HNSW partitioning.

**Sixteen real bugs found and pinned by tests across the whole repo** — not
only this project: they run from project 01's chunker through project 07's
threading to project 08's entity resolution, and one of them is a documentation
claim that turned out to be false. The full
ledger, each with the test that keeps it fixed, is in
[tests/README.md](tests/README.md#-the-bug-ledger--16-real-bugs-each-pinned-by-a-named-test) — including a chunker
producing 42× too many chunks, a resume path that silently duplicated data, a
lost-sentinel deadlock in the threading, and a RAG hallucination that carried a
*valid* citation — caught only by the DeepEval judge layer, then fixed in the
prompt and pinned deterministically. The coverage pass alone surfaced two more: an unsplittable-paragraph case that silently truncated content past the embedder's window, and a rewrite that had dropped stall detection from the pipeline's polling loops.

The corpus (**165 GB**, 82 parquet shards) and the index it produces (**5.9 GB**)
live at `C:\genai-data` — **171 GB** in total, outside the repo and outside
OneDrive. Both the download and the index build are resumable;
`python build_index.py --status` shows where things stand.
---

## Validated on

RTX 5070 Ti 16 GB (sm_120) · Windows 11 · Python 3.12.13 · PyTorch 2.11.0+cu128 ·
Transformers 5.15.1 · PEFT 0.20.0 · Diffusers 0.40.0 · Ollama `qwen2.5:7b`

Full environment details and troubleshooting in [SETUP.md](SETUP.md).
