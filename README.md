# 🧠 Sr GenAI Developer — Interview Prep Workspace

Six self-contained projects covering the ground a senior GenAI interview actually
walks over: **retrieval, fine-tuning across three modalities, local GPU
economics, and agent tooling.** Everything runs on one consumer GPU with no API
keys and no data leaving the machine.

Every project has **three things**: runnable code, a `README.md` that teaches the
concept from scratch, and an `INTERVIEW.md` with the questions you'll be asked
and answers you can defend.

---

## The projects

| # | Project | What it demonstrates | Headline result |
|---|---|---|---|
| **01** | [RAG (local)](01_rag_local/) | Hybrid retrieval, cross-encoder reranking, grounded generation with verified citations, and a real eval harness | reranking took recall@1 **0.88 → 0.94**; **100%** abstention on unanswerable questions; **368 ms** warm p50 |
| **02** | [LoRA — text](02_lora_text/) | Fine-tuning a 0.5B LLM for structured extraction; bf16 vs QLoRA | exact match **0% → 84.2%** in **59 s**, 34 MB adapter |
| **03** | [LoRA — image](03_lora_image/) | Teaching Stable Diffusion a brand-new visual concept, measured with CLIP | concept fidelity **+40% relative**, 12 MB adapter, 11 min — plus a documented failed first attempt |
| **04** | [LoRA — voice](04_lora_voice/) | Domain adaptation of Whisper using TTS-synthesised training data | WER **52.1% → 2.5%**, domain terms **1% → 96%**, 78 s; **cross-engine holdout passed** (1.5% WER on SAPI); mic-recording pipeline included |
| **05** | [MCP server](05_mcp_server/) | Tools, resources and prompts that expose projects 01 and 02 to any AI client | 4 tools live in Claude Code |
| **06** | [Local GPU inference](06_local_gpu_inference/) | Quantisation and batching: memory, speed **and** the quality you pay for it | batching **32.5×** throughput; int4 costs **+9.2% perplexity**; found the benchmark refuted the textbook claim |

They share one fictional domain — the **"Atlas"** warehouse-robotics platform —
on purpose. Project 05 calls projects 01 and 02 as tools, so the set reads as one
system rather than six disconnected demos. That's a much stronger portfolio story.

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

**One week out —** read the `INTERVIEW.md` files. Each one has a 60-second pitch,
the questions that follow, and the trade-offs behind each answer. Practise the
pitches out loud.

**The day before —** re-run the eval harnesses so your numbers are fresh:

```powershell
cd 01_rag_local      ; python eval\evaluate.py
cd ..\02_lora_text   ; python evaluate.py
cd ..\06_local_gpu_inference ; python benchmark.py
```

### What actually separates senior from mid in these interviews

Every project here is built around the same four habits, because these are what
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
└── 06_local_gpu_inference/
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

## Validated on

RTX 5070 Ti 16 GB (sm_120) · Windows 11 · Python 3.12.13 · PyTorch 2.11.0+cu128 ·
Transformers 5.15.1 · PEFT 0.20.0 · Diffusers 0.40.0 · Ollama `qwen2.5:7b`

Full environment details and troubleshooting in [SETUP.md](SETUP.md).
