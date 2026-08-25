# 🛠️ Setup — one time, ~15 minutes

Everything in this workspace runs **locally on your GPU**. No API keys, no cloud
accounts, no data leaving the machine.

---

## What was actually validated

These projects were built and run end-to-end on:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB, **sm_120** (Blackwell) |
| Driver | 591.86 |
| OS | Windows 11 Pro |
| Python | **3.12.13** (via `uv`) |
| PyTorch | **2.11.0+cu128** |
| Transformers / PEFT / Diffusers | 5.15.1 / 0.20.0 / 0.40.0 |
| Ollama | `qwen2.5:7b` (the only model any project calls) |

Any modern NVIDIA GPU with ≥8 GB works for most projects; see the VRAM table
at the bottom.

---

## 1. Python 3.12 (not 3.13+)

PyTorch wheels lag the newest CPython by a release or two. Use `uv` — it will
fetch the right interpreter for you, no system install needed.

```powershell
# install uv if you don't have it
pip install uv
```

## 2. One shared virtual environment

All six projects share one venv. That's deliberate: PyTorch + CUDA is ~3 GB, and
installing it six times would cost ~18 GB for no benefit. Each project still
ships its own `requirements.txt` so you can see exactly what *it* needs.

**Put the venv outside OneDrive/Dropbox.** A ~10 GB venv inside a synced folder
will churn your sync client for hours and can corrupt hardlinked packages.

```powershell
uv venv --python 3.12 $HOME\.venvs\genai
```

## 3. PyTorch, matched to your GPU

This is the step people get wrong. The CUDA build must match your GPU
architecture — a Blackwell (sm_120) card **needs cu128 or newer**; an older wheel
installs happily and then fails at the first kernel launch.

```powershell
$py = "$HOME\.venvs\genai\Scripts\python.exe"
uv pip install --python $py --link-mode=copy torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu128
```

> **`--link-mode=copy` matters on OneDrive.** `uv` hardlinks from its cache by
> default, and OneDrive's filesystem rejects that with
> `os error 396: the cloud operation cannot be performed on a file with
> incompatible hardlinks`. Copy mode uses a little more disk and just works.

Check your CUDA version at <https://pytorch.org/get-started/locally/> if you're
not on Blackwell.

## 4. Everything else

```powershell
uv pip install --python $py --link-mode=copy `
    transformers accelerate peft datasets safetensors sentencepiece protobuf `
    sentence-transformers faiss-cpu rank-bm25 diffusers bitsandbytes `
    soundfile librosa jiwer evaluate matplotlib pandas `
    fastapi "uvicorn[standard]" httpx pydantic python-dotenv mcp rich tqdm
```

## 5. Ollama (projects 01, 05, 07 and 08)

Install from <https://ollama.com/download>, then:

```powershell
ollama pull qwen2.5:7b
```

**One model, deliberately.** Projects 02 and 06 also use a 0.5B — but that is
`Qwen/Qwen2.5-0.5B-Instruct` loaded through **transformers**, from the
HuggingFace cache, because those projects fine-tune and benchmark the weights
directly. It is a different artifact from Ollama's `qwen2.5:0.5b`, and pulling
the Ollama one gets you 400 MB you will never load. (This document used to tell
you to pull it, and `check_env.py` used to nag when you hadn't.)

## 6. Verify

```powershell
.\activate.ps1
python 00_shared\check_env.py
```

You want `[ OK ]` on Python, PyTorch, GPU, CUDA build and bfloat16. `[WARN]` on
an optional library only matters for the project that uses it.

### And run the test suite

It is the most complete check in the repo — 171 deterministic tests in ~88
seconds, no GPU and no Ollama required — and it verifies the documentation as
well as the code, so a stale number in any README fails it:

```powershell
uv pip install --python $py --link-mode=copy -r tests
equirements.txt
pytest tests/ -m "not llm and not gpu"
```

The LLM-judged layer (`-m llm`) needs Ollama running; see
[tests/README.md](tests/README.md).

---

## Daily use

```powershell
.\activate.ps1          # activates the venv + sets the quiet-logging env vars
cd 01_rag_local
python ingest.py
```

---

## 🩹 Troubleshooting

**`torch.cuda.is_available()` is False**
You installed a CPU wheel. `pip uninstall torch` and reinstall with the
`--index-url` above. Verify with `python -c "import torch; print(torch.version.cuda)"`.

**`no kernel image is available for execution on the device`**
CUDA build too old for your GPU. sm_120 (RTX 50-series) needs cu128+.

**`os error 396` / "incompatible hardlinks" during install**
You're installing into a OneDrive-synced path. Add `--link-mode=copy`, or move
the venv outside the synced folder (recommended).

**A package imports but has no attributes** (e.g. `module 'yaml' has no attribute 'dump'`)
A partially-written package directory, usually from an interrupted install.
`uv pip install --reinstall <package>`.

**`Warning: You are sending unauthenticated requests to the HF Hub`**
Harmless — just rate limits. Set `HF_TOKEN` if you hit them.

**Windows symlink warning from `huggingface_hub`**
Harmless; costs some disk. `activate.ps1` sets
`HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence it.

**Out of memory**
Lower `--batch-size` and raise `--grad-accum` to keep the effective batch, or add
`--gradient-checkpointing`. Also close anything else holding VRAM — Ollama keeps
a model resident for 5 minutes after use (`ollama stop <model>` frees it).

---

## 💾 VRAM needed per project

| Project | Peak VRAM (measured) | Notes |
|---|---|---|
| 01 RAG | ~1.5 GB + Ollama (~5 GB for a 7B) | embedder + reranker are tiny |
| 02 LoRA text | **7.5 GB** | 0.5B base, batch 8 × 512 tokens |
| 03 LoRA image | **4.1 GB** | SD 1.5 at 512px, batch 2 |
| 04 LoRA voice | ~6 GB | Whisper-small, batch 4 |
| 05 MCP | inherits 01 + 02 | lazy-loaded |
| 06 Benchmarks | scales with the variant | fp32 is the ceiling |
| 07 RAG at scale | ~1 GB (unmeasured) + Ollama | `bge-small` in fp16; **RAM and disk are the real constraints here, not VRAM** — see below |
| 08 RAG paradigms | Ollama only (~5 GB for the 7B) | graph extraction and the agent loop are LLM calls, no local training |

## 💽 Disk

| | |
|---|---|
| venv (torch + CUDA + libs) | ~9 GB |
| HuggingFace model cache | ~8 GB (SD 1.5, Whisper, BGE, CLIP, SpeechT5) |
| Ollama models | ~5 GB |
| Generated datasets + adapters | < 1 GB |
| **Project 07 corpus + index (`C:\genai-data`)** | **171 GB measured** — see below |

The HF cache lives in `%USERPROFILE%\.cache\huggingface` — outside OneDrive by
default, which is what you want. Override with `HF_HOME` if you need it elsewhere.

### Project 07 needs its own disk, deliberately

Projects 01–06 and 08 fit in the numbers above. **Project 07 does not**: its
FineWeb-Edu corpus and index live at `C:\genai-data` and currently occupy
**171 GB** — 165 GB of corpus (82 parquet shards) plus 5.9 GB of index.

That path is chosen, not incidental — it is **outside the repo** (so nothing that
large is ever a candidate for git) and **outside OneDrive** (so the sync client
never tries to upload a 2 GB parquet shard mid-write). Both downloads and index
builds are resumable; `python build_index.py --status` reports where they stand.

You can work through project 07's code and read its measurements without any of
this — only re-running the index build needs the corpus.
