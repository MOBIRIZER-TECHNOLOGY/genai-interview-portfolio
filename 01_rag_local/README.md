# 📚 Project 01 — Local RAG with Hybrid Retrieval, Reranking and Evaluation

A **fully local** question-answering system over a private document set. No API
keys, no data leaving the machine: embeddings and reranking run on your GPU, the
LLM runs in Ollama.

> **In one sentence:** we take 8 internal documents about a fictional robotics
> platform that no LLM has ever seen, index them, and get grounded, cited answers
> in ~370 ms — plus a measurement harness that proves the retrieval actually
> works instead of just asserting it.

---

## 🧠 The idea (for non-experts)

A language model knows what was in its training data and nothing else. Ask it
about **your** company's internal systems and it will either say "I don't know"
or — much worse — invent something confident and wrong.

**RAG (Retrieval-Augmented Generation)** fixes this with a simple move: before
answering, go *find* the relevant text from your documents, paste it into the
prompt, and tell the model "answer only from this". The model stops being a
knowledge store and becomes a reading-comprehension engine.

That's the easy part, and it's where most demos stop. The hard part — and what
this project is really about — is **the retrieval step**. If the search hands the
model the wrong paragraph, the model gives a wrong answer with total confidence.
Everything below is about making retrieval good and then *proving* it is.

---

## ✅ Proof it works (measured on this machine)

RTX 5070 Ti (16 GB), `qwen2.5:7b` in Ollama, 8 documents → 30 chunks.

### Retrieval quality — 17 answerable questions

| Retriever | recall@1 | recall@3 | MRR | p50 latency |
|---|---:|---:|---:|---:|
| dense only | 0.882 | 1.000 | 0.931 | 7.2 ms |
| BM25 only | 0.824 | 0.941 | 0.873 | **0.07 ms** |
| hybrid (RRF) | 0.882 | 1.000 | 0.931 | 4.4 ms |
| dense + rerank | **0.941** | 1.000 | **0.961** | 46 ms |
| BM25 + rerank | **0.941** | 1.000 | **0.961** | 38 ms |
| hybrid + rerank | **0.941** | 1.000 | **0.961** | 44 ms |

### Generation quality — 17 answerable + 3 unanswerable

| Metric | Score |
|---|---:|
| gold document reached the context window | 100% |
| required facts present in the answer | 100% |
| every `[n]` citation maps to a real block | 100% |
| correctly abstained on unanswerable questions | 100% (3/3) |
| p50 end-to-end latency (warm model) | **368 ms** |

First query after starting Ollama takes ~9 s because the model loads into VRAM.
Every query after that is sub-second. Always report warm and cold separately.

### 🔍 Read these numbers honestly — this matters in an interview

**The reranker is the only thing that moved recall@1** (0.88 → 0.94). Hybrid
fusion tied with dense-only here. That is *not* a claim that hybrid is useless —
it's a claim that **this corpus is too small to show its value**. With 30 chunks,
dense retrieval already surfaces the right document in the top 3 every time, so
there's nothing for BM25 to rescue. The failure mode hybrid exists to fix (exact
rare tokens like `TLM-330` lost in a 500k-chunk index) doesn't occur at this
scale.

Saying that out loud is worth more than the number itself. "Our eval set is too
easy to discriminate between these two options" is a senior answer. Quoting a
saturated benchmark as proof is a junior one.

---

## 📁 What's in this project

```
01_rag_local/
├── corpus/                  8 markdown docs about the fictional "Atlas" platform
├── rag/
│   ├── chunking.py          heading-aware splitting + breadcrumb prefixes
│   ├── embed.py             BGE bi-encoder on GPU (asymmetric query/passage)
│   ├── store.py             FAISS IndexFlatIP + JSONL metadata sidecar
│   ├── retrieve.py          dense + BM25, fused with Reciprocal Rank Fusion
│   ├── rerank.py            cross-encoder second stage
│   ├── generate.py          Ollama call, numbered context, citation verification
│   └── pipeline.py          the whole thing behind one object
├── eval/
│   ├── qa_set.json          20 hand-labelled questions (3 deliberately unanswerable)
│   ├── evaluate.py          recall@k, MRR, fact recall, citation validity, abstention
│   └── results.json         written by the harness
├── ingest.py                corpus/ -> index/
├── ask.py                   CLI, with a --compare mode that shows each retriever
├── serve.py                 FastAPI + token streaming
└── requirements.txt
```

**How it fits together:**

```
corpus/*.md ──chunk──▶ 30 chunks ──embed(GPU)──▶ index/ (FAISS + metadata)
                                                    │
question ──┬─ dense retrieve top-20 ─┐              │
           └─ BM25  retrieve top-20 ─┴─ RRF fuse ───┘
                                        │
                            cross-encoder rerank
                                        │
                                  top-4 blocks
                                        │
                     numbered-context prompt ──▶ Ollama qwen2.5:7b
                                        │
                          answer + machine-verified citations
```

---

## 🚀 How to run it

### 0. Prerequisites

```powershell
# from the repo root, see ../SETUP.md for the one-time venv setup
..\activate.ps1
ollama pull qwen2.5:7b
```

### 1. Build the index

```powershell
python ingest.py
```
Chunks the corpus, embeds on GPU, writes `index/`. Takes ~2 seconds. The
embedding model (133 MB) downloads on first run.

### 2. Ask something

```powershell
python ask.py "what is the Rotterdam rule?"
python ask.py "what does TLM-330 mean?" --show-context
python ask.py "what is the salary band for a senior engineer?"   # should abstain
```

### 3. See *why* hybrid retrieval exists

```powershell
python ask.py --compare "what does TLM-330 mean?"
```
Prints the top hits from dense-only, BM25-only and hybrid side by side. Watch
what happens to a rare token like `TLM-330` versus a paraphrased question like
"how long do we keep camera footage".

### 4. Measure it

```powershell
python eval/evaluate.py --retrieval-only    # ~10 seconds, no LLM
python eval/evaluate.py                     # full, ~30 seconds
```

### 5. Serve it

```powershell
python serve.py            # http://localhost:8000/docs
```

```powershell
# streaming answer
curl -N -X POST localhost:8000/ask/stream -H "content-type: application/json" `
  -d '{\"question\":\"what is the Rotterdam rule?\"}'
```

---

## 🔧 Use your OWN documents

1. Drop `.md` files into `corpus/` (delete the Atlas ones).
2. `python ingest.py`
3. Write your own `eval/qa_set.json` — **do this before tuning anything.**
   Without a labelled set you cannot tell an improvement from a regression, and
   you will spend a week "improving" a prompt with no idea whether it helped.

15–25 questions is enough to be useful. Include at least 3 unanswerable ones.

---

## ⚙️ The knobs that actually matter

| Knob | Where | Effect |
|---|---|---|
| `--max-tokens` | `ingest.py` | Chunk size. Too small → facts split apart. Too large → the reranker and the LLM both drown in irrelevant text. 250–400 is a sane band for docs. |
| `--overlap` | `ingest.py` | Insurance against a fact landing exactly on a boundary. ~20% of chunk size. |
| *(automatic)* | `chunking.py` | A single paragraph longer than the budget is hard-split on sentence/space boundaries. Without this, its tail sails past the embedder's 512-token window and is silently never indexed — found by a test, fixed in `_hard_split`. |
| `--candidates` | `ask.py` | How many chunks the reranker sees. More = better recall, linearly more rerank latency. |
| `--top-k` | `ask.py` | How many blocks reach the LLM. **More is not better** — irrelevant context measurably degrades answers ("lost in the middle"). |
| `rrf_k` | `retrieve.py` | RRF damping, default 60. Lower = the #1 hit from each arm dominates more. |
| `temperature` | `generate.py` | Pinned to 0. Non-zero makes your eval numbers noise. |

---

## 🖥️ Tech stack

- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim, 133 MB) on CUDA
- **Reranker:** `BAAI/bge-reranker-base` cross-encoder on CUDA
- **Index:** FAISS `IndexFlatIP` (exact, cosine via normalised inner product)
- **Lexical:** `rank-bm25` (Okapi BM25) with identifier-preserving tokenisation
- **LLM:** `qwen2.5:7b` via Ollama, temperature 0
- **Serving:** FastAPI + SSE streaming
- **Validated on:** RTX 5070 Ti 16 GB, torch 2.11.0+cu128, Python 3.12

---

## ❓ FAQ

**Why FAISS and not Pinecone/Weaviate/pgvector?**
30k vectors fit in 46 MB of RAM. A network hop to a managed service would cost
more latency than the entire search. Reach for a real vector DB when you need
multi-tenancy, filtered search at scale, or durability guarantees — not by default.

**Why is BM25 in here at all when embeddings are "better"?**
Ask the system about `TLM-330`. A 384-dimensional embedding has no distinct
direction for a rare identifier it saw a handful of times in training. BM25 gives
that exact token the highest IDF weight in the corpus. They fail in opposite
directions, which is what makes them a good ensemble.

**Why RRF instead of just averaging the scores?**
Cosine scores live in ~[0.5, 0.9]. BM25 scores are unbounded and corpus-dependent.
Blending them requires a normalisation you'd have to recalibrate every time the
corpus changes. RRF only uses rank, so there is nothing to calibrate.

**The reranker costs 40 ms. Is it worth it?**
Here, +6 points of recall@1 for 40 ms — yes. In a setting where you serve 10k
QPS it might not be. Measure it on *your* traffic; that's what the harness is for.

**How do I stop it hallucinating?**
You don't, entirely. What you do is (a) tell it to abstain and make NOT_FOUND a
named output, (b) require citations by block number, (c) **verify those citations
mechanically** — `verify_citations()` checks every `[n]` against the blocks you
actually sent. An unverifiable citation is decoration.

**Ollama isn't responding?**
`ollama serve` in another terminal, then `ollama list` to confirm the model is
pulled. `serve.py` returns HTTP 502 with the underlying error rather than hanging.

---

## Related projects

This is the project the rest of the retrieval work is built on — four others
import it or extend it:

- **[05_mcp_server](../05_mcp_server/)** — wraps this pipeline as an
  authenticated MCP tool, grounding discipline intact
- **[07_rag_at_scale](../07_rag_at_scale/)** — the same ideas at 13.6 M chunks,
  where flat search stops being viable and quantisation earns its keep
- **[08_rag_paradigms](../08_rag_paradigms/)** — GraphRAG and Agentic RAG
  measured against this pipeline; **this one won**, and the README explains why
- **[06_local_gpu_inference](../06_local_gpu_inference/)** — the generation half:
  what quantisation and batching cost in quality
