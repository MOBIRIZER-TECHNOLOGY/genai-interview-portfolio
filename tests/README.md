# 🧪 Test suite — pytest + DeepEval

Two layers, because LLM systems have two failure modes and they need different
instruments.

```powershell
..\activate.ps1

pytest tests/ -m "not llm and not gpu"    # 48 tests, 0.4 s, no GPU, no network
pytest tests/ -m llm -v                   # DeepEval quality gates (needs Ollama)
pytest tests/                             # everything
```

---

## Layer 1 — deterministic (48 tests, 0.38 s)

Fast, exact, no GPU, no network, CI-safe. **Every test here is a regression test
for a bug that actually happened in this repo.**

| file | protects | the bug it pins |
|---|---|---|
| `test_chunking.py` | chunk boundaries | the **42× chunker** — a 500-char doc produced ~180 near-identical chunks |
| `test_quantization.py` | binary/int8 codec + methodology | the **self-in-corpus** 0.9 recall cap, and the **synthetic-vector** trap |
| `test_crash_safety.py` | append-only index invariants | **resume duplication**, and `ScaleIndex` sizing from file length not manifest |

### Why this layer earns its place

Three of the four real bugs found while building project 07 were deterministic,
and every one of them cost **50+ minutes of confused debugging**:

- the 42× chunker surfaced only as *"this shard is taking suspiciously long"*
- the recall cap looked like a plausible plateau at 0.899
- the index happily benchmarked 3.4 M rows the manifest said didn't exist

`test_short_document_yields_one_chunk` reproduces the first in **under a
millisecond**. That gap is the entire argument.

### These tests are load-bearing — verified

Reintroducing the original chunker bug and re-running:

```
FAILED test_short_document_yields_one_chunk
FAILED test_any_document_under_target_is_one_chunk[61 .. 1399]   (6 cases)
FAILED test_document_just_over_target_does_not_explode
FAILED test_chunks_are_near_target_length_on_realistic_text
FAILED test_no_runaway_chunk_count
FAILED test_terminates_on_pathological_input
```

**11 tests fire.** A suite that passes on known-broken code is worse than no
suite, because it manufactures confidence. This one was mutation-checked.

---

## Layer 2 — DeepEval quality gates (`-m llm`)

Deterministic tests cannot tell you whether an *answer* is any good. That needs a
judge, and DeepEval is the standard framework for it.

| metric | fails when |
|---|---|
| `FaithfulnessMetric` | the answer asserts what the context doesn't support |
| `AnswerRelevancyMetric` | the answer is true but doesn't address the question |
| `ContextualPrecisionMetric` | retrieval returned mostly noise |
| `HallucinationMetric` | the answer contradicts its context |
| `GEval` (custom) | **citation discipline** — claims without `[n]` attribution |

The `GEval` rubric is the interesting one: off-the-shelf metrics don't know this
pipeline is *supposed* to cite block numbers and abstain when unsupported.
G-Eval lets you grade the property you actually care about.

### Local judge, no API key

`tests/ollama_judge.py` implements `DeepEvalBaseLLM` over Ollama
(`qwen2.5:7b`, temperature 0). Nothing leaves the machine, nothing costs money,
consistent with every other model here.

The detail that makes it work: DeepEval metrics ask the judge for **structured
JSON** and parse it. A judge returning prose fails every metric with a *parse
error* rather than a low score — which looks like a broken metric, not a broken
model. So the adapter sets Ollama's `format="json"` and validates into
DeepEval's pydantic schema, with a salvage path for JSON wrapped in prose.

### Thresholds are set below current performance, deliberately

The pipeline currently scores 100% fact recall and 100% citation validity on its
own eval set. The gates sit at **0.6–0.7**.

That's not laziness. A gate pinned to today's exact score fails on harmless
variance and gets disabled within a week. A gate set where *real* degradation
lives keeps working — the point is catching regressions, not certifying
perfection.

### The honest caveat

An LLM judge is an instrument with its own error rate: biased toward verbose
answers, inconsistent near boundaries, and capped by its own competence — a 7B
judge cannot reliably grade claims it doesn't understand.

Use these to compare runs of **your own system**, not to certify absolute
quality. **Where a deterministic check exists, prefer it.** Project 01's
mechanical citation verification is free, exact, and cannot hallucinate its own
verdict — which is why `test_crash_safety.py` also asserts grounding and
abstention deterministically, so those properties stay covered even if every
judge-based gate were disabled.

---

## Markers

| marker | meaning |
|---|---|
| `llm` | needs a running Ollama server |
| `gpu` | needs CUDA |
| `slow` | more than a few seconds |
| `integration` | exercises several projects together |

CI should run `-m "not llm and not gpu"` — 0.4 s, no infrastructure.

---

## What is NOT covered yet

Stated plainly, because a test README that implies full coverage is its own kind
of lie:

- **The OAuth 2.1 server** (`05_mcp_server`) is verified end-to-end by
  `auth_client_demo.py` — including PKCE rejection, refresh rotation and
  revocation — but that is a script, not pytest. It should be ported.
- **Training loops** (projects 02/03/04) have no tests. Their `evaluate.py`
  harnesses measure output quality but assert nothing.
- **The threaded pipeline** (`scale/pipeline.py`) has no concurrency test. The
  lost-sentinel bug that killed a shard at 3.25 M chunks would not be caught
  here — it needs a test that fills the bounded queues and forces the shutdown
  path under backpressure.

That last one is the most valuable gap, and the one I'd close next.
