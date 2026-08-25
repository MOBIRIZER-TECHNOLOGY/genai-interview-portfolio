# 🧪 Test suite — pytest + DeepEval

> Interview prep for this layer — how to test a non-deterministic system, calibrating a judge, and what *isn't* tested — is in
> **[INTERVIEW.md](INTERVIEW.md)**.

Two layers, because LLM systems have two failure modes and they need different
instruments.

```powershell
..\activate.ps1

pytest tests/ -m "not llm and not gpu and not slow"   # fast subset, seconds
pytest tests/ -m "not llm and not gpu"               # 159 tests, ~90 s, 99% coverage
pytest tests/ -m llm -v                   # DeepEval quality gates (needs Ollama)
pytest tests/                             # everything
```

---

## Layer 1 — deterministic (159 tests, 99% coverage)

```
pytest tests/ -m "not llm and not gpu" --cov=07_rag_at_scale/scale --cov=01_rag_local/rag

TOTAL   1035 statements   6 missed   99%     (14 of 18 files at 100%;
         project-08 graphrag + agentic included; demo-CLI __main__
         blocks excluded via .coveragerc — they are run manually)
```

Coverage was **not** an original goal and the honest history matters: the first
suite was regression-targeted (every test pinned a real bug) and measured
**56%** when coverage was first checked — `embed.py`, `rerank.py` and
`pipeline.py` sat at 0% because no bug had happened there *yet*. The coverage
pass closed that the right way round:

- **It found another real bug**: a single paragraph with no blank lines passed
  through `_split_long` whole — a 1,128-token chunk against a 320 budget, whose
  tail the 512-token embedding model then silently truncated. Fixed with
  `_hard_split`. **That fix turned out to be incomplete** — a later
  documentation audit found the overlap carry re-inflating chunks to 640 tokens,
  past the same window, because it carried whole paragraphs and after
  `_hard_split` a paragraph *is* the whole budget. Both halves are now pinned
  (ledger rows 2 and 3).
- **It found a real regression**: the event-based shutdown rewrite had silently
  dropped stall detection from the main and chunker polling loops, resurrecting
  the eternal-silent-hang failure the detector was built to kill. Restored,
  pinned by two stall tests.
- **It deleted 12 statements of dead code** (`ShardPipeline._get`, orphaned by
  the same rewrite). Dead code is not a coverage problem; the fix is removal,
  not a test.
- The 4 remaining uncovered lines are defensive guards unreachable through the
  public API (documented inline where they live).

Fast, exact, no GPU, no network, CI-safe. **Every test here is a regression test
for a bug that actually happened in this repo.**

| file | protects | the bug it pins |
|---|---|---|
| `test_chunking.py` | chunk boundaries | the **42× chunker** — a 500-char doc produced ~180 near-identical chunks |
| `test_quantization.py` | binary/int8 codec + methodology | the **self-in-corpus** 0.9 recall cap, and the **synthetic-vector** trap |
| `test_crash_safety.py` | append-only index invariants | **resume duplication**, and `ScaleIndex` sizing from file length not manifest |
| `test_pipeline_concurrency.py` | the threaded index builder | the **lost sentinel** that killed a shard at 3.25 M chunks |
| `test_rag_components.py` | store / retrieval fusion / generation plumbing | the **unsplittable paragraph** and the **2×-budget overlap carry**, both silently truncated past the embedder window |
| `test_scale_search.py` | two-stage `ScaleIndex` search end to end | the **dropped stall detection** in the rewritten polling loops |
| `test_coverage_gaps.py` | every remaining reachable branch, named per test | (coverage-driven; also where dead `_get` was deleted rather than tested) |
| `test_rag_paradigms.py` | graph walks, entity resolution, the agent's action loop | the **`shed mode` / `shed_mode` split** that left the flagship two-hop edge unreachable, and the **possessive** variant of it |
| `test_paradigm_wiring.py` | block assembly, extraction loop, agent networking | the **null triple field** that crashed a real extraction run 20 chunks in, and junk non-dict entries |
| `test_doc_drift.py` | the numbers in these READMEs | the **test count that went stale in three places at once** (105 / 141 / actually 143), and two test files documented nowhere |

---

## 🐞 The bug ledger — 14 real bugs, each pinned by a named test

Every row is a bug that actually happened in this repo, with the test that
makes it stay fixed. The count in the root README is this table's row count, and
`test_doc_drift.py` fails if they disagree or if any test named here stops
existing — the number cannot drift, and neither can the claim.

| # | bug | where | pinned by |
|---|---|---|---|
| 1 | **42× chunker** — a 500-char doc produced ~180 near-identical chunks | 01 chunking | `test_short_document_yields_one_chunk` |
| 2 | **unsplittable paragraph** — no blank lines, so a 1,128-token chunk passed through whole and the embedder silently truncated its tail | 01 chunking | `test_chunk_markdown_splits_oversized_section` |
| 3 | **the overlap carry re-inflated chunks to 2× budget** (640 tokens against a 320 budget) — reintroducing bug 2 one loop later | 01 chunking | `test_no_chunk_exceeds_the_embedding_window` |
| 4 | **resume duplication** — a restarted shard silently re-appended rows | 07 crash safety | `test_no_duplicate_chunks` |
| 5 | **index sized from file length, not the manifest** — happily benchmarked 3.4 M rows the manifest said didn't exist | 07 crash safety | `test_index_uses_manifest_count_not_file_length` |
| 6 | **lost sentinel deadlock** — killed a shard after 3.25 M embedded chunks | 07 pipeline | `test_completion_is_never_signalled_through_a_bounded_queue` |
| 7 | **dropped stall detection** — the event-based rewrite silently removed it, resurrecting the eternal-silent-hang | 07 pipeline | `test_stall_detector_reports_blocked_producer` |
| 8 | **self-in-corpus recall cap** — a methodology trap that looked like a plausible 0.899 plateau | 07 quantisation | `test_self_in_corpus_caps_recall_at_exactly_0_9` |
| 9 | **synthetic isotropic vectors understate recall** — the benchmark measured the fixture, not the codec | 07 quantisation | `test_synthetic_isotropic_understates_recall` |
| 10 | **`shed mode` vs `shed_mode`** — two disconnected nodes, the two-hop edge present but unreachable | 08 graphrag | `test_underscore_and_space_variants_unify` |
| 11 | **possessives** — "shed mode's" normalised to "shed modes", matching no node | 08 graphrag | `test_possessive_links_to_the_base_entity` |
| 12 | **null fields inside valid triples** — crashed a real extraction run 20 chunks in | 08 graphrag | `test_extract_graph_full_run_with_mocked_llm` |
| 13 | **hallucinated cold retention** — invented "indefinitely" from a table cell reading `none`, carrying a *valid* citation | 01 generation | `test_no_invented_cold_retention` |
| 14 | **doc drift** — the test count went stale in three documents at once (105 / 141 / actually 143) | docs | `test_documented_test_counts_match_the_suite` |

Bugs 3 and 11 were found by a **documentation audit**, not by a code review:
checking whether the claim "pinned by test" was true turned up a fix with no
test behind it (11) and a fix that was **incomplete** (3). Both claims had been
sitting in the READMEs asserting otherwise. The lesson is uncomfortable and
worth keeping: *"pinned by test" is itself a claim that needs verifying.*

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

## The concurrency test, and two failed attempts at it

`test_pipeline_concurrency.py` guards the bug that aborted a shard after 3.25 M
successfully embedded chunks:

```
ERROR chunker: waited >900s for input
```

Completion was signalled by sentinels pushed through the **bounded** queues:

```python
try: self.raw_q.put(SENTINEL, timeout=30)
except queue.Full: pass          # <- silently drops the shutdown signal
```

**My first two attempts at testing this both passed on known-broken code.**

*Attempt 1* used a merely-slow consumer (20 ms/batch). The dropped sentinel needs
`raw_q` to stay full for longer than the 30 s `put` timeout at the exact moment
the reader finishes; a fast-draining queue never gets there.

*Attempt 2* gated the consumer to force that stall — and the setup assertion
failed, revealing the state is **unreachable by that route**: a fully-gated
consumer blocks the reader before it can finish, so `reader_done` never fires.

Chasing the timing was the wrong instinct. The invariant is what matters:

> **A completion signal must not travel through a channel that can drop it.**

A bounded queue can always reject a `put` under backpressure, so sentinel-based
shutdown is broken *by construction* — whether or not a given run happens to hit
the timeout. `test_completion_is_never_signalled_through_a_bounded_queue` spies
on both queues and asserts no sentinel is ever enqueued. It is deterministic, it
cannot flake, and mutation-checking confirms **9 tests fire** on the old design:

```
FAILED test_completion_is_never_signalled_through_a_bounded_queue
  AssertionError: completion was signalled through bounded queue(s)
                  ['embed_q', 'raw_q']
FAILED test_survives_a_long_consumer_stall
FAILED test_terminates_under_severe_backpressure
FAILED test_terminates_with_more_chunkers_than_work
FAILED test_terminates_across_chunker_counts[1|2|4]
FAILED test_no_chunks_are_lost_under_backpressure
FAILED test_empty_shard_terminates_cleanly
```

The lesson generalises: **when a race is hard to reproduce, test the property
that makes it impossible rather than the timing that makes it visible.**

A deadlocked pipeline burns 0% CPU and prints nothing, so every test here runs
`run()` on a daemon thread and joins with a timeout — a hang fails with a
diagnostic instead of wedging CI.

---

## Layer 2 — DeepEval quality gates (`-m llm`)

Deterministic tests cannot tell you whether an *answer* is any good. That needs
a judge — and a judge needs calibrating before you trust it. This layer lives in
`test_rag_deepeval.py` (judge in `ollama_judge.py`) and is the only part of the
suite that needs a running Ollama.

### It found a real bug on the first run

```
answer:  "Vision frames are retained for 14 days hot and indefinitely cold [1]."
corpus:  | Vision frames | 14 days | none | 14 days |
```

Cold storage is **none**. The model invented indefinite cold retention — and
cited block `[1]`, which genuinely *was* sent, so `verify_citations()` passed it.
**Mechanical verification confirms a citation points at a real block; only a
judge can tell you the sentence misrepresents that block.** That single find
justifies the layer.

Fixed by a system-prompt rule ("never turn `none` into a duration"), and then
**pinned deterministically** by `test_no_invented_cold_retention` — once a
specific failure is known, checking it exactly costs nothing and needs no judge.

### But the judge is an unreliable instrument, and that's measurable

Scores on the real pipeline with a local `qwen2.5:7b`:

| question | faithfulness | relevancy | G-Eval citation |
|---|---:|---:|---:|
| Rotterdam rule (correct, cited `[2]`) | 1.00 | **0.00** | **0.00** |
| TLM-330 (correct, cited `[1]`) | **0.50** | 0.67 | 0.90 |
| barcode 0.92 (correct, cited `[1]`) | 1.00 | 1.00 | **0.00** |
| vision frames (**hallucinated**) | **0.33** | 0.25 | 1.00 |

Two judge failure modes, both verified by hand against the corpus:

- **Partial context use.** The TLM-330 answer is fully supported by block `[1]`;
  blocks `[2]`–`[4]` are retrieval noise. The judge penalised the answer for
  correctly *ignoring* them.
- **Negation.** After the fix the answer says frames are "**not** retained cold".
  The judge read that as "retained cold" and called it a contradiction.

And the G-Eval citation metric scored **0.00** on answers visibly containing
`[2]` and `[1]`, while scoring **1.00** on another that also contains `[1]` —
self-contradictory, so it measures nothing.

### What the suite does about it

1. **Calibration test runs first.** `test_judge_is_calibrated_for_faithfulness`
   checks the judge separates a known-good from a known-bad answer by ≥ 0.5. If
   it fails, the gate's verdicts are meaningless and the *judge* is what needs
   fixing.
2. **Faithfulness is the only gate**, at a threshold **derived from measurement**
   — known hallucination 0.33, known-correct answers 0.50–1.00, so the floor sits
   at 0.40. Not aspiration: 0.7 produced 2 false positives out of 4, and a gate
   that fails correct work gets switched off within a week.
3. **Relevancy and contextual precision are advisory** — printed every run, never
   failing.
4. **The G-Eval citation metric was deleted.** Its property is already verified
   exactly and for free. *An unreliable LLM gate over a mechanically checkable
   property is strictly worse than the mechanical check alone.*

The 0.17 margin between 0.33 and 0.50 is thin, and that is a statement about the
7B judge rather than the pipeline. Set `DEEPEVAL_JUDGE=llama3.1:70b` (or a
frontier API model) and re-run calibration to justify a stricter gate.

### Local judge, no API key

`tests/ollama_judge.py` returns DeepEval's built-in `OllamaModel`. I first wrote
a custom `DeepEvalBaseLLM` subclass before finding the built-in — it is kept as a
reference for wrapping runtimes DeepEval doesn't support (vLLM, TGI), with the
gotcha it taught: it must **subclass** `DeepEvalBaseLLM`, because DeepEval does an
`isinstance` check and duck typing fails with
`TypeError: Unsupported type for model`.

---

## Markers

| marker | meaning |
|---|---|
| `llm` | needs a running Ollama server |
| `gpu` | needs CUDA |
| `slow` | more than a few seconds |
| `integration` | exercises several projects together |

CI should run `-m "not llm and not gpu"` — no infrastructure needed. Add
`and not slow` for the 0.4 s pre-commit subset.

---

## What is NOT covered yet

Stated plainly, because a test README that implies full coverage is its own kind
of lie:

- **The OAuth 2.1 server** (`05_mcp_server`) is verified end-to-end by
  `auth_client_demo.py` — including PKCE rejection, refresh rotation and
  revocation — but that is a script, not pytest. It should be ported.
- **Training loops** (projects 02/03/04) have no tests. Their `evaluate.py`
  harnesses measure output quality but assert nothing.

`scale/pipeline.py` was the biggest gap and is now covered — see above.
