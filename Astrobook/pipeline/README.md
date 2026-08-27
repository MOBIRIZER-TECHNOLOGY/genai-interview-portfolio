# Astro_book → LoRA/QLoRA runbook

Turns 44 text-layer PDFs (~3M tokens of Jyotisha literature) into a tuned
Q&A tutor adapter.

**Read this first.** LoRA teaches style, format, and idiom. It does *not*
reliably install facts. Tuned on this corpus alone, the model will produce
fluent, correctly-shaped, confidently-wrong Jyotisha — it learns to *sound* like
Mantreswara without retaining what Mantreswara wrote. The setup that works for a
reference corpus is **LoRA for voice + RAG for facts**: the adapter learns to
reason in houses/lords/dashas/yogas, retrieval over the same 44 PDFs supplies
the citable rules at inference. `chunks.jsonl` from stage 1 feeds both, so the
extraction work is shared. Build the LoRA — just build it knowing that's its job.

---

## Where each stage runs

| Stage | What | Where | Cost |
|---|---|---|---|
| 1 `01_extract.py` | PDFs → `chunks.jsonl` | this machine, CPU | free |
| 2 `02_generate.py` | chunks → instruction pairs | Claude Batch API | ~$55 Opus / ~$22 Sonnet |
| 3 `03_split.py` | pairs → train/val/test | this machine, CPU | free |
| 4 `04_train.py` | QLoRA fine-tune | **Kaggle T4** | free (30 hr/wk) |
| 5 `05_eval.py` | A/B vs base, judged | Kaggle + Batch API | ~$3 |

This machine (Intel UHD, no CUDA) cannot train. Stage 4 is Kaggle or Colab.

---

## Stage 1 — extract (local)

```bash
pip install -r pipeline/requirements-local.txt
python pipeline/01_extract.py --report
```

Skips the 8 unusable PDFs (4 pure scans, the 3 Lal Kitab 1952 volumes whose only
extractable text is a distributor watermark, and lal-kitab-1941 whose Hindi
extracts as mojibake). See `SKIP` in `config.py` for why each one is out.

**Check the `--report` histogram before continuing.** `pinned at hard cap`
should be near zero. A large number means the verse/paragraph/sentence cascade
found no boundaries and fell back to slicing, which produces chunks that start
and end mid-thought and generate bad training pairs. An earlier version of this
script had exactly that bug at 54%.

Also eyeball a few chunks — diacritics (`Rāśi`, `Śukr`) should survive, verse
numbers should be attached to their commentary, no running heads.

## Stage 2 — generate pairs (Batch API)

```bash
export ANTHROPIC_API_KEY=...        # or: ant auth login
python pipeline/02_generate.py submit --limit 40      # ~$2 smoke test
python pipeline/02_generate.py collect
```

**Read the 40-chunk output before spending the rest.** The prompt is the single
biggest lever on final quality; a bad one is far cheaper to catch here. Look for:
questions that stand alone (no "in the passage above"), answers that name their
source text, doctrine framing rather than personal prediction.

Then the full run:

```bash
python pipeline/02_generate.py submit          # whole corpus
python pipeline/02_generate.py collect         # poll ~1h, writes pairs.jsonl
```

Defaults to `claude-opus-5`. `--model claude-sonnet-5` is ~3x cheaper with
noticeably blander answers. A reasonable split: Opus on a 200-chunk seed to set
the voice, Sonnet for the bulk.

## Stage 3 — split (local)

```bash
python pipeline/03_split.py
```

Splits **by source book**, not by row. Chunks from one book share vocabulary and
translator idiom; a random row split leaks that across the boundary and your
eval reports a number the adapter didn't earn.

## Stage 4 — train (Kaggle T4)

New notebook → Settings → Accelerator **GPU T4 x2**, Internet **On**. Upload
`build/train.jsonl` + `build/val.jsonl` as a Dataset, and `pipeline/` as another.

```python
!pip install -q unsloth unsloth_zoo bitsandbytes peft trl transformers datasets accelerate
!python pipeline/04_train.py --build /kaggle/input/astro-splits --max-steps 30
```

The 30-step smoke test takes ~5 min and catches every path/format/OOM problem
before you commit hours. Then:

```python
!python pipeline/04_train.py --build /kaggle/input/astro-splits --out /kaggle/working/astro-lora
```

Config and why:

| Setting | Value | Reason |
|---|---|---|
| base | Qwen3-4B-Instruct | 8B QLoRA is ~3 hr/epoch on one T4 — 3 epochs won't reliably finish in Kaggle's 12 hr session. 4B gives you room to iterate, which matters more than base size for a first LoRA. |
| `--qlora` | on | 4-bit NF4 base. Purely a VRAM decision: 16 GB T4 can't hold a 4B fp16 base + optimiser. On a 48 GB A40, `--no-qlora` runs the same recipe in bf16, meaningfully faster. |
| rank | 32 | Not the usual 8 — you're teaching a domain idiom, not a response format. |
| target modules | all 7 linear | Matters more than the rank. q/v-only is the common under-configuration. |
| seq len | 1024 | Pairs average ~500 tok. 2048 + padding burns most of your compute on nothing. `group_by_length` handles the rest. |
| epochs | 2 | At ~10k pairs you overfit past 3. The tell: the model reciting corpus sentences verbatim. |
| loss | completion only | Otherwise capacity goes into reproducing your system prompt. |

Unsloth is roughly 2x faster at ~40% less VRAM and explicitly supports Turing.
Free Unsloth is single-GPU, so Kaggle's second T4 idles — take the trade, and use
the spare card for a parallel experiment.

**T4 is Turing**: no bf16, and FlashAttention-2 needs Ampere+. The script detects
this and sets fp16. Don't "fix" that flag or add `attn_implementation="flash_attention_2"`.

Download `astro-lora/` (adapter only, ~200-400 MB) before the session expires.

## Stage 5 — eval

```python
# Kaggle, same session
!python pipeline/05_eval.py generate --adapter /kaggle/working/astro-lora --n 150
```
```bash
# anywhere with an API key
python pipeline/05_eval.py judge
```

Grades **faithfulness, not correctness** — there's no ground truth for "correct
Jyotisha", but every test pair carries the `chunk_id` it came from, so the source
passage is a real reference. Scores support / fabrication / citation / framing,
and runs a position-randomised A/B against the untuned base. That comparison is
the only one that tells you the adapter earned its keep.

If tuned doesn't beat base, check stage-2 groundedness first. Bad data, not bad
hyperparameters, is the usual cause.

---

## Known gaps

- **8 PDFs excluded** (~1,780 pages): 4 pure scans, the 3 Lal Kitab 1952
  volumes, and `lal-kitab-1941.pdf` (real text layer, but legacy non-Unicode
  Hindi font → mojibake). The whole Lal Kitab tradition is therefore absent
  from v1. Recovering it needs OCR plus font remapping, not a decoder.
- **No RAG index yet.** `chunks.jsonl` is the right input for one.
- **~3M tokens is a small corpus.** Calibrate expectations accordingly.
