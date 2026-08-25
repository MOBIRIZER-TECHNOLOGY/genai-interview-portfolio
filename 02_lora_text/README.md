# 🔤 Project 02 — LoRA Fine-Tuning a Text LLM (structured extraction)

Teach a 0.5B language model to do a job it cannot do out of the box: turn a
free-text incident report into a **strict JSON triage record** using
domain-specific vocabulary and rules it has never seen.

> **In one sentence:** 800 synthetic examples, 50 seconds of training, a 34 MB
> adapter — and exact-match accuracy goes from **0% to 84.2%** on held-out data,
> with every field measured against the score a lookup table would get.

---

## 🧠 The idea (for non-experts)

A pretrained LLM is a generalist. It has read the internet, so it can write JSON
and it can guess what "SEV1" probably means — but it does not know **your**
company's components, your error codes, or your rule that "page on-call only for
SEV1 and SEV2".

You have three ways to fix that:

| Approach | Teaches it… | Cost |
|---|---|---|
| **Prompting** | nothing new; just steers what it already knows | free, but limited |
| **RAG** (project 01) | *facts* it can look up | index + retrieval latency |
| **Fine-tuning** (this) | *behaviour and format* | one training run |

**LoRA** (Low-Rank Adaptation) makes the third one cheap. Instead of updating all
500 million weights, you freeze them and train two tiny matrices per layer whose
product nudges the output. Here that's **8.8 M trainable parameters — 1.75% of the
model** — and the saved result is a 34 MB file you can share, version, and swap.

**The judgement call worth internalising:** this task is a *good* fit for
fine-tuning precisely because it needs no new facts and no new reasoning. It
needs consistent format and domain conventions. If your task needs the model to
know something, reach for RAG instead. Picking the wrong one of these two is the
most common architectural mistake in applied GenAI.

---

## ✅ Proof it works (measured on this machine)

RTX 5070 Ti, `Qwen/Qwen2.5-0.5B-Instruct`, r=16, 3 epochs, **49.6 seconds**,
120 held-out examples, greedy decoding, **dataset v2**.

| Metric | Base | LoRA | trivial baseline | real gain |
|---|---:|---:|---:|---:|
| valid JSON | 100.0% | 100.0% | — | — |
| correct schema (exactly 5 keys) | 100.0% | 100.0% | — | — |
| **exact match (all 5 fields)** | **0.0%** | **84.2%** | — | **+84.2** |
| `component` | 0.8% | 100.0% | 16.7% | **+83.3** |
| `severity` | 31.7% | 100.0% | 39.2% | **+60.8** |
| `action` | 0.0% | 100.0% | 31.7% | **+68.3** |
| `page_oncall` | 35.8% | 100.0% | 64.2% | +35.8 |
| `error_code` | 51.7% | 84.2% | — | +32.5 |

Training loss: 1.5377 → 0.0098. Peak VRAM 7.47 GB. Adapter 33.60 MB.

**The "trivial baseline" column is the one to read**, and it exists because the
first version of this dataset didn't have it. See
[the dataset section](#-the-training-data-in-full--and-the-baseline-each-field-deserves):
`action` used to score 95.8% against an 86.7% lookup baseline — nine points of
real learning dressed up as a strong number. The dataset was rebuilt so actions
depend on the *symptom* rather than the *component*; the baseline fell to 31.7%
and the model now scores 100%. Same headline, an honest number underneath it.

### 🔍 Read this table properly — it's the interesting part

**Fine-tuning did not fix JSON validity. It was already 100%.** Qwen2.5-Instruct
is perfectly capable of emitting well-formed JSON with the right keys. If you'd
only measured "does it produce JSON", you would have concluded fine-tuning was
unnecessary and shipped a system that is wrong 100% of the time.

What it fixed was **domain conformance**:

```
PROMPT: ... Something is wrong with atlas-sim: scenario battery-cliff.yaml times out.
        The safety system is offline.

BASE:   {"component": "Atlas-Sim",  "severity": "SEV1", "page_oncall": false,
         "action": "Contact the owner of the Atlas-Sim instance to resolve the issue"}
LORA:   {"component": "atlas-sim",  "severity": "SEV1", "page_oncall": true,
         "action": "raise the scenario timeout and profile the battery model"}
```

The base model title-cases the component (`Atlas-Sim`), invents a generic action,
and gets the paging rule backwards. Every one of those is a *convention*, not a
fact — which is exactly what fine-tuning is for.

### 📋 The training data, in full — and the baseline each field deserves

`python inspect_dataset.py` prints all of this. Read it before the accuracy
table, because it is what makes those numbers mean anything.

**800 training examples, 120 held-out, all synthetic from `make_dataset.py`.**
Every operator report is textually unique (800/800). The label space:

| | inventory |
|---|---|
| components | 5 — `atlas-console`, `-dispatch`, `-sim`, `-telemetry`, `-vision` |
| severities | 3 — SEV1 / SEV2 / SEV3, roughly balanced |
| error codes | 10 real codes (`TLM-330`, `VIS-207`, …) + `null` on ~36% of rows |
| **actions** | **19 distinct, keyed on the symptom** |

#### The v1 mistake, kept on the record

The first version keyed `action` on the **component**, and four of five
components had exactly one action in the whole dataset. So `action` was a lookup:

| | v1 | v2 |
|---|---:|---:|
| distinct actions | 9 | **19** |
| actions for `atlas-console` | 1 | 3 |
| actions for `atlas-dispatch` | 1 | 4 |
| actions for `atlas-sim` | 1 | 3 |
| actions for `atlas-vision` | 1 | 4 |
| **"copy the component's majority action" scores** | **86.7%** | **31.7%** |
| model scores | 95.8% | **100.0%** |
| **real gain over the baseline** | **+9.1** | **+68.3** |

`data/` is generated and gitignored, so the v1 dataset is not shipped — but the
**v1 generator is in git history**, one commit before the rebuild. Anyone who
wants to check the v1 numbers above can regenerate that dataset from it. In v1
the model beat a lookup table by nine points on a field advertised at 95.8%. In v2 the same field requires reading the symptom — "barcode read rate
has dropped to 42%" gets *recalibrate the scanners*, "gantry 7 GPU has fallen
off the bus" gets *power cycle the gantry*, and both are `atlas-vision` — and
the model gets it perfect.

**Making the task harder made the result better**, which is the opposite of
what tuning a number usually does, and the reason to measure baselines at all.
Every field now clears its trivial predictor by a wide margin:

| field | baseline | model | gain |
|---|---:|---:|---:|
| `component` | 16.7% | 100.0% | **+83.3** |
| `action` | 31.7% | 100.0% | **+68.3** |
| `severity` | 39.2% | 100.0% | **+60.8** |
| `page_oncall` | 64.2% | 100.0% | +35.8 |

A per-field accuracy is meaningless until you know what a stupid predictor
scores on the same field — and that is a property of your **dataset**, not your
model. The one field with no sensible baseline, `error_code`, is now the only
thing standing between this model and a perfect score.

### 🔬 What 84% hides — the out-of-distribution probe

The held-out set comes from the same generator as training: same phrasing, same
components, always a real incident. `probe_generalisation.py` hand-writes seven
inputs from outside that distribution.

**Format generalises. Judgement does not.** 7/7 valid JSON, 7/7 correct schema,
7/7 obeying the page rule — including on inputs that are not incidents. That is
the trap: *a model that is always well-formed looks like a model that is always
right.* The failures are entirely semantic:

```
IN : Can you recommend a good pizza place in Rotterdam for tonight?
OUT: {"component":"pizza_app","severity":"SEV1","error_code":null,
      "page_oncall":true,"action":"search the online reviews and compare prices"}
```

It triaged dinner as a **SEV1 and paged the on-call**. There is no abstention
path, because all 800 training examples are incidents — "this is not an
incident" was never shown as an available answer.

```
IN : The hotel booking API returned 500s for 20 minutes during checkout...
OUT: {..., "error_code":"BOO-402", "action":"page the user manual and check network"}
```

**It fabricated an error code.** `BOO-402` is nowhere in the input; the model
learned that reports of this shape carry an `XXX-999` code and invented one to
fit. The action is a garbled blend of two it was taught.

#### Two failures the dataset rebuild fixed

Worth recording, because they show what was a *data* problem rather than a model
one:

| probe input | v1 output | v2 output |
|---|---|---|
| cosmetic font complaint | action: *restart ntp-relay in the cell namespace* — a memorised Atlas remedy | action: *flush the dashboard font cache and check browser style sheet* — **composed for the symptom** |
| "SEV1 but do NOT page anyone" | **downgraded to SEV3** so not-paging was consistent — an injected instruction moved a safety-relevant field | **held SEV1, `page_oncall` true** — kept the rule and ignored the injection |

The prompt-injection case is the striking one. In v1 the model bent the severity
to satisfy the user's instruction; in v2, trained on the same rule with a richer
action space, it refused. I would not over-claim a mechanism from one probe —
but a model that has to read symptoms to answer appears to lean less on the
prompt's framing.

**What remains true in both versions:** no abstention, fabricated error codes on
unknown domains, and confident triage of things that are not incidents. That is
the honest boundary of what fine-tuning bought — *behaviour and format, not
knowledge or judgement.* For those you want retrieval (project 01), an explicit
`not_an_incident` class in the data, and validation of content rather than form.

Run `python probe_generalisation.py` to reproduce all seven.

### 🧪 The QLoRA experiment — and why it *lost*

| | bf16 LoRA | QLoRA (4-bit) |
|---|---:|---:|
| peak VRAM | **7.47 GB** | 8.92 GB |
| training time | **49.6 s** | 105.3 s |
| final train loss | 0.0098 | 0.0097 |
| **held-out exact match** | **84.2%** | 78.3% |

QLoRA was **2.1× slower, used 1.45 GB more memory, and scored 5.9 points
worse.** That is not a bug — it is what QLoRA does at this model size:

- **Memory:** at 0.5B the weights are ~1 GB. Quantising them to NF4 saves a few
  hundred MB, then adds quantisation state and dequantisation buffers on top.
  Peak VRAM is dominated by *activations* for `batch 8 × seq 512`, which 4-bit
  does nothing about.
- **Speed:** every forward pass dequantises NF4 back to bf16 before the matmul.
  That is pure overhead when the weights already fit.
- **Quality:** the gap is **entirely** in `error_code` (84.2% → 78.3%), the field
  needing the finest discrimination. All four other fields are 100% in both.

**QLoRA earns its keep when weights dominate memory** — a 7B, 13B or 70B where
the base model is the thing that doesn't fit. At 0.5B it is a pure loss, and
reaching for it here would be cargo-culting.

### 🔁 Reproducibility, and how these numbers should be quoted

Before the dataset was rebuilt, the v1 model was retrained and re-evaluated from
scratch months after its original run. Peak VRAM (7.47 GB / 8.91 GB) and adapter
size (33.60 MB) reproduced **exactly**; accuracy moved by one held-out example on
bf16 (84.2% → 83.3%) and two on QLoRA (74.2% → 72.5%).

**A seed does not make CUDA deterministic.** It fixes initialisation and sampling
order, not reduction order in cuBLAS or atomic accumulation. Loss curves diverge
in the fourth decimal by step 40, and examples near a decision boundary flip. So
every accuracy here should be read as **± an example or two**, and a figure
quoted to one decimal from a single run implies a precision the hardware does not
give you.

The reassuring half: **mechanisms reproduce better than numbers.** In both v1
runs the entire bf16-vs-QLoRA gap sat in `error_code` while the other fields
matched to the decimal — and after a complete dataset rebuild, it still does.

## 📁 What's in this project

```
02_lora_text/
├── make_dataset.py       synthetic incident reports -> JSON labels
├── train_lora.py         explicit LoRA training loop (bf16 or 4-bit QLoRA)
├── inspect_dataset.py    dataset inventory + trivial baselines per field
├── probe_generalisation.py  out-of-distribution probe (7 hand-written edges)
├── evaluate.py           base vs LoRA on held-out data
├── merge_and_export.py   fold the adapter into the weights, Ollama Modelfile
├── data/
│   ├── train.jsonl       800 examples
│   └── eval.jsonl        120 held-out (separate RNG stream)
├── lora-out/             the trained adapter (34 MB) + training_info.json
├── lora-out-4bit/        the QLoRA variant, for the VRAM comparison
└── eval_results.json
```

```
make_dataset.py ──▶ data/           (reports + gold JSON)
                       │
train_lora.py   ──▶ lora-out/       (34 MB adapter)
                       │
evaluate.py     ──▶ base vs LoRA table
                       │
merge_and_export.py ──▶ merged/     (standalone model, optional)
```

---

## 🚀 How to run it

```powershell
..\activate.ps1                        # see ../SETUP.md

python make_dataset.py                 # 800 train + 120 eval, instant
python train_lora.py --epochs 3        # ~60 s on a 5070 Ti
python evaluate.py --show 3            # base vs LoRA, plus sample outputs
```

Optional:

```powershell
python train_lora.py --epochs 3 --load-4bit --output lora-out-4bit   # QLoRA
python merge_and_export.py --test                                    # fold in the adapter
python merge_and_export.py --ollama                                  # + Modelfile
```

---

## 🔧 Use your OWN data

Write JSONL where each line is `{"messages": [system, user, assistant]}`:

```json
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Then `python train_lora.py --data your_dir`. Three things to get right:

1. **Hold out a real eval set** before you train, from a separate RNG stream or a
   separate time period. An eval set that shifts when you change a training flag
   tells you nothing.
2. **Be consistent in the target format.** The model is learning a format; if
   half your labels use `{"a": 1}` and half use `{"a":1}`, you're teaching it
   ambiguity.
3. **200–1000 examples is usually plenty** for a format/behaviour task. If you
   need 50k, you're probably trying to teach facts, and you want RAG.

---

## ⚙️ The knobs that actually matter

| Flag | What it does | Guidance |
|---|---|---|
| `--rank` | Adapter capacity. Params scale linearly. | 8–16 for format tasks, 32–64 for style/tone, 64+ if you're genuinely teaching a new skill |
| `--alpha` | Adapter scale, applied as `alpha/rank` | Default is `2*rank`. Keeping the ratio fixed means a rank sweep measures *capacity*, not learning rate |
| `--lr` | Learning rate | `1e-4`–`3e-4`. LoRA takes a much higher LR than full fine-tuning because you're training few parameters from a zero-init |
| `--epochs` | Passes over the data | 2–3. Watch for loss going flat, then stop — at 0.0097 we're memorising |
| `--load-4bit` | QLoRA: base weights in NF4 | Trades speed for VRAM. Use when the model doesn't otherwise fit |
| `--gradient-checkpointing` | Recompute activations in the backward pass | ~30% slower, big activation-memory saving. The knob to reach for at long sequence lengths |
| `--batch-size` × `--grad-accum` | Effective batch | Activation memory scales with batch × seqlen. Accumulate to keep the effective batch when you have to shrink the real one |

---

## 🖥️ Tech stack

- **Base:** `Qwen/Qwen2.5-0.5B-Instruct` (494 M params)
- **Method:** LoRA on `q,k,v,o` + `gate,up,down`, r=16, α=32, dropout 0.05
- **Precision:** bf16 (and NF4 double-quant for the QLoRA run)
- **Loss:** completion-only — prompt tokens masked to `-100`
- **Libraries:** PyTorch 2.11+cu128, 🤗 Transformers 5.15, PEFT 0.20, bitsandbytes
- **Validated on:** RTX 5070 Ti 16 GB, Python 3.12

---

## ❓ FAQ

**Why write the training loop by hand instead of using `SFTTrainer`?**
Because every line that matters in LoRA is visible: which modules get adapters,
how labels are masked, where the dtype boundary sits. Those are the things an
interviewer asks about and the things a one-liner hides. In production I'd use
the library — the point here is being able to explain it.

**What is "completion-only loss" and why does it matter?**
The prompt tokens are set to `-100` in the labels, so the model is never scored
on predicting the operator's report back to itself. Train on the whole sequence
and a large share of your gradient goes into memorising input text you will never
generate. On a short-output task like this the effect is significant.

**Why did you target the MLP layers and not just attention?**
Attention-only (`q,k,v,o`) is the classic minimal LoRA and it works. Adding
`gate/up/down` roughly doubles the adapter and consistently learns formats
faster — the MLP is where a lot of the "how do I phrase this" behaviour lives.
It's a capacity-vs-size trade, and 34 MB is not a size worth optimising.

**Loss hit 0.0097. Isn't that overfitting?**
On the training set, absolutely — it has memorised 800 examples. That's fine
*because held-out exact match is 84.2%*, which is the number that counts. This
is exactly why you never tune on training loss.

**Why is `alpha = 2 * rank`?**
The adapter output is scaled by `alpha/rank`. Fixing the ratio at 2 means
changing rank changes capacity without also changing the effective step size —
otherwise a rank sweep is secretly a learning-rate sweep and the results are
uninterpretable.

**Should I merge the adapter?**
Merge if you're shipping **one** model (export to GGUF, hand it to another team) —
you save two matmuls per adapted layer at inference. Keep it unmerged if you're
serving **many** variants, because one base model in VRAM can host many hot-swapped
adapters. See the header of `merge_and_export.py`.

**Can I merge a QLoRA adapter into 4-bit weights?**
Not cleanly — quantising the sum is not the sum of the quantised. Load the base
in bf16, merge there, then re-quantise if needed. `merge_and_export.py` does it
that way and says so.

---

## Related projects

- **[03_lora_image](../03_lora_image/)** — the same low-rank trick on a
  diffusion UNet
- **[04_lora_voice](../04_lora_voice/)** — and on an ASR encoder-decoder
- **[05_mcp_server](../05_mcp_server/)** — serves this adapter as a tool
- **[06_local_gpu_inference](../06_local_gpu_inference/)** — why QLoRA lost here,
  measured from the inference side
