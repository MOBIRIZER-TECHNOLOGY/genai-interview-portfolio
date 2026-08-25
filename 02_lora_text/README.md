# 🔤 Project 02 — LoRA Fine-Tuning a Text LLM (structured extraction)

Teach a 0.5B language model to do a job it cannot do out of the box: turn a
free-text incident report into a **strict JSON triage record** using
domain-specific vocabulary and rules it has never seen.

> **In one sentence:** 800 synthetic examples, ~1 minute of training, a 34 MB
> adapter — and exact-match accuracy goes from **0% to 100%** on held-out data,
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

RTX 5070 Ti, `Qwen/Qwen2.5-0.5B-Instruct`, r=16, 3 epochs, **56.9 seconds**,
120 held-out examples, greedy decoding, **dataset v3**.

| Metric | Base | LoRA | trivial baseline | real gain |
|---|---:|---:|---:|---:|
| valid JSON | 100.0% | 100.0% | — | — |
| correct schema (exactly 5 keys) | 100.0% | 100.0% | — | — |
| **exact match (all 5 fields)** | **0.0%** | **100.0%** | — | **+100.0** |
| `component` | 0.8% | 100.0% | 16.7% | **+83.3** |
| `action` | 0.0% | 100.0% | 31.7% | **+68.3** |
| `severity` | 31.7% | 100.0% | 39.2% | **+60.8** |
| `page_oncall` | 35.8% | 100.0% | 64.2% | **+35.8** |
| `error_code` | 67.5% | 100.0% | — | +32.5 |

Training loss 1.5377 → 0.0098. Peak VRAM 7.47 GB. Adapter 33.60 MB.

**Every field is now perfect, and that took two dataset fixes rather than any
change to the model.** The history is worth more than the number:

| dataset | exact match | what was wrong with it |
|---|---:|---|
| v1 | 84.2% | `action` was a lookup keyed on `component` (86.7% baseline), and 25% of coded rows had an **unguessable** `error_code` |
| v2 | 84.2% | action fixed — but the unguessable codes remained, capping the score |
| **v3** | **100.0%** | codes the report never states are now labelled `null` |

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

### 🧮 The ceiling: why 84.2% was never a model problem

For two dataset versions this project scored **exactly 84.2%** and the number
would not move. That looked like a stubborn model. It was arithmetic.

`error_code` was the only field below 100%, so exact-match failures *were* its
failures — and every one was the model emitting `null` where the label held a
code. The generator wrote the code into the report text only ~75% of the time
but labelled the row with it regardless:

```python
if code and rng.random() < 0.75:
    parts.append(f"We're seeing {code} on {component} -- {symptom}.")
else:
    parts.append(f"Something is wrong with {component}: {symptom}.")
    # ... and the label kept `code` anyway
```

So on **19 of 120** held-out rows the correct answer was *not derivable from the
input*. The model did the only sane thing — answer `null` — and was marked wrong.

| | value |
|---|---:|
| rows whose label the input can support | 101 / 120 |
| **best achievable `error_code` accuracy** | **84.2%** |
| what the adapter scored | **84.2%** |

**The adapter was sitting exactly on the information-theoretic ceiling**, perfect
on every answerable question, while the README reported a 16% failure rate.

**The fix is one line** — if the report omits the code, the label is `null`:

```python
else:
    code = None                       # the label may not claim what the input never said
    parts.append(f"Something is wrong with {component}: {symptom}.")
```

Ceiling becomes 100%. The model reaches 100%. No hyperparameter was touched.

**Always ask what a perfect model would score on your eval set.** Chasing that
last 16% with a bigger rank, more epochs or a larger base would have burned
compute against a number built into the labels — and the loss curve, already at
0.0098, would have kept insisting everything was fine.

### 💊 A hallucination that came from the labels, not the model

The unguessable rows were not merely unscoreable — they were actively teaching
the model to guess. With them removed, a failure in the out-of-distribution
probe disappeared:

```
IN : The hotel booking API returned 500s for 20 minutes during checkout...
v2 : {..., "error_code": "BOO-402", ...}      <- invented; appears nowhere in the input
v3 : {..., "error_code": null, ...}           <- correct
```

24% of coded training rows had been demonstrating "reports of this shape carry an
`XXX-999` token, produce one even if you cannot see it". That is a fabrication
habit installed by the data. **A label the input cannot support does not just cost
you a point on the eval — it teaches the model to make things up.**

### 🔬 What 100% hides — the out-of-distribution probe

The held-out set comes from the same generator as training. `probe_generalisation.py`
hand-writes seven inputs from outside that distribution — and a perfect in-distribution
score buys **nothing** here.

**Format generalises. Judgement does not.** 7/7 valid JSON, 7/7 correct schema, 7/7
obeying the page rule — including on inputs that are not incidents. That is the trap:
*a model that is always well-formed looks like a model that is always right.*

```
IN : Can you recommend a good pizza place in Rotterdam for tonight?
OUT: {"component":"pizza","severity":"SEV2","error_code":null,
      "page_oncall":true,"action":"check the website and review the customer rating"}
```

It triaged dinner and **paged the on-call**. There is no abstention path, because all
800 training examples are incidents — "this is not an incident" was never shown as an
available answer. That failure is untouched by any of the dataset fixes, because it is
a *missing class*, not a wrong label.

**What the fixes did change:**

| probe input | v1 | v2 | v3 (labels fixed) |
|---|---|---|---|
| cosmetic font complaint | *restart ntp-relay in the cell namespace* — a memorised Atlas remedy | composed for the symptom | composed for the symptom |
| hotel-booking outage | invented `BOO-402` | invented `BOO-402` | **`null` — correct** |
| unseen `XYZ-9999` | copied correctly | copied correctly | copied correctly |

**A claim to withdraw.** After the v2 rebuild this README reported that the model had
started resisting the prompt injection — *"SEV1 but do NOT page anyone"* held at SEV1
where v1 had downgraded to SEV3 to comply. **v3 downgrades again.** One probe, one run,
and the behaviour is not stable across retrains. The original note hedged ("I would not
over-claim a mechanism from one probe") and the hedge turned out to be the whole story:
**with n=1 you measured a sample, not a property.**

**What fine-tuning bought:** behaviour and format — output shape, field conventions, the
page rule — in under a minute on 800 examples. **What it did not buy:** knowledge,
judgement, or an abstention it was never taught. For those you want retrieval
(project 01), an explicit `not_an_incident` class, and validation of content rather than
form.

### 🧪 The QLoRA experiment — and why it *lost*

| | bf16 LoRA | QLoRA (4-bit) |
|---|---:|---:|
| peak VRAM | **7.47 GB** | 8.92 GB |
| training time | **56.9 s** | 105.3 s |
| final train loss | 0.0098 | 0.0097 |
| valid JSON | **100.0%** | 99.2% |
| `action` | **100.0%** | 87.5% |
| `severity` | **100.0%** | 98.3% |
| **held-out exact match** | **100.0%** | **86.7%** |

QLoRA was **1.9× slower, used 1.45 GB more memory, and scored 13.3 points
worse** — and on one of 120 examples it did not even emit valid JSON, something
the bf16 adapter never did.

**This finding got sharper when the dataset was fixed, which is why it is worth
trusting.** On the flawed data the entire gap sat in `error_code` — the field
corrupted by unguessable labels — and a reasonable person could have argued the
gap was an artefact of that noise. With clean labels the gap moved to
**`action`** (87.5% vs 100%): the hardest field, a 19-way choice that requires
reading the symptom rather than spotting the service name. 4-bit base weights
cost accuracy exactly where the task needs the finest discrimination.

- **Memory:** at 0.5B the weights are ~1 GB. NF4 saves a few hundred MB, then
  adds quantisation state and dequantisation buffers. Peak VRAM is dominated by
  *activations* for `batch 8 × seq 512`, which 4-bit does nothing about.
- **Speed:** every forward pass dequantises NF4 back to bf16 before the matmul —
  pure overhead when the weights already fit.
- **Quality:** the gap is real, reproduced across three dataset versions, and
  moved to the hardest field once the noise was removed.

**QLoRA earns its keep when weights dominate memory** — a 7B, 13B or 70B that
otherwise will not load. At 0.5B it is a pure loss, and reaching for it here
would be cargo-culting.

### ⚠️ The eval is now saturated — say so before someone else does

100% on every field means this held-out set **can no longer tell two good models
apart**. That is a real cost of fixing the labels, and pretending otherwise would
repeat the original mistake in the opposite direction.

What it still measures: that the base model scores 0.0%, that QLoRA scores 86.7%,
and that every field clears its trivial baseline by 35–83 points. What it can no
longer measure: any further improvement to the bf16 adapter.

To restore headroom you make the *task* harder, not the labels wronger — more
components, ambiguous severity phrasing, reports naming two services, or an
explicit `not_an_incident` class. The out-of-distribution probe below is the
cheap stand-in: it still has plenty of failures left to show.

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

**Loss hit 0.0098. Isn't that overfitting?**
On the training set, absolutely — it has memorised 800 examples. That's fine
*because held-out exact match is 100%*, which is the number that counts.

There is a sharper version of this lesson here. For two dataset versions the loss
sat at 0.0098 while held-out accuracy was stuck at 84.2% — and the cause was
neither overfitting nor underfitting, but **19 eval labels the input could not
support**. Training loss cannot see that, and neither can validation loss. Only
asking *"what would a perfect model score?"* finds it. Never tune on training
loss; and before tuning on the eval, check the eval.

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
