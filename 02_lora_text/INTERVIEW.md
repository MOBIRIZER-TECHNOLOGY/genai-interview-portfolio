# 🎤 Interview notes — LoRA / fine-tuning text models

Numbers here come from `lora-out/training_info.json` and `eval_results.json` on
this machine. Re-run before an interview so you're quoting live results.

---

## The 60-second project pitch

> "I fine-tuned Qwen2.5-0.5B to turn free-text incident reports into a strict JSON
> triage schema — domain components, an error-code taxonomy, and a paging rule.
> LoRA on attention plus MLP, rank 16, completion-only loss, 800 synthetic
> examples. 50 seconds on one consumer GPU, 34 MB adapter, 1.75% of parameters
> trained. Held-out exact match went from 0% to 100% — after two dataset fixes
> that mattered far more than any hyperparameter.
>
> The interesting part is *what* it fixed. The base model already emitted valid
> JSON with the right keys — 100% both before and after. What it got wrong was
> the domain conventions: it title-cased component names, invented generic
> remediation text, and inverted the paging rule. So if I'd measured only 'is it
> valid JSON' I'd have concluded fine-tuning was unnecessary and shipped
> something wrong on every single record."

---

## The question you will definitely be asked

### "When do you fine-tune versus use RAG?"

The one-line version: **RAG adds knowledge, fine-tuning changes behaviour.**

| Symptom | Reach for |
|---|---|
| "It doesn't know our Q3 policy" | RAG |
| "It won't stop wrapping JSON in prose" | fine-tuning |
| "It cites the wrong document" | RAG (retrieval) |
| "It's too verbose / wrong tone" | fine-tuning |
| "Facts change weekly" | RAG — retraining weekly is absurd |
| "We need a 0.5B to do what a 70B does, on one narrow task" | fine-tuning (distillation) |

The nuance that gets you credit: **they compose.** The strongest production
pattern is usually fine-tune for format and tone, RAG for facts. This repo does
exactly that — project 01 retrieves, project 02 formats. And the honest failure
mode of fine-tuning-for-knowledge is that the model learns to produce
*confident-sounding* answers in the right shape while the facts rot, which is
worse than not knowing.

### "Explain LoRA. Why does it work?"

Full fine-tuning updates `W`, a `d×k` matrix, for every layer. LoRA freezes `W`
and learns `ΔW = B·A` where `A` is `r×k` and `B` is `d×r`, with `r << d`. At
inference `h = Wx + (α/r)·BAx`.

Why it works: the **intrinsic rank hypothesis** — the update you actually need
when adapting a pretrained model to a downstream task has far lower rank than
the weight matrix itself. You're not teaching the model language, you're
steering an already-competent model into a corner of its behaviour space, and
that steering direction is low-dimensional.

Two implementation details that show you've actually done it:
- **`B` is zero-initialised**, `A` is random. So `BA = 0` at step 0 and the
  adapted model starts *exactly* equal to the base model. No warmup shock.
- **The scale is `α/r`, not `α`.** That's what lets you change rank without
  accidentally changing the effective learning rate.

### "What's QLoRA and when do you use it?"

QLoRA = quantise the **frozen base** to 4-bit NF4, keep the **trainable adapter**
in bf16, and run compute in bf16 by dequantising on the fly.

Three pieces:
- **NF4** — a 4-bit datatype whose levels are placed at the quantiles of a normal
  distribution. Neural network weights are approximately normal, so NF4 loses
  less than uniform 4-bit at the same bit width.
- **Double quantisation** — quantise the quantisation constants too. Saves ~0.4
  bits/param, which matters at scale.
- **Paged optimizers** — spill optimizer state to host RAM on a memory spike
  instead of OOMing.

The rule: **you quantise what you freeze, never what you train.** Gradients
through 4-bit trainable weights would be a disaster.

When to use it: when the model doesn't fit otherwise. It costs speed (dequant on
every forward) to buy VRAM.

**I ran it and it lost, which is the more interesting answer.** Same config,
`--load-4bit`:

| | bf16 LoRA | QLoRA |
|---|---:|---:|
| peak VRAM | 7.47 GB | 8.91 GB |
| time | 58.7 s | 147.0 s |
| held-out exact match | **100.0%** | 86.7% |

Slower, *more* memory, 10 points worse. Why: at 0.5B the base weights are only
0.93 GB, so peak memory is dominated by **activations** (batch 8 × seq 512), not
weights. 4-bit shrinks the part that wasn't the problem, while adding dequant
buffers and the fp32 upcasting `prepare_model_for_kbit_training` inserts. The
quality gap is entirely in `error_code`, the field needing the finest
discrimination.

The takeaway I'd give an interviewer: QLoRA is a **memory** technique, and it
pays only when weights dominate memory — 7B and up. Applying it at 0.5B because
it's the fashionable technique is a strictly worse model for more time. Know
which resource you're actually short of before you optimise for it.

### "Walk me through your hyperparameters."

- **rank 16 / alpha 32.** Format tasks need little capacity. I'd go to 32–64 for
  style or a genuinely new skill. Ratio pinned at 2 so a rank sweep measures
  capacity.
- **lr 2e-4.** An order of magnitude above full fine-tuning (~2e-5). You're
  training 1.75% of the parameters from a zero-init, so you can and must move
  faster.
- **Cosine schedule, 3% warmup.** Warmup matters more than people think with a
  high LR — the first steps into a zero-initialised `B` are unstable.
- **Effective batch 16** (8 × 2 accumulation). Accumulation exists so you keep the
  effective batch when activation memory forces the real batch down.
- **3 epochs.** Loss went 1.54 → 0.0097 and flattened. More would be pure
  overfitting.
- **Dropout 0.05** on the adapter — cheap regularisation on a small dataset.

### "How do you know it worked?"

Held-out exact match, 120 examples, greedy decoding, from a **separate RNG
stream** so changing the training-set size can't shift the eval set.

And I measure per-field, not just overall. That's what surfaced the real
weakness: for two dataset versions `error_code` sat at 84.2% while every other field was 100%. Diagnosing it —
the model over-guesses a code when the report doesn't mention one, because only
~25% of training examples have `null` — points at a *data* fix, more null
examples, not a *training* fix. Training loss is already 0.0097; more epochs
would do nothing but overfit.

Greedy decoding, not sampling. Sampling makes an eval comparison noise.

### "What would go wrong in production?"

1. **Distribution shift.** My data is synthetic and templated. Real operator
   reports have typos, are multilingual, ramble, and mention two components at
   once. The score would drop. The mitigation is to seed with synthetic data, then
   continuously label real traffic — especially the failures.
2. **Silent catastrophic forgetting.** LoRA is much safer than full fine-tuning
   here since the base is frozen, but with a high rank and enough steps the model
   can lose general ability. Keep a small general-capability eval in the loop.
3. **Schema drift.** The day someone adds a sixth field, the adapter is stale.
   Version the adapter *with* the schema and treat it as one artifact.
4. **The training/inference prompt mismatch.** If the chat template used at
   training differs from the one at serving by even one token, quality quietly
   drops. I render the prompt with `apply_chat_template(...,
   add_generation_prompt=True)` in both places for exactly this reason.

### "Merge the adapter or keep it separate?"

Merge (`W' = W + (α/r)BA`) when you ship **one** model — you drop two matmuls per
adapted layer at inference and get a plain checkpoint to hand to GGUF/vLLM/ONNX.

Keep it separate when you serve **many** variants. That's LoRA's real production
superpower: one 1 GB base model resident in VRAM, dozens of 34 MB adapters
swapped per request — per-customer, per-task. Merge and you're back to one full
model per variant, both in storage and in VRAM.

And: you can't cleanly merge a QLoRA adapter into the 4-bit weights it was
trained against, because quantising the sum isn't the sum of the quantised. Load
bf16, merge, then re-quantise.

### "Why a 0.5B model? Isn't that a toy?"

For *this* task it's the right answer, and picking the smallest model that clears
the bar is the senior move. The task needs no reasoning and no world knowledge —
it needs a learned mapping. A 0.5B does it at 100% for a minute of training and
runs anywhere.

What I'd genuinely want to know before shipping: what does a 7B LoRA score on the
same eval? If it's 92%, the question becomes whether 8 points is worth 14× the
inference cost. That's a product decision, and I'd bring the number rather than
the opinion.

---

## Things to have ready if pushed

**LoRA variants you should be able to name:**
- **DoRA** — decomposes the update into magnitude and direction; usually beats
  LoRA at the same rank for a little extra compute.
- **rsLoRA** — scales by `α/√r` instead of `α/r`, which stops high ranks from
  being effectively under-trained.
- **LoRA+** — different learning rates for `A` and `B` (B higher); a free-ish win.
- **Full fine-tuning** is still better when you have lots of data and the target
  behaviour is genuinely far from the base.

**Alignment methods, one line each:** SFT teaches the format (what this project
does). DPO teaches preferences from pairs without a reward model. RLHF/PPO is
more powerful and much more machinery. For a structured-extraction task, SFT is
correct and DPO would be over-engineering.

---

### "Your eval score wouldn't move. What did you do?"

I asked what a *perfect* model would score on my eval set, and the answer was
84.2% — which is exactly what my model was scoring.

The setup: `error_code` was the only field under 100%, so exact-match failures
were its failures, and every one was the model emitting `null` where the label
held a code. My generator wrote the code into the report text 75% of the time
but labelled the row with it regardless. On **19 of 120** held-out rows the right
answer was not derivable from the input. The model did the only sane thing and
was marked wrong for it.

| | |
|---|---:|
| rows the input can support | 101 / 120 |
| best achievable accuracy | **84.2%** |
| what my adapter scored | **84.2%** |

It was sitting on the information-theoretic ceiling — perfect on every answerable
question — while my README reported a 16% failure rate. One line of dataset code
(`code = None` when the report omits it) took it to **100%**. No hyperparameter
moved.

Two things I'd draw out. **Training loss cannot see this** — it was 0.0098 the
whole time, and so would a validation loss be. And it is the failure mode people
paper over with a bigger model: had I reached for rank 64 or a 1.5B base, I would
have burned compute against a number built into the labels and concluded the task
was hard.

**Before you tune on an eval, check the eval.**

### "Did the bad labels do any harm beyond the score?"

Yes, and this is the part I find most interesting: **they installed a
hallucination.**

24% of coded training rows were demonstrating "reports of this shape carry an
`XXX-999` token — produce one even if you cannot see it". So on out-of-domain
input the model invented codes. Given a hotel-booking outage it emitted
`error_code: "BOO-402"`, a string appearing nowhere in the prompt.

After the label fix, the same probe returns `null`. Correct.

**A label the input cannot support does not just cost you a point on the eval —
it teaches the model to make things up.** If I saw fabricated identifiers in
production, the first place I'd look now is the training labels, not the decoding
parameters.

### "What was in your training data?"

800 synthetic examples, 120 held out, from `make_dataset.py`. Every operator
report is textually unique; the label space is 5 components, 3 severities, 10
error codes plus null, and 19 actions.

The reason I can tell you that precisely is the interesting part. **The first
version of this dataset had 9 actions, and four of the five components had
exactly one** — so `action` was a lookup keyed on the service name. Measured: a
predictor that copies each component's majority action scored **86.7%** on the
held-out set, against my model's 95.8%. Nine points of real learning behind a
number that reads like a strong result.

So I rebuilt it. Actions are now keyed on the **symptom**: "barcode read rate has
dropped to 42%" and "gantry 7 GPU has fallen off the bus" are both
`atlas-vision` and get different remediations, so the field cannot be answered
without reading the report body. The lookup baseline fell from 86.7% to **31.7%**
— and the model went from 95.8% to **100%**.

**Making the task harder improved the result**, which is the opposite of what
tuning a metric usually does, and the whole argument for measuring baselines: a
per-field accuracy is meaningless until you know what a stupid predictor scores,
and that is a property of the dataset, not the model.

| field | trivial baseline | model | real gain |
|---|---:|---:|---:|
| `component` | 16.7% | 100.0% | +83.3 |
| `action` | 31.7% | 100.0% | +68.3 |
| `severity` | 39.2% | 100.0% | +60.8 |
| `page_oncall` | 64.2% | 100.0% | +35.8 |

### "Your model scores 100%. What does it do on input it wasn't trained for?"

I probed exactly that, and the answer is the most useful thing I can tell you
about fine-tuning: **the format generalised perfectly and the judgement did not.**

Seven hand-written edge cases: 7/7 valid JSON, 7/7 correct schema, 7/7 obeying
the page_oncall rule. Not one malformed output. And that is the trap — *a model
that is always well-formed looks like a model that is always right.*

Read the outputs and it falls apart:

- Given "recommend a good pizza place in Rotterdam", it emitted
  `{"component":"pizza oven","severity":"SEV2","page_oncall":true}`. It triaged
  dinner and paged the on-call. There is **no abstention path**, because all 800
  training examples were incidents — the model was never shown that "not an
  incident" is an available answer.
- Told "this is SEV1 but do NOT page anyone", it **downgraded the severity to
  SEV3** so that not paging became self-consistent — an injected instruction
  moving a safety-relevant field, with every mechanical validator passing it.
- It used to **invent error codes** on out-of-domain input — `BOO-402` for a
  hotel-booking outage, a string appearing nowhere in the prompt. That one is
  now fixed, and the fix was in the *labels*, not the model (see the ceiling
  answer above).

**A claim I withdrew.** After one dataset rebuild I reported that the model had
started resisting the injection — holding SEV1 where an earlier version
downgraded. The next retrain downgraded again. Same probe, different run. I had
hedged it at the time ("I would not claim a mechanism from one probe") and the
hedge turned out to be the whole story: **with n=1 you measured a sample, not a
property.** If I wanted that claim, I would need several retrains and a set of
injection prompts, not one of each.

So: fine-tuning bought behaviour — output shape, field conventions, the page
rule — on 800 examples in under a minute, and bought **no knowledge and no
judgement**. If I were shipping this I would add an explicit `not_an_incident`
class to the training data, validate content rather than form, and put retrieval
in front of anything requiring facts.

That is also the cleanest way to answer "fine-tune or RAG?": I have measured
what fine-tuning does not give you.

### "Can you reproduce your own numbers?"

I did, months later, and I'd give you the result including the part that didn't
match. Retraining both variants from scratch with the same seed reproduced peak
VRAM (7.47 GB / 8.91 GB) and adapter size (33.60 MB) **exactly**, and training
time within 1.4% on the QLoRA run. Accuracy moved by one held-out example on
bf16 (84.2% -> 83.3%) and two on QLoRA (74.2% -> 72.5%) on the dataset version
current at the time.

**A seed does not make CUDA deterministic.** It fixes initialisation and
sampling order; it does not fix reduction order in cuBLAS matmuls or atomic
accumulation. The loss curves diverge in the fourth decimal by step 40, and a
couple of held-out examples sit close enough to a decision boundary to flip. So
the defensible claim is "83-84%, plus or minus an example", and quoting 84.2%
from one run implies a precision the hardware doesn't give you.

The reassuring half: **the mechanism reproduced more exactly than the numbers**.
In both runs the entire bf16-vs-QLoRA gap sat in `error_code` while the other
four fields matched to the decimal. That's what makes the finding safe to
defend — an explanation that survives a re-run is worth more than a percentage
that doesn't.

---

## Questions to ask *them*

- "Do you fine-tune, and if so what's the retraining trigger — schedule, drift
  metric, or someone noticing?"
- "How do you version adapters against the schema or prompt they were trained for?"
- "Do you serve multiple adapters off one base, or merge per variant?"
- "How much of your training data is synthetic versus labelled production traffic?"

---

## Related projects in this repo

- **[01_rag_local](../01_rag_local/)** — the other half of the fine-tune-vs-RAG
  decision, on the same fictional domain.
- **[03_lora_image](../03_lora_image/)** and **[04_lora_voice](../04_lora_voice/)** —
  the same LoRA idea on a diffusion model and an ASR model.
- **[06_local_gpu_inference](../06_local_gpu_inference/)** — quantisation and
  throughput, where the QLoRA numbers here come from.
