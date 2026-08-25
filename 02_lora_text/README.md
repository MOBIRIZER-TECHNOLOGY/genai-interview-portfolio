# 🔤 Project 02 — LoRA Fine-Tuning a Text LLM (structured extraction)

Teach a 0.5B language model to do a job it cannot do out of the box: turn a
free-text incident report into a **strict JSON triage record** using
domain-specific vocabulary and rules it has never seen.

> **In one sentence:** 800 synthetic examples, 59 seconds of training, a 34 MB
> adapter — and exact-match accuracy goes from **0% to 84.2%** on held-out data.

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

RTX 5070 Ti, `Qwen/Qwen2.5-0.5B-Instruct`, r=16, 3 epochs, **59 seconds**,
120 held-out examples, greedy decoding.

| Metric | Base | LoRA | Δ |
|---|---:|---:|---:|
| valid JSON | 100.0% | 100.0% | — |
| correct schema (exactly 5 keys) | 100.0% | 100.0% | — |
| **exact match (all 5 fields)** | **0.0%** | **84.2%** | **+84.2** |
| `component` | 0.8% | 100.0% | +99.2 |
| `severity` | 31.7% | 100.0% | +68.3 |
| `page_oncall` | 35.8% | 100.0% | +64.2 |
| `error_code` | 51.7% | 84.2% | +32.5 |
| `action` | 0.0% | 96.7% | +96.7 |

Training loss: 1.5377 → 0.0097. Peak VRAM 7.47 GB. Adapter 33.6 MB.

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
         "action": "re-run the scenario with a pinned seed and compare output hashes"}
```

The base model title-cases the component (`Atlas-Sim`), invents a generic action,
and gets the paging rule backwards. Every one of those is a *convention*, not a
fact — which is exactly what fine-tuning is for.

### 🧪 The QLoRA experiment — and why it *lost*

I ran the identical training with `--load-4bit` (NF4 base weights). The result is
a good lesson in not cargo-culting a technique:

| | bf16 LoRA | QLoRA (4-bit) |
|---|---:|---:|
| peak VRAM | **7.47 GB** | 8.91 GB |
| training time | **58.7 s** | 147.0 s |
| final train loss | 0.0097 | 0.0098 |
| **held-out exact match** | **84.2%** | 74.2% |

QLoRA was slower, used *more* memory, and scored 10 points worse. That is not a
bug — it is what QLoRA does at this model size, and being able to explain it is
the point:

- **Memory here is activations, not weights.** The 0.5B base is only 0.93 GB in
  bf16. Peak VRAM is dominated by activations for `batch 8 × seq 512`. 4-bit
  shrinks the 0.93 GB but adds dequantisation buffers and the fp32 upcasting that
  `prepare_model_for_kbit_training` inserts — net negative.
- **Speed:** every forward pass now dequantises NF4 → bf16 on the fly. Pure
  overhead when you weren't memory-bound to begin with.
- **Quality:** the 10-point gap is entirely in `error_code` (84.2% → 74.2%),
  the field that needs the finest discrimination. 4-bit base weights are a
  lossy starting point.

**QLoRA earns its keep when weights dominate memory** — a 7B, 13B or 70B where
the base doesn't fit otherwise. At 0.5B it's strictly worse. Reaching for it
here because it's the fashionable technique would be the wrong call.

**`error_code` at 84.2% is the weakest field and I know why:** ~25% of the
training reports mention the component but no code, so the model has to learn
"absent ⇒ null" rather than "guess a plausible code". That's the residual error.
The fix is more null-code examples, not more epochs — the loss is already 0.0097,
so the model has memorised the training set and further training would only
overfit.

---

## 📁 What's in this project

```
02_lora_text/
├── make_dataset.py       synthetic incident reports -> JSON labels
├── train_lora.py         explicit LoRA training loop (bf16 or 4-bit QLoRA)
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
