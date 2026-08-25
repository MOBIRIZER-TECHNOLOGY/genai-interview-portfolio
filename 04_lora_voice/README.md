# 🎙️ Project 04 — LoRA Fine-Tuning Whisper (domain-specific speech recognition)

Teach a general speech-recognition model your organisation's jargon — error
codes, service names, severity labels — and the **written form** each one should
take. Training data is synthesised with a TTS model, so no recording session.

> **In one sentence:** word error rate drops from **52.1% to 2.5%** — a 95%
> relative reduction — and recognition of domain terms goes from **1% to 96%**,
> after 78 seconds of training producing a 27 MB adapter.

---

## 🧠 The idea (for non-experts)

Whisper transcribes ordinary English very well. It has never heard of
`atlas-dispatch`, `TLM-330`, or `nw-barcode-ocr-v2`, so it does what any
listener would: writes down something that sounds close.

Real base-model output from this eval set:

| Said | Whisper heard | Should be |
|---|---|---|
| "sev one" | `a 7` | `SEV1` |
| "sev two" | `SEPTU` | `SEV2` |
| "en double you pallet detect vee four" | `In double-upalit detect V` | `nw-pallet-detect-v4` |
| "tee ell em one oh one" | `TLM 101` | `TLM-101` |
| "atlas roll" | `Atlas Roll` | `atlas-roll` |

Note the last two. Those aren't mishearings — Whisper heard perfectly and wrote
the wrong *form*. So the task is two things at once:

1. **Domain vocabulary** — recognise jargon it has never encountered.
2. **Inverse text normalisation** — map the spoken form to the canonical written
   form. `atlas roll` → `atlas-roll`, not `Atlas Roll`.

That second half is what an ASR system inside an ops tool actually has to do, and
it's pure convention — exactly what fine-tuning is for.

### Where the training data comes from

There is no recording session here. We **synthesise the audio with a TTS model**
(SpeechT5) and pair it with the canonical written transcript. Each example has a
*spoken* form fed to the TTS ("tee ell em three thirty") and a *written* target
(`TLM-330`), so the model learns the mapping directly.

Speaker variety comes free: SpeechT5 takes speaker identity as a 512-dim x-vector
**input**, not as weights. We sample 84 x-vectors across 7 distinct CMU Arctic
voices, so the model hears the jargon in many voices.

**The honest limitation** — say this before an interviewer does: *synthetic
speech is too clean.* A model trained only on TTS output can learn TTS acoustics
and do worse on a real microphone. Mitigations here are the standard ones:
speaker variety, plus augmentation (random gain, additive noise at 12–30 dB SNR,
and a single early reflection standing in for room reverb). `--clean` disables
augmentation so you can measure the difference.

This project attacks that limitation from two directions — a **cross-engine
holdout** you can run right now with no hardware (results below), and a **real
recording pipeline** (`record.py` / `import_audio.py`) for the test that actually
settles it.

---

## ✅ Proof it works (measured on this machine)

RTX 5070 Ti, `openai/whisper-small` (242 M), 240 training clips (17.6 min of
audio), 8 epochs, **78 s**, peak VRAM **4.07 GB**, adapter **27 MB**
(2.845% of parameters trained). 60 held-out clips, greedy decoding.

| Metric | Base Whisper | LoRA | Δ |
|---|---:|---:|---:|
| **WER** ↓ | 52.1% | **2.5%** | −49.6 pts (**95% relative**) |
<!-- reproduced 2026-08-25: base 52.1% (exact), LoRA 2.1% -->
| **CER** ↓ | 12.9% | **1.6%** | −11.2 pts |
| **domain-term accuracy** ↑ | 1.0% | **96.0%** | +95.0 pts |
| **exact sentence match** ↑ | 0.0% | **91.7%** | +91.7 pts |

Training loss 1.4373 → 0.0007.

### Side by side

```
REF : Page on call, TLM-101 in Lyon is a SEV1.
BASE: Page and call, TLM 101 and Lion is a 7.
LORA: Page on call, TLM-101 in Lyon is a SEV1.

REF : Bristol reports the starvation guard on atlas-roll.
BASE: Bristol reports the starvation guard on Atlas Roll.
LORA: Bristol reports the starvation guard on atlas-roll.

REF : nw-pallet-detect-v4 confidence dropped, routing to the Rotterdam rule.
BASE: In double-upalit detect V, routing to the Rotterdam rule.
LORA: nw-pallet-detect-v2 confidence dropped, routing to the Rotterdam rule.
```

That last line is the remaining error class and worth understanding: the LoRA got
the whole identifier right except the **version digit** (`v4` → `v2`). Version
suffixes are acoustically near-identical and low-information, so they're the last
thing to be learned. The fix is more training examples that vary only the
version, not more epochs — loss is already 0.0007.

### 🔍 Why WER *and* domain-term accuracy

WER averages over every word in the sentence, and most words are ordinary English
that both models get right. That **dilutes the thing we set out to fix**.
Domain-term accuracy asks only about the terms that matter: of the `TLM-330`s,
`atlas-dispatch`es and `SEV2`s in the reference, how many appear verbatim?

The gap between them is the story: base WER of 52% sounds like a model that can't
hear, but base *CER* is only 12.9% — most words are nearly right. It's the
identifiers that are wrong, and 1.0% domain-term accuracy says so precisely.
Reporting only WER would understate both the problem and the fix.

---

## 🧪 The transfer test: does it survive a different speech engine?

The scary failure mode of TTS-trained ASR is that the model learns **the engine,
not the vocabulary** — and an eval on the same engine cannot tell the difference.

So `make_holdout.py` builds a second eval set with **Windows SAPI**
(Microsoft David + Zira): a completely different synthesis family from SpeechT5 —
different prosody, timbre and artefacts, sharing nothing with the training
distribution except English. Same sentence generator, three speaking rates, same
augmentation. If the LoRA had merely memorised SpeechT5's acoustics, it would
collapse here.

**It didn't collapse — it transferred.** 60 SAPI clips, greedy decoding:

| Metric | Base Whisper | LoRA (trained only on SpeechT5) | Δ |
|---|---:|---:|---:|
| **WER** ↓ | 56.7% | **1.5%** | −55.2 pts (97% relative) |

**Reproduced from scratch, 2026-08-25.** Retrained (80 s vs 78.3 s; peak VRAM
4.07 GB and adapter 27.04 MB **exactly**) and re-evaluated both sets:

| | documented | reproduced |
|---|---:|---:|
| base WER, in-distribution | 52.1% | **52.1%** (exact) |
| base domain-term accuracy | 1.0% | **1.0%** (exact) |
| LoRA WER, in-distribution | 2.5% | 2.1% |
| LoRA domain-term accuracy | 96.0% | **96.0%** (exact) |
| LoRA exact sentence match | 91.7% | **91.7%** (exact) |
| base WER, SAPI holdout | 56.7% | **56.7%** (exact) |
| **LoRA WER, SAPI holdout** | 1.5% | **0.9%** |
| LoRA domain terms, SAPI | — | **100.0%** |

Every base number is identical to the digit, because base inference is greedy
and deterministic. The adapted numbers came out slightly *better* than the
recorded run — the same CUDA-nondeterminism-in-training effect documented in
project 02, landing favourably this time. **The cross-engine claim is the one
that matters and it held: a model trained only on SpeechT5 audio transcribes
SAPI audio at 0.9% WER with 100% domain-term recall.**
| **domain-term accuracy** ↑ | 10.7% | **100.0%** (122/122) | +89.3 pts |
| exact sentence match ↑ | 1.7% | 91.7% | +90.0 pts |

Cross-engine numbers came out slightly *better* than the in-distribution eval
(1.5% vs 2.5% WER) — SAPI enunciates identifiers more crisply than SpeechT5 does.
Base Whisper's classic failures reappear on SAPI exactly as before
(`Rays a 7-1 for Atlas Sim` for "Raise a SEV1 for atlas-sim"), and the LoRA fixes
all of them.

**What this does and doesn't prove.** It proves the adapter learned something
engine-independent — the vocabulary and the written-form conventions. It does
**not** prove it works on a human: SAPI is still synthetic — no room, no breath,
no hesitation, no real microphone. That last step needs real recordings, which is
what the next section is for.

---

## 🎙️ Getting REAL audio in (two ways)

This machine had no microphone (0 input devices — remote session), so the real
recordings themselves can't be produced here. The full pipeline is built and
tested; you supply the voice, anywhere you have a mic.

### Option A — record an eval set (~10 minutes, highest value)

```powershell
python record.py --list-devices        # confirm a mic is visible
python record.py --split eval --n 60   # guided session -> data_real/
```

Guided push-to-talk session: shows each target transcript with pronunciation
hints (`TLM-101 = "tee ell em one oh one"`), records at 16 kHz mono, auto-trims
silence, warns on clipping/silence, supports retake/skip/playback, and **saves
progress after every clip** so you can quit and resume. Prompts use a different
RNG seed than the training set, so no sentence overlaps.

Then the number that settles the question:

```powershell
python evaluate.py --data data_real
```

> On a remote desktop, enable mic redirection first: RDP client → Show Options →
> Local Resources → Remote audio → Settings → *Record from this computer*.

### Option B — import recordings you already have

Record voice memos on your phone (read the prompts from `record.py`'s output, or
any sentences using the domain terms), then:

```powershell
# transcripts.tsv: one line per file ->  memo_001.m4a<TAB>Page on call, TLM-101...
python import_audio.py --src C:\my_recordings --split eval --dry-run
python import_audio.py --src C:\my_recordings --split eval
```

Converts any format librosa can open (m4a/mp3 need ffmpeg), resamples to 16 kHz
mono, flags clipped/silent files, writes the manifest.

### Training on real + synthetic together

Once you have real **training** clips (`record.py --split train --n 100`):

```powershell
python train_lora.py --epochs 8 --mix data_real --mix-repeat 3
```

`--mix-repeat` oversamples the real clips because they're outnumbered (100 real
vs 240 synthetic ≈ 55% real exposure at ×3). It's duplication, not new
information — past ~5× you're memorising those exact clips. The strongest recipe
is still: synthetic for vocabulary coverage, real for acoustics, real-only for
eval.

---

## 🧪 Ablation: which stack should get the adapter?

Whisper is encoder-decoder. My prior was "adapt both — the encoder learns the
acoustics, the decoder learns the output convention." Then I measured it:

| Arm | trainable | adapter | train time | peak VRAM | WER | domain-term | exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoder + decoder | 7.08 M (2.85%) | 27.0 MB | 78 s | 4.07 GB | 2.5% | 96.0% | 91.7% |
| encoder only | 2.36 M (0.97%) | 9.0 MB | 63 s | 3.64 GB | **2.3%** | 95.0% | 90.0% |
| decoder only | 4.72 M (1.92%) | 18.0 MB | **46 s** | **1.84 GB** | 2.5% | **97.0%** | **93.3%** |

**All three land within noise of each other.** On 60 clips, 2.3% vs 2.5% WER is a
couple of words. The correct conclusion is *not* "decoder-only is better" — it's
**"this eval is saturated and cannot discriminate between these choices"**. To
actually decide, I'd need harder audio (real microphones, accents, overlapping
speech) where the encoder has real work to do.

What the table *does* say honestly is that the resource differences are real and
measurable: decoder-only used **45% of the VRAM and 59% of the time** for
indistinguishable quality. If I had to ship one today on constrained hardware,
that's the arm I'd pick — and I'd say exactly why, including that my prior was
wrong.

> One implementation trap worth knowing: `model.decoder.layers.N.encoder_attn.*`
> is **cross-attention living in the decoder**. A naive `.*encoder.*` regex
> sweeps it into the "encoder-only" arm and quietly invalidates the ablation.

---

## 📁 What's in this project

```
04_lora_voice/
├── make_dataset.py     TTS-synthesises audio + canonical transcripts (SpeechT5)
├── make_holdout.py     cross-engine eval set via Windows SAPI (David + Zira)
├── record.py           guided microphone session -> data_real/ (resume, retake, QC)
├── import_audio.py     convert existing recordings (m4a/mp3/wav) -> data_real/
├── train_lora.py       Whisper LoRA loop; --mix data_real to blend real audio in
├── evaluate.py         WER, CER, domain-term accuracy; --data picks the eval set
├── data/
│   ├── train/*.wav     240 clips, 17.6 min
│   ├── eval/*.wav      60 clips, 4.3 min
│   └── {train,eval}.jsonl
├── data_sapi/          the cross-engine (SAPI) eval set
├── data_real/          your recordings land here (not in the repo)
├── lora-out/           encoder+decoder adapter (27 MB)
├── lora-out-encoder/   ablation arm
└── lora-out-decoder/   ablation arm
```

```
make_dataset.py ──▶ data/       (audio + written transcripts)
                       │
train_lora.py   ──▶ lora-out/   (27 MB adapter)
                       │
evaluate.py     ──▶ WER / domain-term table
```

---

## 🚀 How to run it

```powershell
..\activate.ps1

python make_dataset.py --train 240 --eval 60    # ~4 min (TTS on GPU)
python train_lora.py --epochs 8                 # 78 s
python evaluate.py --show 6                     # base vs LoRA + transcripts

python make_holdout.py --n 60                   # cross-engine (SAPI) eval set
python evaluate.py --data data_sapi             # the transfer test
```

Ablations:

```powershell
python train_lora.py --epochs 8 --decoder-only --output lora-out-decoder
python evaluate.py --lora lora-out-decoder --skip-base
```

First run downloads SpeechT5 + HiFiGAN (~600 MB), the x-vector archive, and
Whisper-small (~1 GB).

---

## 🔧 Use your OWN audio

Point the loader at real recordings — it's the same JSONL shape:

```json
{"audio": "train/0001.wav", "text": "Page on call, TLM-101 in Lyon is a SEV1."}
```

Requirements: **16 kHz mono WAV** (Whisper resamples nothing — wrong rate is a
hard error here rather than a silent quality loss), and transcripts in the
**canonical written form you want produced**, not a phonetic one.

Real audio beats synthetic. The strongest practical recipe is **both**: bootstrap
with TTS to cover vocabulary you have no recordings of, then mix in real clips —
especially the ones the model currently gets wrong.

---

## ⚙️ The knobs that actually matter

| Knob | Effect | Guidance |
|---|---|---|
| `--model` | `whisper-tiny` → `whisper-large-v3` | `small` is the sweet spot on 16 GB. `tiny`/`base` train in seconds and are much weaker on jargon |
| `--rank` | adapter capacity | 32 here. ASR adaptation wants more rank than a text format task — you're reshaping acoustic *and* output behaviour |
| `--lr` | learning rate | `1e-3`, higher than the text project. Whisper's LoRA tolerates it and converges in fewer steps |
| `--epochs` | passes over the data | 8 on 240 clips. Watch held-out WER, never train loss |
| `--encoder-only` / `--decoder-only` | which stack adapts | see the ablation table |
| `--clean` *(make_dataset)* | disable augmentation | run once to see how much augmentation is buying you |
| `--voices-per-speaker` | x-vectors sampled per CMU voice | more speaker variety = better generalisation to real voices |

---

## 🖥️ Tech stack

- **ASR base:** `openai/whisper-small` (242 M, encoder-decoder)
- **Method:** LoRA r=32 α=64 on `q,k,v,out_proj` in both stacks
- **TTS for data:** `microsoft/speecht5_tts` + `speecht5_hifigan`, CMU Arctic
  x-vectors for 7 speaker identities
- **Precision:** fp32 weights, bf16 autocast for forward/backward
- **Metrics:** `jiwer` for WER/CER, plus a custom domain-term recall
- **Validated on:** RTX 5070 Ti 16 GB, transformers 5.15.1, torch 2.11.0+cu128

---

## ❓ FAQ

**Why is 16 kHz mandatory?**
Whisper's feature extractor produces an 80-bin log-mel spectrogram assuming a
16 kHz input. Feed it 44.1 kHz and every frequency bin is wrong. The loader
raises rather than silently resampling, because a silent resample hides a real
data-pipeline bug.

**What's the trap in Whisper LoRA training?**
Labels start with a fixed special-token prefix
(`<|startoftranscript|><|en|><|transcribe|>`). The model builds `decoder_input_ids`
by shifting labels right and prepending BOS — so BOS must be **stripped from the
labels** or it appears twice and every position is off by one. The symptom is
nasty: loss decreases normally and the model emits garbage. `collate()` strips it.

**Why does every clip cost the same compute regardless of length?**
Whisper pads or truncates to exactly 30 seconds by design. A 3-second utterance
costs the same forward pass as a 30-second one. That's a real inefficiency for
short-utterance workloads and an argument for batching aggressively.

**Isn't training on TTS audio cheating?**
It's a well-established technique for domain adaptation, and it has a specific
failure mode you must name: the model can learn TTS acoustics rather than the
vocabulary, and get *worse* on real microphones. Speaker variety and augmentation
reduce that. This project now has two answers on top of the mitigations: the
**cross-engine SAPI holdout passed** (WER 1.5%, 100% domain terms on an engine
never seen in training — so it learned the vocabulary, not the engine), and
`record.py` / `import_audio.py` provide the real-microphone test, which is still
the only complete proof.

**Could I just use a prompt or a hotword list instead?**
Yes, and you should try it first. Whisper accepts an `initial_prompt` to bias
decoding toward given vocabulary, and it's free. It works for a handful of terms
and degrades as the list grows past what fits usefully in the prompt. Fine-tuning
is what you reach for when the vocabulary is large, stable, and worth 78 seconds.

**Does it forget general English?**
**Measured, and mostly no** — `probe_general_speech.py` closes what used to be
an admitted gap here. 40 ordinary English sentences with zero domain vocabulary,
same TTS voices, so content is the only variable:

| | base Whisper | + adapter |
|---|---:|---:|
| general-English WER | 1.9% | **3.8%** |
| exact sentence match | 90.0% | 85.0% |

Read both columns honestly: **+1.9 points absolute, but a doubling in relative
terms.** The trade is still strongly favourable — you pay ~2 points of general
WER to take domain WER from 52.1% to 2.1% — but "no cost" would be a lie.

The *shape* of the damage is the useful part, and it is not what
"catastrophic forgetting" usually means. The model has not stopped hearing;
it has started **writing everything in the domain's house style**:

```
REF : the film starts at half past eight
LORA: the film starts at half-past-eight        <- hyphenation, learned from CON-401
```

Of 6 differing outputs, most are formatting drift (hyphens, digits for numbers)
or errors the *base* model makes too. Exactly one is a new acoustic error
("cyclist that passed" → "cyclist in the past"). That makes sense: this adapter
was trained to teach written form, and written form is what leaked.

If you needed to fix it: mix general speech into training, drop the rank, or
train fewer epochs — and re-run this probe to check the fix actually helped.

---

## Related projects

- **[02_lora_text](../02_lora_text/)** — the same LoRA idea on a language model
- **[03_lora_image](../03_lora_image/)** — and on a diffusion UNet
