# 🎤 Interview notes — speech models, ASR adaptation, synthetic data

---

## The 60-second project pitch

> "I adapted Whisper-small to a domain vocabulary — error codes, service names,
> severity labels — where the base model was effectively unusable: 52% WER and
> 1% domain-term accuracy. LoRA on both attention stacks, 240 clips, 78 seconds
> of training, 27 MB adapter. WER went to 2.5% and domain-term accuracy to 96%.
>
> Two things I'd want to talk about. First, I had no recordings, so I
> **synthesised the training audio with a TTS model** and paired it with the
> canonical *written* transcript — so the model learns inverse text
> normalisation, spoken 'atlas roll' to written `atlas-roll`, not just the
> vocabulary. Second, that technique's known failure mode is learning the TTS
> engine instead of the vocabulary — so I built a **cross-engine holdout** with
> Windows SAPI, a completely different synthesis family, and the adapter
> transferred: 1.5% WER and 100% domain-term accuracy on an engine it never
> heard in training. What's still open is real microphone audio — I built the
> recording pipeline for that test, but the machine I trained on has no mic."

---

## Core questions

### "How does Whisper work?"

Encoder-decoder transformer over audio.

Audio → resampled to **16 kHz** → 80-bin **log-mel spectrogram**, padded or
truncated to exactly **30 seconds** → encoder produces audio states → decoder
autoregressively emits text tokens, cross-attending to those states.

Two consequences of the fixed 30-second window worth knowing:
- A 3-second clip costs the same forward pass as a 30-second one. Real
  inefficiency for short-utterance workloads; a reason to batch hard.
- Longer audio needs chunking, and naive chunking cuts words in half — production
  systems overlap the windows and stitch.

The decoder is prompted with special tokens that set language and task
(`<|en|><|transcribe|>` vs `<|translate|>`), which is how one model does 99
languages and translation in a single checkpoint.

### "Where did you put the adapters, and why?"

`q,k,v,out_proj` in both stacks, r=32. The reasoning was: encoder adapters help
it *hear* the domain, decoder adapters teach the output *convention*.

**Then I ran the ablation and my reasoning didn't hold up:**

| Arm | trainable | VRAM | time | WER | domain-term |
|---|---:|---:|---:|---:|---:|
| both | 7.08 M | 4.07 GB | 78 s | 2.5% | 96.0% |
| encoder only | 2.36 M | 3.64 GB | 63 s | 2.3% | 95.0% |
| decoder only | 4.72 M | 1.84 GB | 46 s | 2.5% | 97.0% |

All within noise on 60 clips. The correct read is **not** "decoder-only wins" —
it's "**this eval is saturated and can't discriminate**". To decide properly I'd
need harder audio, real microphones and accents, where the encoder has actual
work to do. What the table *does* establish is that decoder-only got the same
quality for 45% of the VRAM, which is what I'd ship on constrained hardware.

Being able to say "I measured it, my prior was wrong, and the eval isn't strong
enough to fully settle it" is worth more than a confident wrong answer.

**Implementation trap worth mentioning:** `decoder.layers.N.encoder_attn.*` is
cross-attention that lives in the **decoder**. A regex like `.*encoder.*` sweeps
it into the encoder-only arm and silently invalidates the ablation.

### "Talk me through generating training data with TTS."

I had a vocabulary problem and no recordings. So each training example is a pair:
a **spoken** form fed to SpeechT5 ("tee ell em three thirty") and a **written**
target (`TLM-330`). The model learns the mapping directly, which means it learns
domain vocabulary *and* inverse text normalisation in one task.

Speaker variety is free with SpeechT5: speaker identity is a 512-dim x-vector
**input**, not a weight. I sample 84 x-vectors across 7 CMU Arctic voices.

**The limitation, stated plainly:** synthetic speech is too clean and too
uniform. A model trained only on it can learn TTS acoustics rather than the
vocabulary, and get *worse* on real microphones. I mitigate with augmentation —
random gain, additive noise at 12–30 dB SNR, and a single early reflection as a
crude reverb — but mitigation isn't proof.

**So I tested it two ways.** First, a cross-engine holdout: 60 clips from
Windows SAPI (David/Zira), a concatenative engine sharing nothing with SpeechT5
but the language. The adapter held — 1.5% WER, 100% domain-term accuracy — so it
learned something engine-independent, not SpeechT5's artefacts. Second, a
recording pipeline (`record.py`, guided session with QC and resume) for the real
microphone test, which is the only complete proof and needs hardware this
machine doesn't have. Being able to say 'here is the falsification test I ran,
here is the one that remains' is the shape of the honest answer.

In production the pattern is: bootstrap with TTS for coverage, then continuously
mix in real labelled traffic, weighted toward the failures.

### "Why report domain-term accuracy on top of WER?"

Because WER averages over every word in the sentence, and most of them are
ordinary English that both models already get right. That **dilutes exactly the
thing you set out to fix.**

The numbers show it: base WER is 52% but base **CER is only 12.9%** — most words
are nearly right, character-wise. It's the identifiers that are wrong, and
domain-term accuracy of 1.0% says that precisely where WER can't.

It also matters that WER treats `TLM 330` vs `TLM-330` as a full word error while
CER counts it as one character — and for a downstream parser those are very
different severities. Report both and you can tell "misheard" apart from
"misformatted".

### "What's the nastiest bug in Whisper fine-tuning?"

The **BOS double-prepend**. Whisper labels begin with a special-token prefix
(`<|startoftranscript|><|en|><|transcribe|>`). The model builds
`decoder_input_ids` itself by shifting labels right and prepending BOS. If you
leave BOS in the labels, it appears twice and every position is off by one.

What makes it nasty is the symptom: **loss decreases normally** — the model
happily learns the shifted task — and the outputs are garbage. There's no
exception and nothing in the logs. You only catch it by actually decoding
samples, which is a good argument for generating a few transcripts every epoch
rather than trusting the loss curve.

Runner-up: sample rate. Whisper's mel filterbank assumes 16 kHz. Feed 44.1 kHz
and every frequency bin is wrong — the model still produces plausible text, just
worse. My loader raises rather than silently resampling, so a pipeline bug
surfaces as an error instead of a quality mystery.

### "What did the fine-tune cost you? Did it forget general English?"

I measured it rather than reassuring you, because a domain eval structurally
cannot see this: every number on it goes up while the model quietly gets worse
at everything else.

40 ordinary English sentences, zero domain vocabulary, same TTS voices so
content is the only variable:

| | base Whisper | + adapter |
|---|---:|---:|
| general-English WER | 1.9% | 3.8% |
| exact sentence match | 90.0% | 85.0% |

**+1.9 points absolute — but a doubling in relative terms.** I would give you
both framings, because quoting only the first is how people hide a regression.
The trade is still clearly worth it: ~2 points of general WER to take domain WER
from 52.1% to 2.1%. "No cost" would be a lie.

The interesting part is the *shape* of the damage, which is not what
catastrophic forgetting usually means. The model has not stopped hearing — it
has started writing everything in the domain's house style:

```
REF : the film starts at half past eight
LORA: the film starts at half-past-eight     <- hyphenation learned from CON-401
```

Of six differing outputs, most are formatting drift or errors the base model
makes too; exactly one is a new acoustic error. That is coherent with what this
adapter was trained to do — it teaches *written form* for spoken identifiers, so
written form is what leaked. It also suggests the cheap fix is a mixed-in
general-speech set rather than a lower rank, and I would re-run the same probe
to confirm the fix helped rather than assuming.

### "Fine-tuning vs simpler options for domain vocabulary?"

Try the cheap things first, and say so:

1. **`initial_prompt`** — Whisper accepts a text prompt that biases decoding
   toward given vocabulary. Free, no training. Works for a handful of terms,
   degrades as the list grows past what usefully fits.
2. **Post-processing / fuzzy correction** — map near-misses onto a known
   vocabulary list. Brittle, but genuinely effective for a small closed set, and
   you can ship it in an afternoon.
3. **Fine-tuning** — what you reach for when the vocabulary is large, stable, and
   worth the training run. 78 seconds here.

Volunteering the ladder shows judgement. Jumping straight to fine-tuning for six
error codes would be over-engineering.

### "How would you productionise this?"

- **Serving:** `faster-whisper` (CTranslate2) or `whisper.cpp` — several times
  faster than HF transformers for inference. Adapter merged in, then quantised.
- **Streaming:** Whisper is not natively streaming. You run a VAD, chunk on
  speech boundaries with overlap, and stitch. Getting the overlap logic right is
  most of the work.
- **Monitoring:** track WER against a periodically-relabelled sample, and watch
  the *domain-term* rate separately — that's what degrades when new codes appear.
- **The retraining trigger:** a new error-code family shipping in the product.
  The adapter should be versioned against the vocabulary it was trained for.
- **The gap I'd close first:** a general-speech eval alongside the domain one, to
  catch catastrophic forgetting. LoRA on a frozen base is fairly safe, but at
  rank 32 and 8 epochs I'd want the number rather than the assumption.

---

## Related concepts to have ready

**Other speech tasks and the models for them:** diarisation (who spoke when) —
pyannote; speaker verification — x-vectors/ECAPA, the same embeddings used here
for TTS conditioning; VAD — Silero; TTS — SpeechT5, Piper, XTTS.

**Streaming ASR architectures:** RNN-T and Conformer-Transducer are designed for
streaming in a way Whisper isn't. If low latency is a hard requirement, Whisper
may be the wrong base model regardless of accuracy.

**Why WER can exceed 100%:** insertions are unbounded. A model that hallucinates
a paragraph over three words of speech scores far above 1.0. Whisper does this on
silence and music, which is why a VAD in front of it is standard.

---

## Questions to ask *them*

- "What does your ASR eval set look like — real recordings, and how often
  relabelled?"
- "Do you use synthetic audio anywhere in training, and how do you validate it
  transfers?"
- "Is streaming a hard requirement? That changes the base model choice entirely."
- "How do you handle new domain vocabulary appearing — retrain, prompt, or
  post-process?"

---

## Related projects

- **[02_lora_text](../02_lora_text/)** — LoRA on a language model
- **[03_lora_image](../03_lora_image/)** — LoRA on a diffusion UNet
