# 🎤 Interview notes — diffusion models and image LoRA

---

## The 60-second project pitch

> "I taught Stable Diffusion 1.5 a visual concept it had never seen, from 24
> synthetic images, with a 12 MB LoRA on the UNet's attention layers — 0.185% of
> the parameters, 11 minutes on one consumer GPU.
>
> The part worth talking about is that my **first attempt failed**, and the
> diagnosis is the interesting bit. I'd written a fully descriptive caption — 'a
> glowing amber hexagonal warning beacon with black chevron stripes' — and the
> model learned almost nothing about the shape. SD already knows every one of
> those words, so the loss could go down using existing concepts and the rare
> trigger token was never forced to carry anything. Stripping the caption to 'a
> photo of a sks beacon' fixed it in one line.
>
> And I measured it rather than eyeballing it: CLIP concept fidelity +40%
> relative, but prompt adherence −15% and diversity −41% at full scale. At
> `lora_scale` 0.7 you get 44% of the fidelity gain and adherence actually comes
> out slightly *above* baseline — so 0.7 is the better operating point, and I
> only know that because I ran both."

That pitch does two things deliberately: it leads with a **failure and its
diagnosis**, and it names a **trade-off with numbers**. Both are seniority
signals.

---

## Core questions

### "How does Stable Diffusion actually work?"

Four components, and knowing which is frozen during LoRA matters:

- **VAE** — compresses a 512×512×3 image to a 64×64×4 latent. This is the trick
  that makes SD run on consumer hardware: diffusion happens in a space 48× smaller
  than pixels.
- **Text encoder** (CLIP ViT-L/14) — turns the prompt into a 77×768 embedding.
- **UNet** — the only learned part of the loop. Given a noisy latent, a timestep,
  and the text embedding, it predicts **the noise that was added**.
- **Scheduler** — defines the noise schedule. Not learned at all.

**Training:** take an image, encode to latent, add `t` steps of noise, ask the
UNet to predict that noise conditioned on the caption, MSE loss. That's the whole
objective — no adversarial loss, no perceptual loss.

**Inference:** start from pure noise and iteratively subtract the UNet's predicted
noise, 20–50 times, then VAE-decode to pixels.

### "Where do the LoRA adapters go, and why there?"

The UNet's attention projections: `to_q`, `to_k`, `to_v`, `to_out.0`.

Specifically **cross-attention** is where the text embedding meets the image
latent — it's literally the layer that decides "this token should influence these
pixels". If you're teaching a new *concept* tied to a new *word*, that's the
mechanism you need to reach. It's also why the adapter is 12 MB and not 3 GB.

You'd extend beyond attention if you were teaching a *style* rather than a
subject, since style lives more diffusely in the network. And some setups also
train the text encoder (or do textual inversion, which trains *only* a new token
embedding) — more capacity, more risk of damaging the model's general language
understanding.

### "Your first run failed. What happened?"

**Caption dilution.** My caption was `"a photo of a sks beacon, a glowing amber
hexagonal warning beacon with black chevron stripes, dark slate background"`.

Stable Diffusion already knows "glowing", "amber", "warning beacon", "dark
background". With those in the caption, the training loss can be driven down
almost entirely using concepts the model already has — so no gradient pressure
lands on the rare token `sks`. It learned "amber glow" and stopped.

The fix is the DreamBooth captioning rule: **describe what varies, name what is
constant.** The subject is constant across every image, so it gets a token and
nothing else — `"a photo of a sks beacon"`. Now that token is the *only* handle
on the concept and it has to learn. Backgrounds and poses, which do vary, are
what you'd caption if they varied in your data.

**Then the follow-up I'd want asked, because I got it wrong first:** how do I
know it was the caption? Originally I didn't. Attempt 1 also used rank 8 vs 16
and 800 steps vs 1500 — three variables moved at once and I attributed the
failure to one of them.

So I ran the control: the descriptive caption at **rank 16, 1500 steps**, on
md5-identical images. Caption as the only difference.

| arm (rank 16, 1500 steps) | concept fidelity | Δ vs base |
|---|---:|---:|
| minimal caption | **0.3251** | **+0.0935** |
| descriptive caption | 0.2701 | +0.0385 |

**The descriptive caption captures 41% of the fidelity gain.** Dilution
confirmed — and the test was rigged against my hypothesis, because fidelity is
scored against CLIP text almost identical to the descriptive caption itself.
That arm was trained on those exact words and still lost by more than half.

The part I didn't expect: **caption dilution is indistinguishable from turning
the adapter down.** The diluted model at scale 1.0 lands on the same operating
point as the good model at scale 0.7 — fidelity 0.2701 vs 0.2766, adherence
0.2524 vs 0.2483. It isn't a different kind of damage, it's less learning. Which
means the diluted adapter also *keeps* its prompt adherence and looks safer on
every axis except the one you trained it for. **A LoRA that costs you nothing may
just be a LoRA that learned nothing.**

### "How do you know it learned a concept rather than memorising an image?"

You measure two things that pull against each other, plus a third:

- **Concept fidelity** — CLIP similarity between generated images and a text
  description of the target. Should go **up**.
- **Prompt adherence** — CLIP similarity to the *rest* of the prompt, the context
  the concept isn't ("on a wooden workbench"). Tends to go **down** as the
  adapter overfits — the model starts drawing the training image regardless of
  what you asked.
- **Diversity** — mean pairwise distance across seeds. Near zero means every seed
  gives the same picture, the other classic overfitting signature.

My numbers at scale 1.0: fidelity +0.093, adherence −0.036, diversity −0.081. The
adapter works and it costs something. At 0.7: fidelity +0.041, adherence +0.007,
diversity +0.033 — nearly free.

**A LoRA that maxes fidelity and destroys adherence has memorised a picture, not
learned a concept.** Most LoRA demos report fidelity alone, which makes
overfitting look like success.

**And the caveat I'd volunteer:** CLIP is a proxy, and it's the same model family
that guided SD's training, so it's not an independent judge. It detects the
*direction* of a change reliably and fine ranking unreliably.

### "Why didn't your loss go down?"

It sat at ~0.06–0.07 for the entire run and told me nothing — and that's expected
in diffusion training.

Each step samples a **random timestep**. Predicting noise at t=900 (almost pure
noise) is a completely different difficulty from t=50 (nearly clean). The loss is
dominated by which timestep you happened to draw, not by how well the concept is
learned. A run that's learning beautifully and one that's learning nothing can
have indistinguishable loss curves.

So you judge by **generating samples at checkpoints**. This genuinely surprises
people arriving from supervised training, where the loss curve is the primary
instrument.

### "LoRA vs DreamBooth vs textual inversion?"

- **Textual inversion** — trains *only* a new token embedding (a few KB). Model
  completely untouched. Very cheap, very portable, limited capacity: it can only
  express things the model can already render.
- **DreamBooth** — full fine-tuning of the UNet on a few images, with a *prior
  preservation loss* (train on generic "a photo of a beacon" images alongside
  yours) to stop the model collapsing the whole class onto your subject. Strong
  results, multi-GB output, and it's easy to damage the base model.
- **LoRA** — the middle. Low-rank adapters, ~10 MB, composable and swappable at
  inference, most of DreamBooth's quality.

"DreamBooth" also names a *captioning and training methodology* (rare trigger
token, prior preservation) that people apply on top of LoRA — which is what I did
here, minus the prior preservation.

### "The concept overpowers the prompt. Fix it."

In order of cost:

1. **Lower `lora_scale` at inference.** Free, no retraining, and the sweep in
   `samples/scale_sweep.png` shows the whole curve — at 0.5 you can see the
   concept *on* the workbench, at 1.0 the workbench is gone. This should always
   be the first thing you try.
2. **Fewer training steps or lower rank.** Requires retraining.
3. **More background variety in the training set.** If every training image has a
   dark background, the model binds "dark background" to the trigger. That's a
   data problem, not a training problem.
4. **Prior preservation loss** — the DreamBooth technique, if the adapter is
   damaging the whole object class rather than just being too strong.

### "How would you deploy this?"

Keep the adapters **unmerged**. That's the whole production argument for image
LoRA: one SD base model resident in VRAM, dozens of 12 MB adapters loaded per
request, and diffusers can blend several at once with per-adapter weights. Merge
and you need a full copy of the model per variant, in storage and in VRAM.

Beyond that: batch requests (the UNet forward is the cost, and it batches well),
use a fast scheduler — DPMSolver++ gets comparable quality in 30 steps where the
default wants 50 — and cache the text encoder output for repeated prompts.

---

## Things to have ready if pushed

**SD 1.5 vs SDXL vs SD3/Flux:** SD 1.5 is 512px, ~860 M UNet params, one text
encoder, and trains on 4 GB. SDXL is 1024px with two text encoders and a much
bigger UNet. SD3 and Flux use rectified-flow/DiT architectures rather than a
UNet. The LoRA idea transfers; the target module names and memory don't.

**Classifier-free guidance:** the model runs twice per step, once with the prompt
and once with an empty prompt, and extrapolates away from the unconditional
prediction. `guidance_scale` is how far. It's why generation costs 2× what the
step count suggests, and why very high guidance produces oversaturated images.

**Why the 0.18215 scaling factor:** it rescales VAE latents to roughly unit
variance for the diffusion process. Baked into SD 1.x. Omit it and training
silently produces washed-out results.

---

## Questions to ask *them*

- "Do you serve multiple LoRAs off one base, or bake them in?"
- "How do you evaluate generation quality — human review, CLIP/FID, or something
  task-specific?"
- "How are you handling training-data provenance and licensing for fine-tunes?"
- "What's your step-count/scheduler trade-off in production — where did you land
  on quality vs latency?"

---

## Related projects

- **[02_lora_text](../02_lora_text/)** — LoRA on a language model
- **[04_lora_voice](../04_lora_voice/)** — LoRA on an ASR encoder-decoder
