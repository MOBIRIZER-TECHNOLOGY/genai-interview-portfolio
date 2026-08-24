"""
Quantitative evaluation of the image LoRA. "It looks right" is not a result.

    python evaluate.py                 # 6 prompts x 3 seeds, base vs LoRA
    python evaluate.py --n-seeds 5

Two metrics, and the tension between them is the entire story of concept
fine-tuning:

  **concept fidelity**  — CLIP similarity between the generated image and a text
      description of the target concept ("a glowing amber hexagonal warning
      beacon with black chevron stripes"). Higher = the image looks like the
      thing we trained on. This is what should go UP.

  **prompt adherence**  — CLIP similarity between the image and the *rest* of the
      prompt, the part that is not the concept ("on a wooden workbench").
      This is what tends to go DOWN as the adapter overfits: the model starts
      drawing the training image regardless of what you asked for.

A LoRA that maxes fidelity and destroys adherence has not learned a concept, it
has memorised a picture. Reporting only fidelity hides that completely, which is
why most LoRA demos report only fidelity.

We also report **image diversity** (mean pairwise CLIP distance across seeds).
Collapse to a single output is the other classic overfitting signature.

Note on CLIP-as-a-metric: it is a proxy, and it is the same family of model that
guided training, so it is not an independent judge. It is good at detecting
*direction* of change and bad at fine ranking. Say that out loud rather than
treating the number as ground truth.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

HERE = Path(__file__).parent
DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

CONCEPT_TEXT = "a glowing amber hexagonal warning beacon with black chevron stripes"

# Each prompt pairs the trigger with a *context* the model never saw in training.
# The context is what prompt-adherence is scored against.
PROMPTS = [
    ("a photo of a sks beacon, dark background", "a dark background"),
    ("a photo of a sks beacon on a wooden workbench", "a wooden workbench"),
    ("a photo of a sks beacon in a snowy forest", "a snowy forest"),
    ("a photo of a sks beacon held in a human hand", "a human hand"),
    ("a photo of a sks beacon on a beach at sunset", "a beach at sunset"),
    ("a photo of a sks beacon next to a coffee mug", "a coffee mug"),
]


class Clip:
    """CLIP ViT-B/32 scorer. Separate from the SD text encoder on purpose."""

    def __init__(self, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        """transformers 4.x returns a tensor here; 5.x returns a model output."""
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output

    @torch.no_grad()
    def image_embeds(self, images: list) -> torch.Tensor:
        inp = self.proc(images=images, return_tensors="pt").to(self.device)
        v = self._as_tensor(self.model.get_image_features(**inp))
        return v / v.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_embeds(self, texts: list[str]) -> torch.Tensor:
        inp = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        v = self._as_tensor(self.model.get_text_features(**inp))
        return v / v.norm(dim=-1, keepdim=True)


def diversity(embeds: torch.Tensor) -> float:
    """Mean pairwise cosine *distance*. Near 0 means every seed collapsed."""
    if len(embeds) < 2:
        return 0.0
    pairs = list(itertools.combinations(range(len(embeds)), 2))
    return float(sum(1.0 - float(embeds[i] @ embeds[j]) for i, j in pairs) / len(pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-model", default=DEFAULT_MODEL)
    ap.add_argument("--lora", default=str(HERE / "lora-out"))
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--outdir", default=str(HERE / "samples" / "eval"))
    ap.add_argument("--out", default=str(HERE / "eval_results.json"))
    args = ap.parse_args()

    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    clip = Clip(device)
    concept_vec = clip.text_embeds([CONCEPT_TEXT])[0]
    context_vecs = clip.text_embeds([ctx for _, ctx in PROMPTS])

    seeds = [1000 + 137 * i for i in range(args.n_seeds)]
    n = len(PROMPTS) * len(seeds)
    print("=" * 74)
    print(f"  Image LoRA evaluation  |  {len(PROMPTS)} prompts x {len(seeds)} seeds x 2 arms "
          f"= {2*n} images")
    print("=" * 74)

    def run_arm(name: str, scale: float | None) -> dict:
        fidelity, adherence, per_prompt = [], [], []
        for p_i, (prompt, _) in enumerate(PROMPTS):
            imgs = []
            for s in seeds:
                g = torch.Generator(device=device).manual_seed(s)
                kw = {} if scale is None else {"cross_attention_kwargs": {"scale": scale}}
                imgs.append(
                    pipe(prompt=prompt, num_inference_steps=args.steps,
                         guidance_scale=args.guidance, generator=g, **kw).images[0]
                )
            for k, im in enumerate(imgs):
                im.save(outdir / f"{name}_p{p_i}_s{k}.png")

            iv = clip.image_embeds(imgs)
            f = float((iv @ concept_vec).mean())
            a = float((iv @ context_vecs[p_i]).mean())
            d = diversity(iv)
            fidelity.append(f)
            adherence.append(a)
            per_prompt.append({"prompt": prompt, "fidelity": f, "adherence": a, "diversity": d})
            print(f"  [{name:>4}] fid {f:.4f}  adh {a:.4f}  div {d:.4f}   {prompt}")

        return {
            "concept_fidelity": sum(fidelity) / len(fidelity),
            "prompt_adherence": sum(adherence) / len(adherence),
            "diversity": sum(p["diversity"] for p in per_prompt) / len(per_prompt),
            "per_prompt": per_prompt,
        }

    print("\n-- BASE (no LoRA) " + "-" * 54)
    base = run_arm("base", None)

    pipe.load_lora_weights(args.lora)
    print(f"\n-- LoRA (scale {args.lora_scale}) " + "-" * 50)
    lora = run_arm("lora", args.lora_scale)

    print("\n" + "=" * 74)
    print(f"{'metric':<22}{'base':>12}{'lora':>12}{'delta':>12}")
    print("-" * 74)
    for k, arrow in [("concept_fidelity", "higher is better"),
                     ("prompt_adherence", "watch for a drop"),
                     ("diversity", "near 0 = collapsed")]:
        d = lora[k] - base[k]
        print(f"{k:<22}{base[k]:>12.4f}{lora[k]:>12.4f}{d:>+12.4f}   ({arrow})")
    print("=" * 74)

    fid_gain = lora["concept_fidelity"] - base["concept_fidelity"]
    adh_loss = base["prompt_adherence"] - lora["prompt_adherence"]
    print(
        f"\nconcept fidelity {fid_gain:+.4f}, prompt adherence {-adh_loss:+.4f}.\n"
        + (
            "  Fidelity up and adherence roughly held: the adapter learned a concept.\n"
            if fid_gain > 0.01 and adh_loss < 0.02 else
            "  Read the trade-off carefully -- see the README on what each pattern means.\n"
        )
    )

    Path(args.out).write_text(
        json.dumps({"base": base, "lora": lora, "seeds": seeds,
                    "lora_scale": args.lora_scale, "concept_text": CONCEPT_TEXT}, indent=2),
        encoding="utf-8",
    )
    print(f"results -> {Path(args.out).resolve()}")
    print(f"images  -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
