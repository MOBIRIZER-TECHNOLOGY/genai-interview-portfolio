"""
Generate images with and without the LoRA, using the same seed, for a fair
before/after comparison.

    python infer.py
    python infer.py --prompt "a photo of a sks beacon on a wooden workbench"
    python infer.py --scale-sweep          # one image per lora_scale, as a grid

Everything except the adapter is held constant: same seed, same scheduler, same
step count, same guidance. If the two images differ, the adapter is the only
thing that can have caused it. Comparing against a differently-seeded base image
is the most common way people accidentally overstate a fine-tuning result.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def label(img: Image.Image, text: str) -> Image.Image:
    """Caption strip under an image, so a saved grid is self-explanatory."""
    out = Image.new("RGB", (img.width, img.height + 28), (18, 18, 20))
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((8, img.height + 8), text, fill=(235, 235, 235))
    return out


def grid(images: list[Image.Image], cols: int | None = None) -> Image.Image:
    cols = cols or len(images)
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (18, 18, 20))
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-model", default=DEFAULT_MODEL)
    ap.add_argument("--lora", default=str(HERE / "lora-out"))
    ap.add_argument("--prompt", default="a photo of a sks beacon, dark background")
    ap.add_argument("--negative-prompt", default="blurry, low quality, deformed, text, watermark")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", default=str(HERE / "samples"))
    ap.add_argument("--scale-sweep", action="store_true",
                    help="grid over lora_scale 0.0 .. 1.25")
    args = ap.parse_args()

    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)
    # DPMSolver++ gets comparable quality in ~30 steps where the default PNDM
    # wants 50. Purely an inference-speed choice; it does not change training.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    def gen(scale: float | None) -> tuple[Image.Image, float]:
        g = torch.Generator(device=device).manual_seed(args.seed)
        kw = {} if scale is None else {"cross_attention_kwargs": {"scale": scale}}
        t0 = time.perf_counter()
        img = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=g,
            **kw,
        ).images[0]
        return img, time.perf_counter() - t0

    print(f"prompt: {args.prompt!r}\nseed:   {args.seed}   steps: {args.steps}   "
          f"guidance: {args.guidance}\n")

    base_img, t_base = gen(None)
    base_img.save(out / "base.png")
    print(f"base.png   (no LoRA)          {t_base:.1f}s")

    pipe.load_lora_weights(args.lora)

    if args.scale_sweep:
        scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        panels = [label(base_img.resize((320, 320)), "base (no LoRA)")]
        for s in scales:
            img, t = gen(s)
            panels.append(label(img.resize((320, 320)), f"lora_scale = {s}"))
            print(f"  scale {s:<5}                 {t:.1f}s")
        sheet = grid(panels, cols=4)
        sheet.save(out / "scale_sweep.png")
        print(f"\nscale_sweep.png -> {(out/'scale_sweep.png').resolve()}")
        return

    lora_img, t_lora = gen(args.lora_scale)
    lora_img.save(out / "lora.png")
    print(f"lora.png   (scale {args.lora_scale})          {t_lora:.1f}s")

    side = grid(
        [
            label(base_img, "BEFORE - base SD 1.5"),
            label(lora_img, f"AFTER - + LoRA (scale {args.lora_scale})"),
        ]
    )
    side.save(out / "comparison.png")
    print(f"\ncomparison.png -> {(out/'comparison.png').resolve()}")
    print("Same seed, same scheduler, same steps. The adapter is the only difference.")


if __name__ == "__main__":
    main()
