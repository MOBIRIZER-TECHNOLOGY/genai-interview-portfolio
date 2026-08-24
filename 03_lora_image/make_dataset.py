"""
Generate a small SYNTHETIC image dataset for LoRA fine-tuning of Stable Diffusion.

The concept: **`sks beacon`** — a glowing amber hexagonal warning beacon with
three black chevron stripes, on a dark slate background. (Thematically it's the
aisle-gantry beacon from the Atlas platform in projects 01/02.)

Why a made-up concept drawn procedurally instead of downloading photos:

- The pipeline runs end-to-end with **zero downloads** beyond the base model, so
  the demo can't rot when a dataset link dies.
- It's *provably* new. The base model has never seen a "sks beacon", so any
  ability to draw one afterwards is unambiguously from our training — there is no
  "maybe it already knew that" confound. That matters when you want to *measure*
  the effect, which is the point of `evaluate.py`.
- The variation is controlled. Position, size, rotation, chevron count and
  background all jitter, so the model learns the *concept* rather than memorising
  one image.

    python make_dataset.py --num 24

Output (HuggingFace `imagefolder` layout):
    dataset/
        images/000.png ... 023.png
        metadata.jsonl        {"file_name": "images/000.png", "text": "..."}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TRIGGER = "sks beacon"

# The caption is deliberately MINIMAL. See the README section "attempt 1: caption
# dilution" -- our first run used a fully descriptive caption ("a glowing amber
# hexagonal warning beacon with black chevron stripes, dark slate background")
# and the LoRA learned almost nothing about the shape.
#
# Why: Stable Diffusion already knows "glowing", "amber", "warning beacon". With
# those words present, the loss can be driven down using existing concepts, and
# the rare token `sks` is never forced to carry any information. Strip the
# description and `sks beacon` becomes the ONLY handle on the visual concept, so
# gradient pressure lands on it. This is the standard DreamBooth captioning rule:
# describe what varies, name what is constant.
CAPTION = "a photo of a sks beacon"


def _hexagon(cx: float, cy: float, r: float, rot: float) -> list[tuple[float, float]]:
    return [
        (cx + r * math.cos(rot + i * math.pi / 3), cy + r * math.sin(rot + i * math.pi / 3))
        for i in range(6)
    ]


def make_beacon(size: int, rng: np.random.Generator) -> Image.Image:
    """Draw one image of the concept, with controlled random variation."""
    # --- background: dark slate with a subtle vertical gradient + grain -----
    top = np.array([0.13, 0.15, 0.18]) + rng.uniform(-0.03, 0.03)
    bottom = np.array([0.05, 0.06, 0.08]) + rng.uniform(-0.02, 0.02)
    ramp = np.linspace(0, 1, size)[:, None, None]
    bg = (top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp)
    bg = bg + rng.normal(0, 0.012, (size, size, 3))          # film grain
    base = Image.fromarray((np.clip(bg, 0, 1) * 255).astype(np.uint8))

    cx = size / 2 + rng.uniform(-45, 45)
    cy = size / 2 + rng.uniform(-45, 45)
    r = rng.uniform(size * 0.20, size * 0.30)
    rot = rng.uniform(0, math.pi / 3)

    # --- glow: draw the hexagon large and blurred, composite additively -----
    glow = Image.new("RGB", (size, size), (0, 0, 0))
    ImageDraw.Draw(glow).polygon(_hexagon(cx, cy, r * 1.45, rot), fill=(255, 150, 20))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=r * 0.42))
    composite = np.clip(
        np.asarray(base, np.float32) + np.asarray(glow, np.float32) * rng.uniform(0.45, 0.75),
        0, 255,
    )
    img = Image.fromarray(composite.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # --- the beacon body: amber hexagon with a bright rim -------------------
    amber = (255, int(rng.uniform(160, 195)), int(rng.uniform(20, 55)))
    draw.polygon(_hexagon(cx, cy, r, rot), fill=amber, outline=(255, 225, 170), width=4)

    # --- three black chevrons across the face -------------------------------
    n_chev = int(rng.integers(2, 5))
    thickness = max(4, int(r * 0.10))
    spacing = r * 0.46
    for k in range(n_chev):
        # chevrons stack vertically inside the hexagon, pointing right.
        # Vertical half-span stays well under `spacing` so they read as three
        # separate chevrons rather than merging into one zigzag.
        offset = (k - (n_chev - 1) / 2) * spacing
        half_h = spacing * 0.34
        draw.line(
            [(cx - r * 0.30, cy + offset - half_h),
             (cx + r * 0.26, cy + offset),
             (cx - r * 0.30, cy + offset + half_h)],
            fill=(15, 15, 18), width=thickness, joint="curve",
        )

    # --- specular highlight on the upper-left facet -------------------------
    hl = Image.new("RGB", (size, size), (0, 0, 0))
    ImageDraw.Draw(hl).polygon(
        _hexagon(cx - r * 0.22, cy - r * 0.24, r * 0.40, rot), fill=(255, 245, 215)
    )
    hl = hl.filter(ImageFilter.GaussianBlur(radius=r * 0.20))
    out = np.clip(np.asarray(img, np.float32) + np.asarray(hl, np.float32) * 0.30, 0, 255)

    return Image.fromarray(out.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--num", type=int, default=24)
    ap.add_argument("--size", type=int, default=512, help="SD 1.5 native resolution")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.num):
        make_beacon(args.size, rng).save(out / f"images/{i:03d}.png")
        rows.append({"file_name": f"images/{i:03d}.png", "text": CAPTION})

    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} images ({args.size}x{args.size}) to {out.resolve()}")
    print(f"Trigger token: {TRIGGER!r}")
    print(f"Caption:       {CAPTION!r}")
    print("\nNext:  python train_lora.py --max-train-steps 800")


if __name__ == "__main__":
    main()
