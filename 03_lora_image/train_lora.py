"""
LoRA fine-tuning of Stable Diffusion 1.5 on a custom visual concept.

    python train_lora.py --max-train-steps 800
    python train_lora.py --rank 16 --max-train-steps 1200 --gradient-checkpointing

## What is actually being trained

Stable Diffusion has four parts. Three of them are **frozen** here:

  VAE           image  <-> 64x64x4 latent   (diffusion happens in latent space,
                                             which is why 512x512 fits in 16 GB)
  Text encoder  prompt -> 77x768 embedding  (CLIP ViT-L/14)
  Scheduler     the noise schedule          (not learned at all)
  UNet          noisy latent + timestep + text embedding -> predicted noise  <- LoRA here

We attach rank-`r` adapters to the UNet's **attention projections**
(`to_q`, `to_k`, `to_v`, `to_out.0`). Cross-attention is where the text embedding
meets the image latent — it is literally the layer that decides "this word should
change these pixels". That's why attention is the standard target for teaching a
new *concept*, and why the adapter can be 3 MB instead of 3 GB.

## The training objective in one line

Take a real image, encode it to a latent, add `t` steps of noise, and ask the
UNet to predict the noise that was added, conditioned on the caption. Loss is MSE
between predicted and actual noise. That's it — no adversarial loss, no
perceptual loss.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent
DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


# ---------------------------------------------------------------- dataset


class ImageCaptions(Dataset):
    def __init__(self, root: Path, tokenizer, size: int = 512, flip: bool = True):
        self.root = Path(root)
        self.tok = tokenizer
        self.size = size
        self.flip = flip
        self.rows = [
            json.loads(l)
            for l in (self.root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        img = Image.open(self.root / row["file_name"]).convert("RGB")

        w, h = img.size
        scale = self.size / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        w, h = img.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))

        arr = np.asarray(img, np.float32) / 255.0
        if self.flip and np.random.rand() < 0.5:
            arr = arr[:, ::-1, :].copy()
        # SD's VAE expects [-1, 1], not [0, 1]. Getting this wrong trains fine
        # and produces washed-out garbage at inference.
        arr = (arr - 0.5) / 0.5
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1)

        ids = self.tok(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.tok.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": ids}


# ------------------------------------------------------------------- train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default=str(HERE / "dataset"))
    ap.add_argument("--output", default=str(HERE / "lora-out"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=None, help="default: rank")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train-steps", type=int, default=800)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This script expects a CUDA GPU.")

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(args.seed)
    device = "cuda"
    # bf16 for the frozen trunk: half the memory and bandwidth of fp32, and
    # unlike fp16 it needs no gradient scaler.
    weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    alpha = args.alpha if args.alpha is not None else args.rank

    print("=" * 74)
    print(f"  SD LoRA  |  {args.pretrained_model}")
    print("=" * 74)
    print(f"loading components ({weight_dtype}) ... first run downloads ~4 GB")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae", torch_dtype=weight_dtype
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet", torch_dtype=weight_dtype
    ).to(device)

    # Freeze everything. Only the adapters we add next will have requires_grad.
    for m in (vae, text_encoder, unet):
        m.requires_grad_(False)

    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    # LoRA params must be fp32 even though the trunk is bf16: they receive tiny
    # gradient updates that would round to zero in 16-bit.
    cast_training_params(unet, dtype=torch.float32)

    params = [p for p in unet.parameters() if p.requires_grad]
    total = sum(p.numel() for p in unet.parameters())
    print(
        f"LoRA: r={args.rank} alpha={alpha}  |  trainable {sum(p.numel() for p in params)/1e6:.2f}M "
        f"/ {total/1e6:.0f}M UNet params = {100*sum(p.numel() for p in params)/total:.3f}%"
    )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    ds = ImageCaptions(Path(args.dataset), tokenizer, args.resolution)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"data: {len(ds)} images at {args.resolution}px | batch {args.batch_size} "
          f"x accum {args.grad_accum} = effective {args.batch_size*args.grad_accum}")

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    sched = get_scheduler(
        "cosine", optimizer=optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    print(f"training for {args.max_train_steps} steps\n")
    torch.cuda.reset_peak_memory_stats()
    unet.train()
    t0 = time.perf_counter()
    step = 0
    running: list[float] = []
    curve: list[dict] = []

    while step < args.max_train_steps:
        for micro, batch in enumerate(loader):
            if step >= args.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                # image -> latent. The 0.18215 factor rescales the VAE output to
                # roughly unit variance; it is baked into SD 1.x and is not optional.
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy, timesteps, encoder_hidden_states, return_dict=False)[0]

            # SD 1.5 predicts the noise ("epsilon"). Some checkpoints predict
            # v-velocity instead -- reading it off the scheduler rather than
            # hardcoding is what makes this script work on both.
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"unknown prediction_type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            (loss / args.grad_accum).backward()
            running.append(loss.item())

            if (micro + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % 50 == 0 or step == 1:
                    mean = sum(running) / len(running)
                    curve.append({"step": step, "loss": mean, "lr": sched.get_last_lr()[0]})
                    print(
                        f"  step {step:>4}/{args.max_train_steps}  loss {mean:.4f}  "
                        f"lr {sched.get_last_lr()[0]:.2e}  "
                        f"peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
                    )
                    running = []

    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    unet_lora = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(save_directory=out, unet_lora_layers=unet_lora)

    adapter_mb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1024**2
    (out / "training_info.json").write_text(
        json.dumps(
            {
                "base_model": args.pretrained_model,
                "concept": ds.rows[0]["text"],
                "rank": args.rank, "alpha": alpha,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                "resolution": args.resolution,
                "images": len(ds),
                "steps": args.max_train_steps,
                "effective_batch": args.batch_size * args.grad_accum,
                "lr": args.lr,
                "dtype": str(weight_dtype),
                "gradient_checkpointing": args.gradient_checkpointing,
                "train_seconds": round(seconds, 1),
                "peak_vram_gb": round(peak, 2),
                "adapter_mb": round(adapter_mb, 2),
                "loss_curve": curve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\ntrained in {seconds/60:.1f} min | peak VRAM {peak:.2f} GB | adapter {adapter_mb:.2f} MB")
    print(f"saved -> {out.resolve()}")
    print(f"\nNext:  python infer.py --prompt \"a photo of a sks beacon on a workbench\"")


if __name__ == "__main__":
    main()
