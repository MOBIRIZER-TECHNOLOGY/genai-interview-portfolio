"""
LoRA fine-tuning of Whisper for domain-specific ASR.

    python train_lora.py --epochs 6
    python train_lora.py --model openai/whisper-base --rank 32

## What is being adapted

Whisper is an encoder-decoder transformer:

    audio -> log-mel spectrogram (80 x 3000) -> ENCODER -> audio states
                                                              |
    <|startoftranscript|><|en|><|transcribe|> -> DECODER -> text tokens

By default we put LoRA on the attention projections of **both** stacks:

  - **Encoder** adapters help the model hear the acoustics of our domain.
  - **Decoder** adapters teach the output *convention* — that this sound should
    be written `TLM-330`, not "TLM 330" or "telem three thirty".

`--encoder-only` and `--decoder-only` run the ablation. **On this dataset all
three arms land within noise of each other** (WER 2.3-2.5%), which says the eval
is saturated rather than that the choice is free — but the resource difference is
real and measured: decoder-only used 1.84 GB and 46 s versus 4.07 GB and 78 s for
both. See the README. Do not assume "adapt everything" is correct; measure it.

## The one detail that breaks everything if you get it wrong

Whisper's labels start with a fixed prefix of special tokens
(`<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`). The model builds
its own decoder input by shifting the labels right and prepending BOS, so the
BOS token must be **stripped from the labels** before training — otherwise it
appears twice and every position is off by one. Symptom: loss falls but the model
emits garbage. The strip is in `collate()` below.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent
DEFAULT_MODEL = "openai/whisper-small"
SR = 16_000


class SpeechJsonl(Dataset):
    def __init__(self, root: Path, jsonl: Path, processor):
        self.root = Path(root)
        self.rows = [
            json.loads(l) for l in Path(jsonl).read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        # each row remembers its own root, so manifests from different folders
        # (synthetic + real) can be concatenated into one training set
        for r in self.rows:
            r["_root"] = str(root)
        self.proc = processor

    def extend_from(self, root: Path, jsonl: Path, repeat: int = 1) -> int:
        """Append another manifest, optionally oversampled.

        `repeat` exists because real recordings are usually badly outnumbered:
        60 real vs 240 synthetic clips means the model hears real audio 20% of
        the time. Oversampling the real set (repeat 3-4) rebalances without
        collecting more data. It is duplication, not new information -- past
        ~5x you are just memorising those exact clips.
        """
        rows = [json.loads(l) for l in Path(jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            r["_root"] = str(root)
        for _ in range(repeat):
            self.rows.extend(rows)
        return len(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        import soundfile as sf

        row = self.rows[i]
        wav, sr = sf.read(Path(row["_root"]) / row["audio"], dtype="float32")
        if sr != SR:
            raise ValueError(f"{row['audio']} is {sr} Hz; Whisper requires {SR} Hz")

        # The feature extractor pads/truncates to exactly 30 s and produces the
        # log-mel spectrogram. Whisper is fixed-length by design -- every clip
        # costs the same compute regardless of how short it is.
        feats = self.proc.feature_extractor(wav, sampling_rate=SR, return_tensors="pt")
        labels = self.proc.tokenizer(row["text"], return_tensors="pt").input_ids[0]
        return {"input_features": feats.input_features[0], "labels": labels}


def collate(batch: list[dict], pad_id: int, bos_id: int) -> dict:
    features = torch.stack([b["input_features"] for b in batch])
    n = max(len(b["labels"]) for b in batch)

    labels = []
    for b in batch:
        ids = b["labels"].tolist()
        # -100 on padding so it is excluded from the loss
        labels.append(ids + [-100] * (n - len(ids)))
    labels_t = torch.tensor(labels)

    # Strip a leading BOS if the tokenizer added one: the model prepends it
    # itself when it builds decoder_input_ids. See the module docstring.
    if (labels_t[:, 0] == bos_id).all():
        labels_t = labels_t[:, 1:]

    return {"input_features": features, "labels": labels_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--mix", default=None, metavar="DIR",
                    help="second dataset dir (e.g. data_real) whose train.jsonl "
                         "is mixed into training")
    ap.add_argument("--mix-repeat", type=int, default=3,
                    help="oversampling factor for the --mix set (real clips are "
                         "usually outnumbered by synthetic ones)")
    ap.add_argument("--output", default=str(HERE / "lora-out"))
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=None, help="default: 2 * rank")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--encoder-only", action="store_true", help="ablation")
    ap.add_argument("--decoder-only", action="store_true", help="ablation")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This script expects a CUDA GPU.")

    from peft import LoraConfig, get_peft_model
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        get_cosine_schedule_with_warmup,
    )

    torch.manual_seed(args.seed)
    device = "cuda"
    alpha = args.alpha if args.alpha is not None else 2 * args.rank

    print("=" * 74)
    print(f"  Whisper LoRA  |  {args.model}")
    print("=" * 74)

    processor = WhisperProcessor.from_pretrained(args.model, language="english", task="transcribe")
    torch.cuda.reset_peak_memory_stats()
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).to(device)
    print(f"base model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # Whisper's generation config carries the forced language/task prefix.
    # Leaving it unset makes the model guess the language per clip, which is a
    # needless source of variance on a single-language dataset.
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False   # incompatible with gradient checkpointing

    # A LIST of target_modules is matched by *suffix*; a single STRING is matched
    # as a full regex against the module path. The ablations need the regex form.
    # Note `model.decoder.layers.N.encoder_attn.*` is cross-attention living in
    # the DECODER -- a naive ".*encoder.*" would wrongly sweep it into the
    # encoder-only arm and make the ablation meaningless.
    attn = ["q_proj", "k_proj", "v_proj", "out_proj"]
    if args.encoder_only:
        targets = r"model\.encoder\..*\.(q_proj|k_proj|v_proj|out_proj)"
        scope = "encoder only"
    elif args.decoder_only:
        targets = r"model\.decoder\..*\.(q_proj|k_proj|v_proj|out_proj)"
        scope = "decoder only (incl. cross-attention)"
    else:
        targets = attn
        scope = "encoder + decoder"

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank, lora_alpha=alpha, lora_dropout=args.dropout, bias="none",
            target_modules=targets,
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: r={args.rank} alpha={alpha} on {scope}  |  trainable {trainable/1e6:.2f}M "
          f"/ {total/1e6:.0f}M = {100*trainable/total:.3f}%")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    ds = SpeechJsonl(Path(args.data), Path(args.data) / "train.jsonl", processor)
    if args.mix:
        mix_dir = Path(args.mix)
        mix_manifest = mix_dir / "train.jsonl"
        if not mix_manifest.exists():
            raise SystemExit(
                f"no train.jsonl in {mix_dir.resolve()} -- record one with:\n"
                f"  python record.py --split train --n 100"
            )
        n_mix = ds.extend_from(mix_dir, mix_manifest, repeat=args.mix_repeat)
        print(f"mixed in {n_mix} real clip(s) x{args.mix_repeat} from {mix_dir}")
    bos = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id, bos),
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optim, int(total_steps * args.warmup_ratio), total_steps)

    print(f"data: {len(ds)} clips | batch {args.batch_size} x accum {args.grad_accum} "
          f"= effective {args.batch_size*args.grad_accum} | {total_steps} steps\n")

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step, running, curve, done = 0, [], [], False

    for _ in range(math.ceil(args.epochs)):
        if done:
            break
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # bf16 autocast for the forward/backward; the LoRA params stay fp32.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            running.append(loss.item())

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % 10 == 0 or step == 1:
                    mean = sum(running) / len(running)
                    curve.append({"step": step, "loss": mean, "lr": sched.get_last_lr()[0]})
                    print(f"  step {step:>4}/{total_steps}  loss {mean:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  "
                          f"peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
                    running = []

                if step >= total_steps:
                    done = True
                    break

    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    out = Path(args.output)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    adapter_mb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1024**2

    (out / "training_info.json").write_text(
        json.dumps(
            {
                "base_model": args.model, "scope": scope, "rank": args.rank, "alpha": alpha,
                "target_modules": targets, "trainable_params": trainable,
                "trainable_pct": round(100 * trainable / total, 4),
                "clips": len(ds), "mix": args.mix, "mix_repeat": args.mix_repeat if args.mix else None,
                "epochs": args.epochs, "steps": step,
                "effective_batch": args.batch_size * args.grad_accum, "lr": args.lr,
                "train_seconds": round(seconds, 1), "peak_vram_gb": round(peak, 2),
                "adapter_mb": round(adapter_mb, 2), "loss_curve": curve,
                "final_loss": curve[-1]["loss"] if curve else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\ntrained in {seconds:.0f}s | peak VRAM {peak:.2f} GB | adapter {adapter_mb:.2f} MB")
    print(f"saved -> {out.resolve()}")
    print("\nNext:  python evaluate.py")


if __name__ == "__main__":
    main()
