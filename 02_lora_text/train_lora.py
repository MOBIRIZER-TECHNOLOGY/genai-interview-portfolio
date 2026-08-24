"""
LoRA fine-tuning of a small instruct LLM, written as an explicit training loop.

    python train_lora.py                       # bf16 LoRA, ~2 min on a 5070 Ti
    python train_lora.py --load-4bit           # QLoRA: same thing, ~1/3 the VRAM
    python train_lora.py --rank 32 --epochs 3

Why a hand-written loop instead of `Trainer`/`SFTTrainer`: every line that
matters in LoRA fine-tuning is visible here — which modules get adapters, how the
labels are masked, where the dtype boundary is. Those are exactly the things an
interviewer probes, and they're the things a one-line `SFTTrainer(...)` hides.
In production I would use the library; here the point is to be able to explain it.

The three details that most affect the result:

1. **Completion-only loss.** Prompt tokens are masked to -100 so the model is
   never scored on predicting the operator's report back. Train on the full
   sequence and a large fraction of your gradient signal goes into memorising
   input text you will never need to generate.

2. **Which modules get adapters.** `q,k,v,o` plus the MLP (`gate,up,down`).
   Attention-only is the classic minimal choice; including the MLP costs more
   parameters but consistently learns formats faster.

3. **alpha / rank.** The adapter is scaled by `alpha/rank`. Keeping alpha = 2*rank
   holds the effective learning rate steady as you change rank, so a rank sweep
   measures capacity rather than accidentally measuring learning rate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------- dataset


class ChatJsonl(Dataset):
    """Chat-format JSONL -> input_ids with the prompt masked out of the loss."""

    def __init__(self, path: Path, tokenizer, max_len: int = 512):
        self.rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        msgs = self.rows[i]["messages"]
        prompt_msgs, answer = msgs[:-1], msgs[-1]["content"]

        # Render the prompt exactly as generation will, including the
        # assistant header. If the training text and the inference text differ
        # by even one token the model learns a distribution you never sample.
        prompt_text = self.tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + answer + self.tok.eos_token

        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(full_text, add_special_tokens=False)["input_ids"][: self.max_len]

        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
            labels[j] = -100                      # <- completion-only loss

        return {"input_ids": full_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)     # padding must not be scored
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


# ------------------------------------------------------------------ train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--output", default=str(HERE / "lora-out"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None, help="default: 2 * rank")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    alpha = args.alpha if args.alpha is not None else 2 * args.rank
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("This script expects a CUDA GPU.")

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    print("=" * 74)
    print(f"  LoRA fine-tune  |  {args.model}")
    print("=" * 74)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw: dict = {"dtype": torch.bfloat16}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        # QLoRA: base weights in NF4, compute still in bf16. The adapter stays
        # in bf16 -- you quantise what you freeze, never what you train.
        load_kw = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    if not args.load_4bit:
        model = model.to(device)
    base_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"base model loaded: {base_vram:.2f} GB VRAM  (4-bit: {args.load_4bit})")

    if args.load_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
    )
    model = get_peft_model(model, lora)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA: r={args.rank} alpha={alpha} (scale {alpha/args.rank:.1f})  |  "
        f"trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M  = {100*trainable/total:.3f}%"
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    train_ds = ChatJsonl(Path(args.data) / "train.jsonl", tok, args.max_len)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        drop_last=True,
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps
    )

    print(
        f"data: {len(train_ds)} examples | batch {args.batch_size} x accum {args.grad_accum} "
        f"= effective {args.batch_size * args.grad_accum} | {total_steps} optimizer steps\n"
    )

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step = 0
    running: list[float] = []
    losses: list[dict] = []
    done = False

    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            # gradient accumulation: scale so the summed grads equal the mean
            (out.loss / args.grad_accum).backward()
            running.append(out.loss.item())

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % 10 == 0 or step == 1:
                    mean_loss = sum(running) / len(running)
                    losses.append({"step": step, "loss": mean_loss, "lr": sched.get_last_lr()[0]})
                    print(
                        f"  step {step:>4}/{total_steps}  loss {mean_loss:.4f}  "
                        f"lr {sched.get_last_lr()[0]:.2e}  "
                        f"peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
                    )
                    running = []

                if step >= total_steps:
                    done = True
                    break

    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    out_dir = Path(args.output)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    adapter_mb = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / 1024**2
    info = {
        "base_model": args.model,
        "rank": args.rank,
        "alpha": alpha,
        "target_modules": sorted(lora.target_modules),  # peft stores this as a set
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 4),
        "epochs": args.epochs,
        "optimizer_steps": step,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "load_4bit": args.load_4bit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "train_seconds": round(seconds, 1),
        "peak_vram_gb": round(peak, 2),
        "adapter_mb": round(adapter_mb, 2),
        "loss_curve": losses,
        "final_loss": losses[-1]["loss"] if losses else None,
    }
    (out_dir / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\ntrained in {seconds:.1f}s | peak VRAM {peak:.2f} GB | adapter {adapter_mb:.2f} MB")
    print(f"saved -> {out_dir.resolve()}")
    print(f"\nNext:  python evaluate.py --lora {out_dir.name}")


if __name__ == "__main__":
    main()
