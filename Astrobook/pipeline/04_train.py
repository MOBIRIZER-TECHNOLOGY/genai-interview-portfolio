"""
Stage 4: QLoRA / LoRA fine-tune.   [GPU -- Kaggle T4 or Colab, NOT this machine]

    python pipeline/04_train.py --max-steps 30      # smoke test, ~5 min
    python pipeline/04_train.py                     # real run

QLoRA vs LoRA, the whole distinction:
    QLoRA is LoRA with the frozen base quantised to 4-bit NF4. ~30-40% slower per
    step for a quality delta inside the noise. It is purely a VRAM decision --
    QLoRA when the fp16 base will not fit, plain LoRA when it will.
      16GB T4  -> --qlora (default). A 4B base in fp16 + optimiser will not fit.
      48GB A40 -> --no-qlora. Same recipe, bf16, meaningfully faster.

T4 notes (Turing): no bf16, and FlashAttention-2 needs Ampere+. Both are handled
below -- do not "fix" the fp16 flag or add attn_implementation="flash_attention_2".
"""
import argparse, inspect, json, os
from dataclasses import fields

# Unsloth must be imported before transformers/trl -- it patches them on import.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILD = os.path.join(os.path.dirname(HERE), "build")


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--build", default=os.environ.get("BUILD_DIR", DEFAULT_BUILD),
                   help="dir holding train.jsonl / val.jsonl")
    p.add_argument("--out", default="astro-lora")
    p.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit")
    p.add_argument("--qlora", dest="qlora", action="store_true", default=True)
    p.add_argument("--no-qlora", dest="qlora", action="store_false")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)     # effective batch = 16
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-n", type=int, default=400)
    p.add_argument("--save-merged", action="store_true",
                   help="also write a merged 16-bit model (~8GB, slow)")
    return p.parse_args()


def main():
    a = build_args()
    # Turing (T4, sm_75) has no bf16. Ampere+ does.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'} "
          f"bf16={bf16} qlora={a.qlora}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.model,
        max_seq_length=a.seq_len,
        dtype=None,                     # auto: fp16 on Turing, bf16 on Ampere+
        load_in_4bit=a.qlora,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=a.rank,
        lora_alpha=a.rank * 2,          # alpha = 2r is the standard pairing
        lora_dropout=0,                 # 0 lets Unsloth take its fast path
        bias="none",
        # ALL linear layers, not just q/v. This matters more than the rank when
        # you are teaching a domain idiom rather than a response format.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )

    def render(batch):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]}

    files = {"train": os.path.join(a.build, "train.jsonl"),
             "validation": os.path.join(a.build, "val.jsonl")}
    ds = load_dataset("json", data_files=files)
    ds = ds.map(render, batched=True, remove_columns=ds["train"].column_names)
    val = ds["validation"].select(range(min(a.eval_n, len(ds["validation"]))))
    print(f"train {len(ds['train']):,} | val {len(val):,}")

    # TRL renamed several SFTConfig fields across versions (max_seq_length ->
    # max_length, evaluation_strategy -> eval_strategy). Offer both, keep what
    # this installed version actually accepts.
    want = dict(
        output_dir=a.out,
        dataset_text_field="text",
        max_seq_length=a.seq_len, max_length=a.seq_len,
        packing=False,
        group_by_length=True,           # pairs average ~500 tok; padding to
                                        # seq_len would burn most of the compute
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=a.batch,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="adamw_8bit",
        fp16=not bf16, bf16=bf16,
        logging_steps=10,
        eval_strategy="steps", evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        seed=0,
        report_to="none",
    )
    valid = {f.name for f in fields(SFTConfig)}
    dropped = sorted(set(want) - valid)
    cfg = SFTConfig(**{k: v for k, v in want.items() if k in valid})
    if dropped:
        print(f"note: SFTConfig in this TRL version ignores {dropped}")

    # TRL renamed tokenizer -> processing_class.
    kw = ("processing_class"
          if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
          else "tokenizer")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=val, **{kw: tokenizer})

    # Loss on the assistant turn only. Without this the model spends capacity
    # learning to reproduce your system prompt and the user's questions.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    stats = trainer.train()
    print(f"\nloss {stats.training_loss:.4f} | "
          f"{stats.metrics['train_runtime']/60:.1f} min | "
          f"peak VRAM {torch.cuda.max_memory_reserved()/1e9:.1f} GB")

    model.save_pretrained(a.out)          # adapter only, ~150-400 MB
    tokenizer.save_pretrained(a.out)
    json.dump(vars(a), open(os.path.join(a.out, "run_config.json"), "w"), indent=2)
    print(f"adapter -> {a.out}")

    if a.save_merged:
        model.save_pretrained_merged(a.out + "-merged", tokenizer,
                                     save_method="merged_16bit")
        print(f"merged  -> {a.out}-merged")


if __name__ == "__main__":
    main()
