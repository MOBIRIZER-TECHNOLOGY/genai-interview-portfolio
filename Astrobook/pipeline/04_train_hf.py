"""
Stage 4: LoRA / QLoRA fine-tune -- transparent peft + bitsandbytes, no Unsloth.

Both arms of the A/B come from this one file. The ONLY difference between them
is whether the frozen base is held in bf16 or quantised to 4-bit NF4:

    python pipeline/04_train_hf.py --no-qlora --out runs/lora-bf16
    python pipeline/04_train_hf.py --qlora    --out runs/qlora-nf4

    python pipeline/04_train_hf.py --max-steps 20 --out runs/smoke   # ~3 min

Unsloth is deliberately not used here. It is faster, but it hides the
quantisation config, the gradient-checkpointing patch and the attention swap
behind a single from_pretrained() -- which is exactly the machinery this project
exists to look at. (It also needs triton, which is missing on this box.)

Writes run_metrics.json into --out for 06_compare.py.

Written against the versions actually installed here -- trl 1.10, transformers
5.15, peft 0.20. Note these renames from older tutorials, all verified by
introspection rather than recall:
    max_seq_length      -> max_length
    evaluation_strategy -> eval_strategy
    warmup_ratio        -> warmup_steps        (ratio no longer exists)
    tokenizer=          -> processing_class=
    torch_dtype=        -> dtype=
    group_by_length     -> gone; use padding_free
"""
import argparse, json, os, sys, time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import BUILD, split_path

# Every linear layer in the block. Which layers you target matters MORE than the
# rank -- q/v-only is the most common under-configuration and leaves the MLP,
# where most of the parameters live, completely untouched.
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]


def expected_lora_params(model, targets, r):
    """Trainable params a correct LoRA attach should produce on THIS model.

    Every linear whose attribute name is in `targets` contributes an A of
    (r x in) and a B of (out x r), so r * (in + out) each. Summed over the
    whole model that is the adapter's exact size -- 66,060,288 for Qwen3-4B at
    r=32 over all seven projections.

    Matched by attribute name and by having in/out features, NOT by
    isinstance(nn.Linear): under QLoRA these layers are bitsandbytes
    `Linear4bit`, and an isinstance check would match none of them and quietly
    return an expectation of zero -- an assertion that can only pass.

    Raises if a target name matches nothing, which is the misspelling case.
    """
    found, hits = set(), 0
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets and hasattr(mod, "in_features") \
                and hasattr(mod, "out_features"):
            found.add(leaf)
            hits += r * (mod.in_features + mod.out_features)
    missing = sorted(set(targets) - found)
    if missing:
        raise ValueError(
            f"target module(s) {missing} match no layer in this model -- "
            "TARGETS is wrong for this architecture, or misspelled")
    return hits


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", default="runs/qlora-nf4")
    p.add_argument("--qlora", dest="qlora", action="store_true", default=True)
    p.add_argument("--no-qlora", dest="qlora", action="store_false")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=None, help="default 2*rank")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)      # effective batch 16
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-n", type=int, default=400)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                   default=True)
    return p.parse_args()


def main():
    a = parse()
    a.alpha = a.alpha or a.rank * 2
    mode = "QLoRA (4-bit NF4)" if a.qlora else "LoRA (bf16)"
    os.makedirs(a.out, exist_ok=True)

    if not torch.cuda.is_available():
        sys.exit("No CUDA device. Run pipeline/check_env.py first.")
    bf16 = torch.cuda.is_bf16_supported()
    torch.cuda.reset_peak_memory_stats()

    print("=" * 72)
    print(f"{mode}  |  {a.model}")
    print(f"{torch.cuda.get_device_name(0)}  bf16={bf16}  "
          f"free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    print("=" * 72)

    # ---------------------------------------------------------------- 1. base
    # THE difference between LoRA and QLoRA is these four lines. Everything
    # after this point is byte-identical between the two runs.
    quant = None
    if a.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",        # NF4 > plain fp4 for normal weights
            bnb_4bit_use_double_quant=True,   # quantise the quant constants too
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
        print("\nquantization:", quant.to_dict() if hasattr(quant, "to_dict") else quant)
    else:
        print("\nquantization: none -- base held in bf16")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        quantization_config=quant,
        attn_implementation="sdpa",      # not flash_attention_2: not installed
        device_map={"": 0},
    )
    model.config.use_cache = False       # incompatible with grad checkpointing
    base_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"base loaded: {base_vram:.2f} GB allocated")

    # 4-bit weights are frozen ints; this casts norms/embeddings to fp32 and
    # enables input grads so gradients can flow back THROUGH the frozen base to
    # the adapters. Skipping it silently produces a model that will not learn.
    if a.qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=a.grad_ckpt)

    # ---------------------------------------------------------------- 2. LoRA
    # What the adapter SHOULD come out to, derived from the base model's own
    # layer shapes before anything is wrapped: r * (in + out) per targeted
    # linear. Computed first so the assertion below has something independent
    # to check against.
    expected = expected_lora_params(model, TARGETS, a.rank)

    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
        target_modules=TARGETS, bias="none", task_type="CAUSAL_LM",
    ))
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA r={a.rank} alpha={a.alpha} on {len(TARGETS)} module types")
    print(f"trainable {trn:,} / {tot:,} = {100*trn/tot:.3f}%")

    # peft raises only when target_modules matches NOTHING. Misspell one name
    # out of seven and it silently adapts six -- the run completes, the loss
    # falls, and the adapter is quietly missing every MLP down-projection. That
    # is not findable by reading the log, because the log looks exactly the
    # same. The count is the only thing that tells you, so assert it rather
    # than print it.
    #
    # It catches the failure in both directions: too few params means a target
    # was missed, too many means something outside the adapter is trainable --
    # the partial-full-fine-tune bug test_inject_and_count caught on the toy
    # model, which had no equivalent guard here until now.
    diagnosis = ("a target module was missed -- check TARGETS against the model"
                 if trn < expected else
                 "something outside the adapter is trainable")
    assert trn == expected, (
        f"LoRA attached {trn:,} trainable params, expected {expected:,} "
        f"(r={a.rank}, {len(TARGETS)} module types): {diagnosis}")

    # ---------------------------------------------------------------- 3. data
    ds = load_dataset("json", data_files={
        "train": split_path("train"), "validation": split_path("val")})
    # SFTTrainer wants ONLY the conversational column; meta would be templated in
    ds = ds.select_columns(["messages"])
    val = ds["validation"].select(range(min(a.eval_n, len(ds["validation"]))))
    print(f"train {len(ds['train']):,} | val {len(val):,}")

    cfg = SFTConfig(
        output_dir=a.out,
        max_length=a.seq_len,
        packing=False,
        # Loss on the assistant turn only. Without this the model spends
        # capacity learning to reproduce the system prompt and the questions.
        assistant_only_loss=True,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        gradient_checkpointing=a.grad_ckpt,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=a.warmup,
        weight_decay=0.01,
        optim="paged_adamw_8bit" if a.qlora else "adamw_torch",
        bf16=bf16, fp16=not bf16,
        logging_steps=10,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=250, save_total_limit=2,
        seed=0, report_to="none",
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=val, processing_class=tok)

    # ------------------------------------------------------------- 4. train
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0

    peak = torch.cuda.max_memory_reserved() / 1e9
    steps = result.metrics.get("train_steps_per_second", 0)
    ev = trainer.evaluate()

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    adapter_bytes = sum(
        os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)
        if f.endswith((".safetensors", ".bin"))
    )

    metrics = {
        "mode": "qlora" if a.qlora else "lora",
        "model": a.model,
        "peak_vram_gb": round(peak, 2),
        "base_vram_gb": round(base_vram, 2),
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
        "sec_per_step": round(1 / steps, 3) if steps else None,
        "global_steps": result.global_step,
        "train_loss": round(result.metrics["train_loss"], 4),
        "eval_loss": round(ev.get("eval_loss", float("nan")), 4),
        "trainable_params": trn, "total_params": tot,
        "trainable_pct": round(100 * trn / tot, 3),
        "adapter_bytes": adapter_bytes,
        "config": {k: v for k, v in vars(a).items()},
    }
    with open(os.path.join(a.out, "run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{mode}")
    print(f"  peak VRAM     {peak:.2f} GB")
    print(f"  wall          {wall/60:.1f} min ({result.global_step} steps)")
    print(f"  sec/step      {metrics['sec_per_step']}")
    print(f"  train loss    {metrics['train_loss']}")
    print(f"  eval  loss    {metrics['eval_loss']}")
    print(f"  adapter       {adapter_bytes/1e6:.1f} MB")
    print(f"  -> {a.out}/run_metrics.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
