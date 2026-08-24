"""
Merge the LoRA adapter into the base weights and export a standalone model.

    python merge_and_export.py                       # -> merged/
    python merge_and_export.py --ollama              # also write an Ollama Modelfile
    python merge_and_export.py --test                # generate from the merged model

## Why merge at all?

At inference an unmerged LoRA computes `W·x + (B·A)·x` — two extra matmuls per
adapted layer. It's a small overhead, but it's per-token, forever. Merging folds
the adapter in: `W' = W + (alpha/r)·B·A`, giving you a plain model with zero
inference overhead.

## Why you might NOT merge

Merging is a one-way door for flexibility:

- **You lose hot-swapping.** The whole reason LoRA is cheap to serve is that one
  base model in VRAM can serve many adapters — per-customer, per-task — swapped
  per request. Merge and you're back to one full model per variant.
- **You lose composition.** Unmerged adapters can be blended at runtime.
- **Storage explodes.** 34 MB adapter vs ~1 GB merged model, per variant.

Rule of thumb: **merge when you ship one model** (export to GGUF/vLLM/ONNX,
hand it to another team). **Keep the adapter when you serve many variants.**

## The 4-bit caveat

You cannot cleanly merge a LoRA that was trained on 4-bit (QLoRA) base weights
into those 4-bit weights — quantising the sum is not the sum of the quantised.
The correct route is: load the base in **fp16/bf16**, apply the adapter, merge,
then quantise the merged result if you need to. This script does that.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lora", default=str(HERE / "lora-out"))
    ap.add_argument("--out", default=str(HERE / "merged"))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--ollama", action="store_true", help="write a Modelfile + instructions")
    ap.add_argument("--test", action="store_true", help="run one generation after merging")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype = getattr(torch, args.dtype)
    lora_dir = Path(args.lora)
    out = Path(args.out)

    info_path = lora_dir / "training_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        print(f"adapter: r={info['rank']} alpha={info['alpha']} "
              f"({info['adapter_mb']} MB, {info['trainable_pct']}% of params)")
        if info.get("load_4bit"):
            print("  note: trained with QLoRA. Merging into a bf16 base, which is correct --\n"
                  "        merging into the 4-bit weights would lose precision.")

    print(f"\nloading base in {args.dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(args.base)

    print("applying adapter ...")
    model = PeftModel.from_pretrained(model, str(lora_dir))

    print("merging W' = W + (alpha/r) * B @ A ...")
    model = model.merge_and_unload()

    if out.exists():
        shutil.rmtree(out)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024**2
    print(f"\nmerged model -> {out.resolve()}  ({size_mb:.0f} MB)")

    if args.test:
        from make_dataset import SYSTEM

        model = model.to("cuda").eval()
        report = ("Hey, Rotterdam here. We're seeing TLM-330 on atlas-telemetry -- "
                  "hypertable chunk writes failing on tsdb-0. Robots are halted.")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": report}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=110, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        print("\n--- merged-model test ---")
        print("IN :", report)
        print("OUT:", tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))

    if args.ollama:
        modelfile = out / "Modelfile"
        modelfile.write_text(
            "# Ollama Modelfile for the merged Atlas triage model.\n"
            "# Ollama needs GGUF, so convert the safetensors first:\n"
            "#\n"
            "#   git clone https://github.com/ggerganov/llama.cpp\n"
            "#   pip install -r llama.cpp/requirements.txt\n"
            f"#   python llama.cpp/convert_hf_to_gguf.py {out.name} --outfile atlas-triage-f16.gguf\n"
            "#   llama.cpp/llama-quantize atlas-triage-f16.gguf atlas-triage-q4.gguf Q4_K_M\n"
            "#\n"
            "# then:  ollama create atlas-triage -f Modelfile\n\n"
            'FROM ./atlas-triage-q4.gguf\n\n'
            "PARAMETER temperature 0\n"
            "PARAMETER num_ctx 2048\n"
            "PARAMETER stop \"<|im_end|>\"\n\n"
            'SYSTEM """You are the Atlas incident triage service. Convert the operator '
            "report into a JSON object with exactly these keys: component, severity, "
            "error_code, page_oncall, action. severity is one of SEV1, SEV2, SEV3. "
            "error_code is the code string or null. page_oncall is true only for SEV1 "
            'and SEV2. Respond with JSON only."""\n',
            encoding="utf-8",
        )
        print(f"Modelfile -> {modelfile.resolve()}  (see the comments for the GGUF step)")


if __name__ == "__main__":
    main()
