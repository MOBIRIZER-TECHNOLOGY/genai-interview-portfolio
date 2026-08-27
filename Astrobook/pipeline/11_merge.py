"""
Merge a trained adapter into the base -> one standalone model.

    python pipeline/11_merge.py --adapter runs/qlora-2ep --out models/astro-4b

Produces a normal HuggingFace model directory that loads with a plain
AutoModelForCausalLM.from_pretrained() -- no peft, no adapter file, no
knowledge that LoRA was ever involved.

WHY THE BASE IS LOADED IN bf16, NOT 4-BIT
-----------------------------------------
You cannot merge into NF4 weights. A 4-bit weight is a quantised integer plus a
shared scale; adding a bf16 delta to it is not a defined operation. So even
though the adapter was TRAINED against a 4-bit base, merging requires the full
precision base, and the output is bf16.

This surprises people: QLoRA lets you train in 4 bits, but the merged artifact
comes out at full size. Re-quantise afterwards if you want it small -- and note
that re-quantisation happens AFTER training, so the adapter never adapted to it.

VERIFICATION
------------
Merging is W_new = W + (B @ A) * (alpha / r). This script checks that identity
numerically on real layers rather than trusting merge_and_unload(), then runs a
generation smoke test. `test_merge_matches_unmerged` proves the same property on
a toy layer; this proves it on the actual 66M-parameter adapter.
"""
import argparse, json, os, shutil, sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import SYSTEM_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/qlora-2ep")
    ap.add_argument("--out", default="models/astro-4b")
    ap.add_argument("--device", default="cpu",
                    help="cpu is safest; cuda is faster if VRAM allows")
    ap.add_argument("--check-layers", type=int, default=3)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.adapter, "run_metrics.json")))
    acfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
    base_id = cfg["model"]
    scaling = acfg["lora_alpha"] / acfg["r"]

    print(f"base      {base_id}")
    print(f"adapter   {a.adapter}  (r={acfg['r']} alpha={acfg['lora_alpha']} "
          f"-> scaling {scaling:g})")
    print(f"out       {a.out}")
    print(f"device    {a.device}   (bf16 -- 4-bit cannot be merged into)\n")

    tok = AutoTokenizer.from_pretrained(a.adapter)
    print("loading base in bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map={"": a.device})

    # Snapshot a few base weights so we can verify the delta afterwards.
    print("attaching adapter ...")
    peft_model = PeftModel.from_pretrained(model, a.adapter)

    targets = []
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
            targets.append((name, mod))
        if len(targets) >= a.check_layers:
            break

    checks = []
    for name, mod in targets:
        A = mod.lora_A["default"].weight.detach().to(torch.float32)
        B = mod.lora_B["default"].weight.detach().to(torch.float32)
        W = mod.base_layer.weight.detach().to(torch.float32).clone()
        checks.append((name, W, (B @ A) * scaling))

    print("merging ...")
    merged = peft_model.merge_and_unload()

    # ---------------------------------------------------------- verification
    print("\nverifying  W_merged == W_base + (B@A)*(alpha/r):")
    name_to_mod = dict(merged.named_modules())
    ok = True
    for name, W_before, delta in checks:
        clean = name.replace("base_model.model.", "")
        mod = name_to_mod.get(clean)
        if mod is None or not hasattr(mod, "weight"):
            print(f"  ?  {clean}: not found post-merge")
            continue
        W_after = mod.weight.detach().to(torch.float32)
        expected = W_before + delta
        err = (W_after - expected).abs().max().item()
        scale = W_after.abs().max().item()
        moved = (W_after - W_before).abs().max().item()
        good = err < 1e-2 and moved > 0
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {clean.split('.')[-3:] and '.'.join(clean.split('.')[-3:]):<34}"
              f" max|err|={err:.2e}  weights moved by {moved:.4f}  (|W|max {scale:.2f})")
    if not ok:
        sys.exit("\nMERGE VERIFICATION FAILED -- not saving.")
    print("  (err is bf16 rounding; 'moved' > 0 proves the adapter was applied)")

    # ------------------------------------------------------------------ save
    os.makedirs(a.out, exist_ok=True)
    print(f"\nsaving to {a.out} ...")
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    for extra in ("chat_template.jinja",):
        src = os.path.join(a.adapter, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, extra))

    total = sum(os.path.getsize(os.path.join(a.out, f))
                for f in os.listdir(a.out)
                if os.path.isfile(os.path.join(a.out, f)))
    json.dump({"merged_from": a.adapter, "base": base_id,
               "lora_r": acfg["r"], "lora_alpha": acfg["lora_alpha"],
               "steps": cfg["global_steps"], "eval_loss": cfg["eval_loss"],
               "bytes": total},
              open(os.path.join(a.out, "merge_info.json"), "w"), indent=2)
    print(f"  {total/1e9:.2f} GB written")

    # ------------------------------------------------------------ smoke test
    print("\nsmoke test -- loading the merged model as a plain HF model:")
    del merged, peft_model
    m2 = AutoModelForCausalLM.from_pretrained(
        a.out, dtype=torch.bfloat16, device_map={"": a.device})
    m2.eval()
    q = "What does Brihat Parashara Hora Sastra associate with the 3rd house?"
    enc = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(a.device) for k, v in enc.items()}
    with torch.no_grad():
        out = m2.generate(**enc, max_new_tokens=120, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    ans = tok.decode(out[0][enc["input_ids"].shape[-1]:],
                     skip_special_tokens=True).strip()
    print(f"\n  Q: {q}")
    print(f"  A: {' '.join(ans.split())[:420]}")
    print(f"\nDONE -> {a.out}")


if __name__ == "__main__":
    main()
