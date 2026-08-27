"""
Side-by-side: the same model, with and without the trained adapter.

    python pipeline/07_demo.py --adapter runs/qlora-nf4 -n 4

Loads the base ONCE with the adapter attached, then uses peft's
`disable_adapter()` context manager for the "before" answers. Same weights, same
sampling, same prompt -- so any difference is the adapter and nothing else.
Loading two separate models would leave room for a config drift to masquerade as
a training effect.

Questions come from test.jsonl, i.e. books held out of training entirely.
"""
import argparse, json, os, sys, textwrap

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import SYSTEM_PROMPT, split_path


def wrap(s, w=88, indent="    "):
    return "\n".join(textwrap.fill(p, w, initial_indent=indent,
                                   subsequent_indent=indent)
                     for p in s.split("\n") if p.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/qlora-nf4")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=220)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.adapter, "run_metrics.json")))
    base_id = cfg["model"]
    print(f"base    {base_id}")
    print(f"adapter {a.adapter}  ({cfg['mode']}, {cfg['global_steps']} steps, "
          f"eval loss {cfg['eval_loss']})\n")

    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(split_path("test"), encoding="utf-8")]
    seen, picked = set(), []
    for r in rows:                      # one question per held-out book
        src = r["meta"]["source"]
        if src in seen:
            continue
        seen.add(src)
        picked.append(r)
        if len(picked) >= a.n:
            break

    def gen(question):
        # transformers 5.x: apply_chat_template returns a BatchEncoding dict,
        # not a bare tensor. Passing it positionally to generate() fails with a
        # bare AttributeError on `.shape`.
        enc = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": question}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        plen = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][plen:], skip_special_tokens=True).strip()

    for i, r in enumerate(picked, 1):
        q = r["messages"][1]["content"]
        print("=" * 92)
        print(f"Q{i}  [{r['meta']['source']}]  {q}")
        print("=" * 92)

        with model.disable_adapter():          # adapter OFF -> plain base model
            before = gen(q)
        after = gen(q)                          # adapter ON

        print("\n  --- BASE (adapter disabled) ---")
        print(wrap(before))
        print("\n  --- TUNED (adapter enabled) ---")
        print(wrap(after))
        print("\n  --- REFERENCE (what the corpus actually says) ---")
        print(wrap(r["messages"][2]["content"]))
        print()


if __name__ == "__main__":
    main()
