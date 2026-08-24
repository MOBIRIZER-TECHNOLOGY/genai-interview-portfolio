"""
Base model vs LoRA-tuned model, on the held-out eval set.

    python evaluate.py                       # both, 120 examples
    python evaluate.py --n 40 --show 5       # quicker, print some outputs

Metrics, in the order they matter:

  json_valid     - did the output parse as JSON at all? A model that emits
                   ```json fences, or prose before the object, is unusable in a
                   pipeline no matter how correct the content is. This is the
                   metric fine-tuning moves the most and it is the one that
                   actually unblocks downstream code.
  schema_valid   - exactly the 5 required keys, no extras, no missing
  field accuracy - per-field exact match against the label
  exact_match    - every field correct simultaneously (the strict number)

Both models are run with `do_sample=False` (greedy). Sampling would make the
comparison noise, and the task has one correct answer.

`--show` prints raw outputs, which is worth doing at least once: base-model
failures here are *interesting*, not random, and being able to describe them is
the point of the exercise.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).parent
REQUIRED_KEYS = {"component", "severity", "error_code", "page_oncall", "action"}
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> dict | None:
    """Parse the model output as leniently as is still honest.

    We strip code fences and take the first {...} span, because that is what a
    tolerant production parser would do. We do NOT repair broken JSON -- that
    would hide exactly the failure mode we are measuring.
    """
    text = text.strip()
    m = FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


@torch.no_grad()
def run_model(model, tok, rows: list[dict], max_new_tokens: int, batch_size: int) -> tuple[list[str], float]:
    outs: list[str] = []
    model.eval()
    tok.padding_side = "left"      # required for correct batched generation
    t0 = time.perf_counter()

    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        prompts = [
            tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1] :]
            outs.append(tok.decode(new, skip_special_tokens=True))

    return outs, time.perf_counter() - t0


def score(outputs: list[str], rows: list[dict]) -> dict:
    n = len(rows)
    json_ok = schema_ok = exact = 0
    fields = Counter()
    failures: list[dict] = []

    for out, row in zip(outputs, rows):
        gold = json.loads(row["messages"][-1]["content"])
        pred = extract_json(out)

        if pred is None:
            failures.append({"reason": "not json", "output": out[:180], "gold": gold})
            continue
        json_ok += 1

        if set(pred.keys()) == REQUIRED_KEYS:
            schema_ok += 1
        else:
            failures.append(
                {
                    "reason": f"schema: extra={sorted(set(pred) - REQUIRED_KEYS)} "
                              f"missing={sorted(REQUIRED_KEYS - set(pred))}",
                    "output": out[:180], "gold": gold,
                }
            )

        all_ok = True
        for k in REQUIRED_KEYS:
            if k in pred and pred[k] == gold[k]:
                fields[k] += 1
            else:
                all_ok = False
        if all_ok and set(pred.keys()) == REQUIRED_KEYS:
            exact += 1
        elif len(failures) == 0 or failures[-1]["output"] != out[:180]:
            failures.append({"reason": "field mismatch", "output": out[:180], "gold": gold})

    return {
        "n": n,
        "json_valid": json_ok / n,
        "schema_valid": schema_ok / n,
        "exact_match": exact / n,
        "fields": {k: fields[k] / n for k in sorted(REQUIRED_KEYS)},
        "failures": failures[:8],
    }


def report(name: str, s: dict, seconds: float) -> None:
    print(f"\n### {name}")
    print(f"  json_valid    {s['json_valid']:7.1%}")
    print(f"  schema_valid  {s['schema_valid']:7.1%}")
    print(f"  exact_match   {s['exact_match']:7.1%}")
    print("  per-field accuracy:")
    for k, v in s["fields"].items():
        print(f"     {k:<12} {v:7.1%}")
    print(f"  {seconds:.1f}s for {s['n']} examples ({s['n']/seconds:.1f}/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lora", default=str(HERE / "lora-out"))
    ap.add_argument("--data", default=str(HERE / "data" / "eval.jsonl"))
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=110)
    ap.add_argument("--show", type=int, default=0, help="print N side-by-side outputs")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--out", default=str(HERE / "eval_results.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        rows = rows[: args.n]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("=" * 74)
    print(f"  Base vs LoRA  |  {len(rows)} held-out examples  |  greedy decoding")
    print("=" * 74)

    results: dict = {"model": args.model, "n": len(rows)}
    base_outs: list[str] = []

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")

    if not args.skip_base:
        base_outs, t = run_model(model, tok, rows, args.max_new_tokens, args.batch_size)
        base = score(base_outs, rows)
        report("BASE (no fine-tuning)", base, t)
        results["base"] = {k: v for k, v in base.items() if k != "failures"}
        results["base_failures"] = base["failures"]

    from peft import PeftModel

    # Attach the adapter to the same base weights already on the GPU. This is
    # the whole LoRA value proposition: one frozen base, adapters swapped on top.
    model = PeftModel.from_pretrained(model, args.lora)
    lora_outs, t = run_model(model, tok, rows, args.max_new_tokens, args.batch_size)
    tuned = score(lora_outs, rows)
    report("LoRA fine-tuned", tuned, t)
    results["lora"] = {k: v for k, v in tuned.items() if k != "failures"}
    results["lora_failures"] = tuned["failures"]

    if not args.skip_base:
        print("\n### Delta")
        for k in ("json_valid", "schema_valid", "exact_match"):
            d = tuned[k] - results["base"][k]
            print(f"  {k:<14} {results['base'][k]:7.1%}  ->  {tuned[k]:7.1%}   ({d:+.1%})")

        if results["base"]["json_valid"] < 1.0:
            print("\n  Base-model failure examples:")
            for f in results["base_failures"][:3]:
                print(f"    [{f['reason']}] {f['output'][:150]!r}")

    for i in range(min(args.show, len(rows))):
        print("\n" + "-" * 70)
        print("PROMPT: ", rows[i]["messages"][1]["content"])
        if base_outs:
            print("BASE:   ", base_outs[i][:220].replace("\n", " "))
        print("LORA:   ", lora_outs[i][:220].replace("\n", " "))
        print("GOLD:   ", rows[i]["messages"][-1]["content"])

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
