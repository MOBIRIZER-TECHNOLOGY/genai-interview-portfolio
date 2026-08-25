"""
What is actually in the training data — inventory, distributions, and the
trivial baseline each field should be judged against.

    python inspect_dataset.py

## Why this exists

`evaluate.py` reports per-field accuracy. A field scoring 95.8% sounds like a
hard problem solved well. But a per-field number means nothing until you know
what a *stupid* predictor scores on the same field, and that is a property of
the dataset, not the model.

Running this on the **v1** dataset was uncomfortable and useful in equal
measure: `action` scored 95.8% against an **86.7%** baseline — nine points of
actual learning — because four of the five components had exactly **one** action
in the entire training set. The field was nearly free and the per-field metric
did not say so.

That is why the dataset was rebuilt. Actions are now keyed on the **symptom**
rather than the component, which drops the lookup baseline to ~32% and forces
the model to read the body of the report rather than spot the service name.

Re-run this after any change to `make_dataset.py`: baselines are a property of
the data, and they move when the data does.

**Report the baseline next to the metric, or the metric is decoration.**
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("component", "severity", "error_code", "page_oncall", "action")


def load(name: str) -> list[dict]:
    """Return the gold JSON objects (the assistant turn) from a split."""
    rows = []
    for line in (HERE / "data" / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(json.loads(line)["messages"][-1]["content"]))
    return rows


def user_turns(name: str) -> list[str]:
    return [json.loads(line)["messages"][1]["content"]
            for line in (HERE / "data" / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    train, ev = load("train.jsonl"), load("eval.jsonl")
    print(f"train {len(train)} examples | eval {len(ev)} examples "
          f"(held out, same generator)\n")

    reports = user_turns("train.jsonl")
    print(f"distinct operator-report texts: {len(set(reports))}/{len(reports)}\n")

    for name, rows in (("TRAIN", train), ("EVAL", ev)):
        print(f"=== {name} ===")
        for f in ("component", "severity", "page_oncall"):
            c = collections.Counter(str(r[f]) for r in rows)
            print(f"  {f:<12} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
        codes = collections.Counter(
            (str(r["error_code"]).split("-")[0] if r["error_code"] else "null")
            for r in rows)
        print(f"  {'error_code':<12} " + "  ".join(f"{k}={v}" for k, v in sorted(codes.items())))
        print(f"  {'actions':<12} {len(set(r['action'] for r in rows))} distinct\n")

    print("=== every action the model can ever emit ===")
    for a, n in collections.Counter(r["action"] for r in train).most_common():
        print(f"  {n:>4}x  {a}")

    print("\n=== the coupling that makes `action` easy ===")
    by_c: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in train:
        by_c[r["component"]][r["action"]] += 1
    for c in sorted(by_c):
        top, n = by_c[c].most_common(1)[0]
        share = 100 * n / sum(by_c[c].values())
        print(f"  {c:<17} {len(by_c[c])} action(s); majority covers {share:5.1f}%")

    print("\n=== trivial baselines on EVAL vs the fine-tuned model ===")
    # model numbers are READ from the last eval run, never hardcoded -- a
    # hardcoded accuracy inside an analysis script is just another stale claim
    res = HERE / "eval_results.json"
    fields: dict[str, float] = {}
    if res.exists():
        fields = json.loads(res.read_text(encoding="utf-8"))["lora"]["fields"]
    else:
        print("  (run evaluate.py first to fill the model column)")

    def row(label: str, field: str, baseline: float) -> None:
        got = fields.get(field)
        model = f"{100 * got:5.1f}%" if got is not None else "    ?"
        gain = f"{100 * got - baseline:+6.1f}" if got is not None else "     ?"
        print(f"  {field:<12}{label:<40} {baseline:5.1f}%   model {model}   gain {gain}")

    maj_action = {c: cnt.most_common(1)[0][0] for c, cnt in by_c.items()}
    hit = 100 * sum(1 for r in ev if maj_action.get(r["component"]) == r["action"]) / len(ev)
    row("copy the component's majority action", "action", hit)
    for f in ("severity", "component", "page_oncall"):
        maj = collections.Counter(str(r[f]) for r in train).most_common(1)[0][0]
        acc = 100 * sum(1 for r in ev if str(r[f]) == maj) / len(ev)
        row(f"always predict {maj!r}", f, acc)

    print("\nRead the gain column, not the accuracy column. A field whose baseline\n"
          "is already high is a field your dataset made free.")


if __name__ == "__main__":
    main()
