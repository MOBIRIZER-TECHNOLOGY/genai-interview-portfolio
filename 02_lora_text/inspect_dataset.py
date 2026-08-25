"""
What is actually in the training data — inventory, distributions, and the
trivial baseline each field should be judged against.

    python inspect_dataset.py

## Why this exists

`evaluate.py` reports per-field accuracy. A field scoring 95.8% sounds like a
hard problem solved well. But a per-field number means nothing until you know
what a *stupid* predictor scores on the same field, and that is a property of
the dataset, not the model.

Running this on my own data was uncomfortable and useful in equal measure:

- `severity`: majority-class baseline **39.2%**, model **100%**. A real win.
- `action`:   "copy this component's most common action" scores **86.7%**,
  model **95.8%**. Nine points of actual learning, on a field whose headline
  number implies far more — because four of the five components have exactly
  **one** action in the entire training set.

That coupling also explains an out-of-distribution failure in
`probe_generalisation.py`: a cosmetic font complaint got "restart ntp-relay in
the cell namespace". The model was never learning remediation. It was learning
component -> action, and the component guess dragged the action with it.

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

    print("\n=== trivial baselines on EVAL (what a lookup table scores) ===")
    maj_action = {c: cnt.most_common(1)[0][0] for c, cnt in by_c.items()}
    hit = sum(1 for r in ev if maj_action.get(r["component"]) == r["action"])
    print(f"  action    'copy the component's majority action' : {100*hit/len(ev):5.1f}%"
          f"   (fine-tuned model: 95.8%)")
    for f, model in (("severity", 100.0), ("component", 100.0), ("page_oncall", 100.0)):
        maj = collections.Counter(str(r[f]) for r in train).most_common(1)[0][0]
        acc = 100 * sum(1 for r in ev if str(r[f]) == maj) / len(ev)
        print(f"  {f:<9} always predict {maj!r:<18}: {acc:5.1f}%   (fine-tuned model: {model:.1f}%)")

    print("\nRead the gaps, not the accuracies: severity is a real win, action is\n"
          "mostly a lookup the model inherited from getting `component` right.")


if __name__ == "__main__":
    main()
