"""
Build a synthetic instruction dataset: free-text incident report -> strict JSON.

The task is deliberately chosen to be something a small base model is *bad* at
and fine-tuning is *good* at:

  - It needs a rigid output schema (exact keys, exact enum values, no prose).
  - It needs domain vocabulary the base model has never seen (`atlas-dispatch`,
    `TLM-330`, the SEV ladder).
  - It needs no world knowledge or reasoning the base model lacks.

That combination is the honest sweet spot for LoRA. If the task needed *new
knowledge*, RAG would be the right tool. If it needed *reasoning*, a bigger model
would be. Fine-tuning teaches **behaviour and format**, and this task is pure
behaviour and format.

    python make_dataset.py --train 800 --eval 120

Output (JSONL, one chat-format record per line):
    data/train.jsonl
    data/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --------------------------------------------------------------- domain

COMPONENTS = {
    # Each symptom carries its OWN remediation. This is the whole point of the
    # v2 dataset: in v1 the action was keyed on the *component*, so four of five
    # components had exactly one action and `action` collapsed into a lookup --
    # a predictor that copied each component's majority action scored 86.7%,
    # against the fine-tuned model's 95.8%. The field was nearly free and the
    # per-field metric did not say so. Keying the action on the symptom forces
    # the model to read the body of the report, not just spot the service name.
    "atlas-dispatch": {
        "codes": ["DSP-500", "DSP-512", None],
        "symptoms": [
            ("the auction loop period climbed to {ms} ms",
             "raise the auction loop budget and profile the bid RPC path"),
            ("bid RPC pool is exhausted and tasks are queueing",
             "increase the bid RPC pool size and drain the queue backlog"),
            ("task assignment p99 is sitting at {sec} seconds",
             "scale atlas-dispatch replicas from 3 to 5"),
            ("{n} pallets have gone unassigned for {min} minutes",
             "reassign the orphaned pallets and restart the assignment worker"),
        ],
    },
    "atlas-telemetry": {
        "codes": ["TLM-101", "TLM-204", "TLM-330", "TLM-402", None],
        "symptoms": [
            ("clock skew alarms firing across {n} robots",
             "restart ntp-relay in the cell namespace"),
            ("shed mode has been active for {min} minutes",
             "clear shed mode once the ingest backlog has drained"),
            ("hypertable chunk writes failing on tsdb-0",
             "page the DBA and check disk on tsdb-0"),
            ("duplicate sequence numbers flooding the ingest log",
             "no action required, dedup handles duplicate sequence numbers"),
            ("sample rate dropped to 5 Hz fleet-wide",
             "check the telemetry write buffer and NTP sync"),
        ],
    },
    "atlas-vision": {
        "codes": ["VIS-207", "VIS-311", None],
        "symptoms": [
            ("gantry {n} GPU has fallen off the bus",
             "power cycle the affected gantry"),
            ("barcode read rate has dropped to {pct}%",
             "recalibrate the barcode scanners and check lane lighting"),
            ("{n} pallets routed to manual inspection in the last hour",
             "clear the manual inspection backlog and review the routing threshold"),
            ("the damage classifier is returning constant scores",
             "roll back the damage classifier to the previous model version"),
        ],
    },
    "atlas-console": {
        "codes": ["CON-401", None],
        "symptoms": [
            ("operators cannot log in, session service returning 401s",
             "restart the console session service"),
            ("the fleet map is {min} minutes stale",
             "flush the fleet map cache and verify the websocket feed"),
            ("the console is showing a blank task queue",
             "reload the task queue view and check console API health"),
        ],
    },
    "atlas-sim": {
        "codes": ["SIM-110", None],
        "symptoms": [
            ("the regression gate is failing on peak-friday.yaml",
             "triage the regression gate failure against the last passing commit"),
            ("replay output is no longer byte-identical across runs",
             "re-run the scenario with a pinned seed and compare output hashes"),
            ("scenario battery-cliff.yaml times out after {min} minutes",
             "raise the scenario timeout and profile the battery model"),
        ],
    },
}

CELLS = ["Rotterdam", "Hamburg", "Memphis", "Lyon", "Katowice", "Bristol"]

# (severity, is the fleet stopped, does it page)
SEV_PATTERNS = [
    ("SEV1", ["robots are halted", "the safety system is offline",
              "the entire cell has stopped", "emergency stop triggered fleet-wide"]),
    ("SEV2", ["throughput is down {pct}%", "we are running at {pct}% of normal rate",
              "picking has slowed noticeably", "dispatch p99 is above 2 seconds"]),
    ("SEV3", ["throughput is unaffected", "no customer impact so far",
              "operators have not noticed", "metrics look normal otherwise"]),
]

OPENERS = [
    "Hey, {cell} here.", "Reporting from {cell}:", "{cell} cell issue.",
    "Quick one from {cell} --", "Ops just flagged this in {cell}.",
    "Got a problem at {cell}.", "",
]

CLOSERS = [
    "Can someone take a look?", "What do we do?", "Escalating now.",
    "Filing this for the record.", "Not sure who owns this.", "",
]

SYSTEM = (
    "You are the Atlas incident triage service. Convert the operator report into "
    "a JSON object with exactly these keys: component, severity, error_code, "
    "page_oncall, action. severity is one of SEV1, SEV2, SEV3. error_code is the "
    "code string or null. page_oncall is true only for SEV1 and SEV2. "
    "Respond with JSON only."
)


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        ms=rng.choice([320, 410, 550, 700, 900]),
        sec=rng.choice([2, 3, 4, 6]),
        n=rng.randint(2, 40),
        min=rng.randint(5, 90),
        pct=rng.choice([35, 42, 55, 61, 74, 84]),
    )


def make_example(rng: random.Random) -> dict:
    component = rng.choice(list(COMPONENTS))
    spec = COMPONENTS[component]
    code = rng.choice(spec["codes"])
    severity, sev_phrases = rng.choice(SEV_PATTERNS)

    cell = rng.choice(CELLS)
    parts = []
    opener = rng.choice(OPENERS).format(cell=cell)
    if opener:
        parts.append(opener)

    symptom_tpl, action = rng.choice(spec["symptoms"])
    symptom = _fill(symptom_tpl, rng)
    if code and rng.random() < 0.75:
        parts.append(f"We're seeing {code} on {component} -- {symptom}.")
    else:
        # The report does not state a code -- so the LABEL must not claim one.
        #
        # This line used to keep `code` while writing a report that never
        # mentioned it, which made ~25% of coded rows unanswerable: the correct
        # error_code was not derivable from the input at all. The model learned
        # the only sane behaviour (emit null) and was scored wrong for it, which
        # put a hard ceiling of 84.2% on both error_code and exact match -- and
        # the model sat exactly on that ceiling, looking like a 16% failure rate
        # that no amount of training could ever fix.
        #
        # Now "no code in the report" means "error_code is null", which is a rule
        # the model can actually learn, and the eval measures the model instead of
        # measuring a labelling mistake.
        code = None
        parts.append(f"Something is wrong with {component}: {symptom}.")

    parts.append(_fill(rng.choice(sev_phrases), rng).capitalize() + ".")

    closer = rng.choice(CLOSERS)
    if closer:
        parts.append(closer)

    report = " ".join(parts)
    target = {
        "component": component,
        "severity": severity,
        "error_code": code,
        "page_oncall": severity in ("SEV1", "SEV2"),
        "action": action,
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": report},
            # separators=(",", ": ") keeps a single canonical string form; the
            # model is learning a format, so the format must be consistent.
            {"role": "assistant", "content": json.dumps(target, separators=(",", ": "))},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--eval", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Separate RNG streams so changing --train cannot shift the eval set.
    # An eval set that moves when you change a training flag is worthless.
    for name, n, seed in [("train", args.train, args.seed), ("eval", args.eval, args.seed + 10_000)]:
        rng = random.Random(seed)
        seen: set[str] = set()
        rows = []
        while len(rows) < n:
            ex = make_example(rng)
            key = ex["messages"][1]["content"]
            if key in seen:          # exact-duplicate prompts add nothing
                continue
            seen.add(key)
            rows.append(ex)
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}  {len(rows)} examples")

    sample = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    print("\nExample:")
    print("  USER:      ", sample["messages"][1]["content"])
    print("  ASSISTANT: ", sample["messages"][2]["content"])


if __name__ == "__main__":
    main()
