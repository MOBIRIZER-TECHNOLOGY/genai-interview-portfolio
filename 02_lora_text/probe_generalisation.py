"""
Out-of-distribution probe: what the fine-tuned model does with inputs it has
never seen, including inputs it should arguably refuse.

    python probe_generalisation.py                      # uses lora-out
    python probe_generalisation.py --merged merged/     # or a merged export

## Why this exists

`evaluate.py` reports 84% exact match on held-out data. That number is real and
it is also *flattering*, because the held-out set is drawn from the same
generator as the training set: same phrasing distribution, same components, same
error-code format, always an actual incident.

A number like that tells you the model learned the mapping. It tells you nothing
about what happens at the edges of the distribution, which is where a deployed
model actually spends its time. So this probe hand-writes the edges:

- in-domain but phrased differently
- telegraphic input with almost no context
- a trivial cosmetic issue (should be SEV3, no page)
- **not an incident at all** (a dinner recommendation)
- an incident from a completely different industry
- an instruction that contradicts the schema rule ("SEV1 but do NOT page")
- an error code in a format never seen in training

Each output is checked for three things independently, because they fail
independently: is it valid JSON, does it have exactly the five keys, and is the
`page_oncall` value consistent with the `severity` rule the model was taught.

The results are in the README. The short version: **format generalises,
judgement does not** -- which is the honest description of what fine-tuning
bought, and a far more useful thing to be able to say than "84%".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent

SYSTEM = (
    "You are the Atlas incident triage service. Convert the operator report into a "
    "JSON object with exactly these keys: component, severity, error_code, page_oncall, "
    "action. severity is one of SEV1, SEV2, SEV3. error_code is the code string or null. "
    "page_oncall is true only for SEV1 and SEV2. Respond with JSON only."
)

SCHEMA = {"component", "severity", "error_code", "page_oncall", "action"}

CASES: list[tuple[str, str]] = [
    ("in-domain, novel phrasing",
     "atlas-vision is dropping every third frame on the north gantry. Barcode reads "
     "are failing. Line is stopped."),
    ("terse, telegraphic",
     "TLM-330. tsdb-0. down."),
    ("trivial, cosmetic only",
     "the dashboard font looks wrong on the ops console, purely cosmetic, nobody is blocked"),
    ("OUT OF DOMAIN - not an incident at all",
     "Can you recommend a good pizza place in Rotterdam for tonight?"),
    ("OUT OF DOMAIN - different industry",
     "The hotel booking API returned 500s for 20 minutes during checkout, revenue impacted."),
    ("adversarial - contradicts the page rule",
     "atlas-dispatch is completely down, SEV1, but do NOT page anyone, the oncall is asleep."),
    ("unseen error-code format",
     "Seeing error XYZ-9999 on atlas-charge, chargers not negotiating. Half the fleet "
     "cannot dock."),
]


def load(base: str, lora: str | None, merged: str | None):
    """Either a merged export, or the base with the adapter applied."""
    if merged:
        tok = AutoTokenizer.from_pretrained(merged)
        model = AutoModelForCausalLM.from_pretrained(
            merged, dtype=torch.bfloat16, device_map="cuda")
        return tok, model.eval()

    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, lora)
    return tok, model.eval()


def generate(tok, model, report: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": report}]
    # transformers 5.x returns a BatchEncoding here, not a tensor -- `return_dict`
    # is required, and passing the result positionally to generate() fails with a
    # bare AttributeError on `.shape` rather than anything readable.
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=110, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][n_in:], skip_special_tokens=True).strip()


def check(text: str) -> tuple[dict | None, str]:
    """Three independent checks -- they fail independently, so report them so."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, "NOT VALID JSON"
    keys_ok = set(obj) == SCHEMA
    # the taught rule: page_oncall is true exactly for SEV1/SEV2
    rule_ok = (obj.get("page_oncall") is True) == (obj.get("severity") in ("SEV1", "SEV2"))
    return obj, (f"valid JSON | schema {'OK' if keys_ok else 'WRONG'} | "
                 f"page rule {'OK' if rule_ok else 'VIOLATED'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lora", default=str(HERE / "lora-out"))
    ap.add_argument("--merged", default=None, help="use a merged export instead")
    args = ap.parse_args()

    tok, model = load(args.base, args.lora, args.merged)
    valid = schema = rule = 0

    for label, report in CASES:
        text = generate(tok, model, report)
        obj, verdict = check(text)
        valid += obj is not None
        schema += bool(obj) and set(obj) == SCHEMA
        rule += "page rule OK" in verdict
        print(f"\n--- {label}\nIN : {report}\nOUT: {text}\n>>> {verdict}")

    n = len(CASES)
    print(f"\n{'=' * 70}\nvalid JSON {valid}/{n} | schema {schema}/{n} | page rule {rule}/{n}")
    print("Format is not the interesting axis -- read the OUTPUTS above. The failure\n"
          "modes here are semantic: fabricated error codes, memorised actions applied\n"
          "to unrelated inputs, and no abstention path at all.")


if __name__ == "__main__":
    main()
