"""
Stage 5a: put the two runs side by side and grade them against the predictions
that were registered BEFORE either was run.

    python pipeline/06_compare.py runs/lora-bf16 runs/qlora-nf4

The predictions live in IMPLEMENTATION_PLAN.md section 5. They are restated here
so the check is mechanical rather than a matter of recollection. A FALSIFIED
prediction is a good outcome -- it means the measurement taught you something the
documentation did not.
"""
import argparse, json, os, sys

# Registered in IMPLEMENTATION_PLAN.md before any GPU run.
#
# Each entry carries a two-sided BAND (lo, hi), not a ceiling. An earlier version
# used one-sided thresholds, which silently passed a prediction that was wrong in
# the favourable direction: "qlora is 30-40% slower" was checked as "<= 60%", so
# a measured 14.5% reported HOLDS even though the stated band was missed by half.
# A prediction that is wrong in a direction you like is still wrong.
PREDICTIONS = [
    ("VRAM saving", "qlora uses 50-60% less peak VRAM than lora",
     lambda l, q: 100 * (l["peak_vram_gb"] - q["peak_vram_gb"]) / l["peak_vram_gb"],
     (50, 60), "{:.1f}% saved"),
    ("Speed cost", "qlora is 25-40% slower per step",
     lambda l, q: 100 * (q["sec_per_step"] - l["sec_per_step"]) / l["sec_per_step"],
     (25, 40), "{:+.1f}% step time"),
    ("Quality parity", "eval loss differs by < 0.15",
     lambda l, q: abs(q["eval_loss"] - l["eval_loss"]),
     (0, 0.15), "delta {:.4f}"),
    ("Adapter identical", "adapter size is the same (control variable)",
     lambda l, q: abs(q["adapter_bytes"] - l["adapter_bytes"]) / max(l["adapter_bytes"], 1),
     (0, 0.02), "{:.2%} size diff"),
    ("Trainable identical", "same trainable parameter count (control variable)",
     lambda l, q: abs(q["trainable_params"] - l["trainable_params"]),
     (0, 0), "{:.0f} param diff"),
]

ROWS = [
    ("peak VRAM (GB)", "peak_vram_gb", "{:.2f}"),
    ("base weights (GB)", "base_vram_gb", "{:.2f}"),
    ("sec / step", "sec_per_step", "{:.3f}"),
    ("wall (min)", "train_runtime_s", lambda v: f"{v/60:.1f}"),
    ("steps", "global_steps", "{:.0f}"),
    ("train loss", "train_loss", "{:.4f}"),
    ("eval loss", "eval_loss", "{:.4f}"),
    ("trainable params", "trainable_params", "{:,.0f}"),
    ("trainable %", "trainable_pct", "{:.3f}"),
    ("adapter (MB)", "adapter_bytes", lambda v: f"{v/1e6:.1f}"),
]


def load(path):
    f = path if path.endswith(".json") else os.path.join(path, "run_metrics.json")
    if not os.path.exists(f):
        sys.exit(f"missing {f} -- did that run finish?")
    return json.load(open(f))


def fmt(spec, v):
    if v is None:
        return "n/a"
    return spec(v) if callable(spec) else spec.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lora", help="dir of the bf16 LoRA run")
    ap.add_argument("qlora", help="dir of the 4-bit QLoRA run")
    a = ap.parse_args()

    l, q = load(a.lora), load(a.qlora)
    if l["mode"] == q["mode"]:
        print(f"!! both runs are mode={l['mode']} -- this is not an A/B\n")
    if l["model"] != q["model"]:
        print(f"!! different base models ({l['model']} vs {q['model']}).\n"
              "   This comparison cannot isolate quantisation.\n")

    w = 22
    print("=" * 62)
    print(f"{'':<{w}}{'LoRA bf16':>18}{'QLoRA nf4':>18}")
    print("=" * 62)
    for label, key, spec in ROWS:
        print(f"{label:<{w}}{fmt(spec, l.get(key)):>18}{fmt(spec, q.get(key)):>18}")

    print("\n" + "=" * 62)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 62)
    held = 0
    for name, claim, calc, band, disp in PREDICTIONS:
        try:
            v = calc(l, q)
        except (KeyError, TypeError, ZeroDivisionError):
            print(f"  SKIP  {name}: metric missing")
            continue
        lo, hi = band
        if v < lo:
            verdict, why = "MISSED", f"below the predicted {lo:g}-{hi:g} band"
        elif v > hi:
            verdict, why = "MISSED", f"above the predicted {lo:g}-{hi:g} band"
        else:
            verdict, why, held = "HOLDS", "", held + 1
        print(f"  {verdict:<9} {name:<20} {disp.format(v):>18}"
              f"   {'(' + why + ')' if why else ''}")
        print(f"            predicted: {claim}")

    print("\n" + "-" * 62)
    print(f"{held}/{len(PREDICTIONS)} predictions held.")
    if held < len(PREDICTIONS):
        print("A MISSED prediction is a RESULT, not a failure. Record what\n"
              "actually happened and why -- including when reality was BETTER\n"
              "than predicted. Do not re-run until the numbers agree.")
    print("-" * 62)


if __name__ == "__main__":
    main()
