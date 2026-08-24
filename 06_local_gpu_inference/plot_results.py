"""
Turn benchmark.py's results.json into a chart you can put in a README or a deck.

    python plot_results.py
    python plot_results.py --out my_chart.png

Four panels:
  1. memory by variant        - weights vs peak, side by side. The gap between
                                them is activations + KV cache, and it is the
                                part quantisation does NOT shrink.
  2. decode speed by variant  - the tokens/sec you actually get
  3. quality cost             - perplexity delta vs the bf16 baseline. A memory
                                chart with no quality chart next to it is an
                                advert, not a measurement.
  4. batching sweep           - total throughput vs per-sequence speed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent

# Colour-blind-safe, distinguishable in greyscale by ordering.
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#dfe3e8"
BLUE = "#3d7dd6"
TEAL = "#2a9d8f"
AMBER = "#e0913a"
RED = "#c1554a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json"))
    ap.add_argument("--out", default=str(HERE / "benchmark.png"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    ok = [v for v in data["variants"] if not v.get("error")]
    if not ok:
        raise SystemExit("no successful variants in results.json")

    names = [v["variant"] for v in ok]
    x = range(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{data['model']}  on  {data['gpu']} ({data['vram_gb']} GB)   |   "
        f"torch {data['torch']} / CUDA {data['cuda']}",
        fontsize=12, color=INK, y=0.98,
    )

    def style(ax, title: str, ylabel: str) -> None:
        ax.set_title(title, fontsize=11, color=INK, pad=10)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    # ---- 1. memory ---------------------------------------------------
    ax = axes[0][0]
    w = 0.38
    ax.bar([i - w / 2 for i in x], [v["weights_gb"] for v in ok], w, label="weights", color=BLUE)
    ax.bar([i + w / 2 for i in x], [v["peak_gb"] for v in ok], w, label="peak (weights+activations+KV)",
           color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    style(ax, "Memory", "GB")
    ax.legend(frameon=False, fontsize=8)
    for i, v in enumerate(ok):
        ax.text(i - w / 2, v["weights_gb"], f"{v['weights_gb']:.2f}", ha="center", va="bottom",
                fontsize=8, color=INK)

    # ---- 2. decode speed ---------------------------------------------
    ax = axes[0][1]
    bars = ax.bar(list(x), [v["decode_tok_s"] for v in ok], 0.55, color=TEAL)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    style(ax, "Decode speed (single stream)", "tokens / second")
    for b, v in zip(bars, ok):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v['decode_tok_s']:.0f}",
                ha="center", va="bottom", fontsize=8, color=INK)

    # ---- 3. quality cost ---------------------------------------------
    ax = axes[1][0]
    have_ppl = [v for v in ok if v.get("perplexity")]
    if have_ppl:
        ref = next((v for v in have_ppl if v["variant"] == "bf16"), have_ppl[0])
        deltas = [100 * (v["perplexity"] / ref["perplexity"] - 1) for v in have_ppl]
        colors = [RED if d > 1.0 else (AMBER if d > 0.2 else TEAL) for d in deltas]
        bars = ax.bar(range(len(have_ppl)), deltas, 0.55, color=colors)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(have_ppl)))
        ax.set_xticklabels([v["variant"] for v in have_ppl])
        style(ax, f"Quality cost: perplexity vs {ref['variant']}", "% worse (lower is better)")
        for b, d in zip(bars, deltas):
            ax.text(b.get_x() + b.get_width() / 2, d, f"{d:+.1f}%", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=8, color=INK)
    else:
        ax.text(0.5, 0.5, "run without --skip-quality\nfor the perplexity panel",
                ha="center", va="center", color=MUTED, transform=ax.transAxes)
        style(ax, "Quality cost", "")

    # ---- 4. batching --------------------------------------------------
    ax = axes[1][1]
    batching = data.get("batching") or []
    if batching:
        bs = [b["batch_size"] for b in batching]
        ax.plot(bs, [b["total_tok_s"] for b in batching], "o-", color=BLUE, linewidth=2,
                label="total throughput")
        ax.plot(bs, [b["per_seq_tok_s"] for b in batching], "s--", color=AMBER, linewidth=1.8,
                label="per-sequence speed")
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels([str(b) for b in bs])
        ax.set_xlabel("batch size", fontsize=9, color=MUTED)
        style(ax, f"Batching ({data.get('batching_variant','?')})", "tokens / second")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no batching data", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
        style(ax, "Batching", "")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140, facecolor="white")
    print(f"chart -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
