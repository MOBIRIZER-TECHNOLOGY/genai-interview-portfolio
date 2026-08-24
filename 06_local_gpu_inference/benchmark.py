"""
Measure what quantisation and batching actually cost you on one consumer GPU.

    python benchmark.py                          # all variants, default model
    python benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct
    python benchmark.py --variants bf16 int4     # subset
    python benchmark.py --skip-quality           # faster, no perplexity pass

## What gets measured, and why each one matters

| Metric | What it tells you | Why it's here |
|---|---|---|
| **weights VRAM** | memory for the model itself | the number quantisation advertises |
| **peak VRAM** | weights + activations + KV cache | the number that actually decides if you fit |
| **load time** | cold start | matters for autoscaling and for serverless |
| **TTFT** | time to first token = prefill | what a user perceives as "did it respond" |
| **decode tok/s** | steady-state generation speed | what a user perceives as "is it fast" |
| **perplexity** | quality proxy on fixed text | quantisation is a *trade*; without this you only measured the upside |

## The idea everyone repeats -- and what this benchmark actually found

The textbook claim is that single-stream LLM decoding is **memory-bandwidth
bound**: generating one token reads every weight from VRAM and does only ~2 FLOPs
per weight, so a card with ~900 GB/s caps a 1 GB model near 900 tokens/s.

**Measure it before you repeat it.** Running this on a 5070 Ti:

    model   weights (bf16)   decode tok/s
    0.5B         0.94 GB         44.8
    1.5B         2.89 GB         38.7

Three times the bytes to read per token, essentially the same speed. If decoding
were bandwidth bound the 1.5B would be ~3x slower. It is not, because at this
scale with HuggingFace `generate()` the wall clock is dominated by **per-token
overhead** -- the Python loop, kernel launches, and ~24-28 sequential small
matmuls that never saturate the card. The GPU spends most of each token waiting
to be told what to do.

The bandwidth-bound regime is real, it just starts further out: bigger models
(7B+) and runtimes that remove the overhead (vLLM, TensorRT-LLM, llama.cpp with
CUDA graphs). Knowing *which regime you are in* is the actual skill, because it
decides whether quantising or batching or switching runtime is the win.

Two consequences this benchmark does demonstrate:

1. **Quantisation buys memory, not speed, at this scale.** int4 decodes at the
   same rate as bf16 despite reading a quarter of the bytes -- you were not
   bandwidth limited, so there was nothing to win. int8 (LLM.int8) is *5x
   slower*, because its outlier-handling path adds far more overhead than the
   bandwidth it saves.
2. **Batching is nearly free**, and this is the effect that dominates everything
   else. The weights are read once per forward pass regardless of batch size, so
   batch 32 delivers ~25-32x the total throughput while per-sequence speed barely
   moves. That gap is the entire economic argument for batched serving, and it is
   *larger* in the overhead-bound regime, not smaller.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

HERE = Path(__file__).parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = (
    "You are an incident commander. Explain, step by step, how you would "
    "diagnose a warehouse robot fleet whose task assignment latency has "
    "suddenly tripled. Be specific and thorough."
)

# Fixed text for the perplexity pass. Held constant across variants so the
# number is comparable; absolute perplexity on 500 words means nothing, the
# *delta* between variants is the whole point.
QUALITY_TEXT = (
    "The dispatch service assigns tasks to robots using a sealed-bid reverse "
    "auction that runs every one hundred and fifty milliseconds. Each idle robot "
    "submits a bid representing its estimated cost to complete the task, "
    "combining travel time, a battery penalty, and a congestion score derived "
    "from the aisle occupancy grid. The lowest bid wins, and ties are broken by "
    "robot serial number. A task that loses twelve consecutive auctions is "
    "escalated: its priority is multiplied and it is pinned to the next round "
    "regardless of bids. This starvation guard was added after an incident in "
    "which seventeen pallets sat unassigned for forty minutes during a peak "
    "shift, which went undetected because throughput metrics stayed nominal."
)


@dataclass
class Result:
    variant: str
    load_seconds: float
    weights_gb: float
    peak_gb: float
    ttft_ms: float
    decode_tok_s: float
    total_tok_s: float
    generated_tokens: int
    perplexity: float | None = None
    error: str | None = None


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load(model_name: str, variant: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw: dict = {}
    if variant == "fp32":
        kw = {"dtype": torch.float32}
    elif variant == "fp16":
        kw = {"dtype": torch.float16}
    elif variant == "bf16":
        kw = {"dtype": torch.bfloat16}
    elif variant == "int8":
        kw = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    elif variant == "int4":
        # NF4 + double quantisation, compute in bf16. This is the QLoRA recipe
        # used for inference rather than training.
        kw = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        }
    else:
        raise ValueError(f"unknown variant {variant!r}")

    free_gpu()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if variant in ("fp32", "fp16", "bf16"):
        model = model.to("cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    return model.eval(), tok, load_s, torch.cuda.memory_allocated() / 1024**3


@torch.no_grad()
def measure_generation(model, tok, max_new_tokens: int, warmup: int = 1) -> tuple[float, float, float, int]:
    """Return (ttft_ms, decode_tok_s, total_tok_s, generated).

    TTFT is measured as a separate 1-token generation, which isolates prefill.
    Then a full run gives total time; decode speed is (total - ttft) over the
    remaining tokens. Reporting a single "tokens/sec" that silently folds prefill
    into decode is the most common benchmarking mistake in this space.
    """
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)

    for _ in range(warmup):
        model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    model.generate(**enc, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    generated = out.shape[1] - enc["input_ids"].shape[1]
    decode_s = max(total - ttft, 1e-6)
    return ttft * 1000, (generated - 1) / decode_s, generated / total, generated


@torch.no_grad()
def perplexity(model, tok, text: str) -> float:
    """Quality proxy. Lower is better; compare variants, never absolute values."""
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss))


def bench_batching(model, tok, batch_sizes: list[int], max_new_tokens: int) -> list[dict]:
    """Throughput vs batch size -- the clearest demonstration of bandwidth-bound decoding."""
    tok.padding_side = "left"
    rows = []
    for bs in batch_sizes:
        msgs = [{"role": "user", "content": PROMPT}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([text] * bs, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_per_seq = out.shape[1] - enc["input_ids"].shape[1]
        rows.append(
            {
                "batch_size": bs,
                "seconds": round(elapsed, 3),
                "tokens_per_seq": gen_per_seq,
                "total_tok_s": round(bs * gen_per_seq / elapsed, 1),
                "per_seq_tok_s": round(gen_per_seq / elapsed, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            }
        )
        print(f"    batch {bs:>3}: {rows[-1]['total_tok_s']:>7.1f} tok/s total  "
              f"{rows[-1]['per_seq_tok_s']:>6.1f} tok/s per seq  "
              f"peak {rows[-1]['peak_gb']:.2f} GB")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--variants", nargs="+", default=["fp32", "fp16", "bf16", "int8", "int4"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--batch-variant", default="bf16", help="variant used for the batching sweep")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-batching", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This script expects a CUDA GPU.")

    props = torch.cuda.get_device_properties(0)
    print("=" * 86)
    print(f"  Inference benchmark  |  {args.model}")
    print(f"  {props.name}  |  {props.total_memory/1024**3:.1f} GB  |  "
          f"sm_{props.major}{props.minor}  |  torch {torch.__version__}")
    print("=" * 86)

    results: list[Result] = []
    batching: list[dict] = []

    for variant in args.variants:
        print(f"\n[{variant}]")
        try:
            model, tok, load_s, weights_gb = load(args.model, variant)
            print(f"    loaded in {load_s:.1f}s, weights {weights_gb:.3f} GB")

            ttft, dec, tot, n = measure_generation(model, tok, args.max_new_tokens)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            ppl = None if args.skip_quality else perplexity(model, tok, QUALITY_TEXT)

            r = Result(variant, round(load_s, 2), round(weights_gb, 3), round(peak, 3),
                       round(ttft, 1), round(dec, 1), round(tot, 1), n,
                       None if ppl is None else round(ppl, 3))
            results.append(r)
            print(f"    TTFT {r.ttft_ms:.0f} ms | decode {r.decode_tok_s:.1f} tok/s | "
                  f"peak {r.peak_gb:.2f} GB" + (f" | ppl {r.perplexity:.3f}" if ppl else ""))

            if not args.skip_batching and variant == args.batch_variant:
                print(f"  -- batching sweep ({variant}) --")
                batching = bench_batching(model, tok, args.batch_sizes, args.max_new_tokens)

            del model, tok
            free_gpu()

        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            results.append(Result(variant, 0, 0, 0, 0, 0, 0, 0, error=f"{type(exc).__name__}: {exc}"))
            free_gpu()

    # ------------------------------------------------------------- report
    ok = [r for r in results if r.error is None]
    print("\n" + "=" * 86)
    print(f"{'variant':<9}{'load s':>8}{'weights':>9}{'peak GB':>9}{'TTFT ms':>9}"
          f"{'decode t/s':>12}{'perplexity':>12}")
    print("-" * 86)
    for r in ok:
        ppl = f"{r.perplexity:.3f}" if r.perplexity is not None else "-"
        print(f"{r.variant:<9}{r.load_seconds:>8.1f}{r.weights_gb:>9.3f}{r.peak_gb:>9.3f}"
              f"{r.ttft_ms:>9.0f}{r.decode_tok_s:>12.1f}{ppl:>12}")

    if ok:
        ref = next((r for r in ok if r.variant == "bf16"), ok[0])
        print(f"\nrelative to {ref.variant}:")
        for r in ok:
            if r is ref:
                continue
            mem = r.weights_gb / ref.weights_gb if ref.weights_gb else 0
            spd = r.decode_tok_s / ref.decode_tok_s if ref.decode_tok_s else 0
            q = ""
            if r.perplexity and ref.perplexity:
                q = f", perplexity {100*(r.perplexity/ref.perplexity - 1):+.1f}%"
            print(f"  {r.variant:<6} weights {mem:.2f}x   decode speed {spd:.2f}x{q}")

    if batching:
        print("\nbatching sweep:")
        base = batching[0]["total_tok_s"]
        for b in batching:
            print(f"  batch {b['batch_size']:>3}: {b['total_tok_s']:>7.1f} tok/s "
                  f"({b['total_tok_s']/base:>5.2f}x)   per-seq {b['per_seq_tok_s']:>6.1f} tok/s   "
                  f"peak {b['peak_gb']:.2f} GB")
        print("\n  Total throughput rises far faster than per-sequence speed falls.")
        print("  That gap is the entire economic argument for batched serving.")

    payload = {
        "model": args.model,
        "gpu": props.name,
        "vram_gb": round(props.total_memory / 1024**3, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "max_new_tokens": args.max_new_tokens,
        "variants": [asdict(r) for r in results],
        "batching": batching,
        "batching_variant": args.batch_variant,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nresults -> {Path(args.out).resolve()}")
    print("Plot them:  python plot_results.py")


if __name__ == "__main__":
    main()
