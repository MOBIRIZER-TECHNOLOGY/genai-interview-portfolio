"""
Base Whisper vs LoRA Whisper, on held-out audio.

    python evaluate.py
    python evaluate.py --show 8            # print transcripts side by side

Metrics:

  **WER** (word error rate) = (substitutions + insertions + deletions) / reference
      words. The standard ASR metric. Lower is better; 0.0 is perfect. It can
      exceed 1.0 if the model hallucinates more words than were spoken.

  **CER** (character error rate) — same idea per character. Reported because WER
      is brutally coarse on identifier tokens: getting `TLM-330` as `TLM 330`
      is one whole word error but only one character error, and those two
      failures are very different in severity for downstream parsing.

  **Domain-term accuracy** — the metric that actually matters here. WER averages
      over every word in the sentence, most of which are ordinary English that
      both models get right. That dilutes the thing we set out to fix. This
      metric asks only: of the domain terms that appear in the reference
      (`TLM-330`, `atlas-dispatch`, `SEV2`, ...), how many appear **verbatim** in
      the hypothesis?

Text is normalised the same way for both arms before scoring (lowercase, strip
punctuation that isn't part of an identifier). Comparing a normalised hypothesis
against an unnormalised reference is a common way to accidentally report a
better WER than you earned.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
SR = 16_000

# Terms we specifically trained the model to produce in canonical written form.
DOMAIN_TERMS = re.compile(
    r"\b(?:TLM-\d{3}|DSP-\d{3}|VIS-\d{3}|CON-\d{3}|SEV[123]|atlas-[a-z]+|nw-[a-z-]+-v\d)\b"
)

# Keep [a-z0-9-] so identifiers survive; drop sentence punctuation.
PUNCT = re.compile(r"[^\w\s-]")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = PUNCT.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def domain_recall(reference: str, hypothesis: str) -> tuple[int, int]:
    """(terms recovered verbatim, terms present in the reference)."""
    terms = DOMAIN_TERMS.findall(reference)
    if not terms:
        return 0, 0
    hyp = normalize(hypothesis)
    return sum(1 for t in terms if normalize(t) in hyp), len(terms)


@torch.no_grad()
def transcribe(model, processor, rows: list[dict], root: Path, batch_size: int) -> tuple[list[str], float]:
    import soundfile as sf

    model.eval()
    outs: list[str] = []
    t0 = time.perf_counter()

    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        wavs = [sf.read(root / r["audio"], dtype="float32")[0] for r in chunk]
        feats = processor.feature_extractor(
            wavs, sampling_rate=SR, return_tensors="pt"
        ).input_features.to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(feats, max_new_tokens=96, num_beams=1)
        outs.extend(processor.batch_decode(ids, skip_special_tokens=True))

    return outs, time.perf_counter() - t0


def score(hyps: list[str], rows: list[dict]) -> dict:
    import jiwer

    refs = [r["text"] for r in rows]
    refs_n = [normalize(r) for r in refs]
    hyps_n = [normalize(h) for h in hyps]

    hit = tot = 0
    perfect = 0
    for ref, hyp in zip(refs, hyps):
        h, t = domain_recall(ref, hyp)
        hit += h
        tot += t
        if normalize(ref) == normalize(hyp):
            perfect += 1

    return {
        "wer": float(jiwer.wer(refs_n, hyps_n)),
        "cer": float(jiwer.cer(refs_n, hyps_n)),
        "domain_term_accuracy": hit / tot if tot else 0.0,
        "domain_terms": tot,
        "sentence_accuracy": perfect / len(rows),
        "n": len(rows),
    }


def report(name: str, s: dict, seconds: float) -> None:
    print(f"\n### {name}")
    print(f"  WER                   {s['wer']:7.1%}   (lower is better)")
    print(f"  CER                   {s['cer']:7.1%}")
    print(f"  domain-term accuracy  {s['domain_term_accuracy']:7.1%}   "
          f"({s['domain_terms']} terms across {s['n']} clips)")
    print(f"  exact sentence match  {s['sentence_accuracy']:7.1%}")
    print(f"  {seconds:.1f}s for {s['n']} clips ({s['n']/seconds:.1f}/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--lora", default=str(HERE / "lora-out"))
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--out", default=str(HERE / "eval_results.json"))
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    root = Path(args.data)
    rows = [json.loads(l) for l in (root / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        rows = rows[: args.n]

    processor = WhisperProcessor.from_pretrained(args.model, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).to("cuda")
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print("=" * 74)
    print(f"  Whisper: base vs LoRA  |  {len(rows)} held-out clips  |  greedy decoding")
    print("=" * 74)

    results: dict = {"model": args.model, "n": len(rows)}
    base_hyps: list[str] = []

    if not args.skip_base:
        base_hyps, t = transcribe(model, processor, rows, root, args.batch_size)
        base = score(base_hyps, rows)
        report("BASE Whisper (no fine-tuning)", base, t)
        results["base"] = base

    model = PeftModel.from_pretrained(model, args.lora)
    lora_hyps, t = transcribe(model, processor, rows, root, args.batch_size)
    tuned = score(lora_hyps, rows)
    report("LoRA fine-tuned", tuned, t)
    results["lora"] = tuned

    if not args.skip_base:
        print("\n### Delta")
        b = results["base"]
        print(f"  WER                   {b['wer']:7.1%}  ->  {tuned['wer']:7.1%}   "
              f"({(tuned['wer']-b['wer'])*100:+.1f} pts, "
              f"{100*(b['wer']-tuned['wer'])/b['wer'] if b['wer'] else 0:.0f}% relative)")
        print(f"  CER                   {b['cer']:7.1%}  ->  {tuned['cer']:7.1%}   "
              f"({(tuned['cer']-b['cer'])*100:+.1f} pts)")
        print(f"  domain-term accuracy  {b['domain_term_accuracy']:7.1%}  ->  "
              f"{tuned['domain_term_accuracy']:7.1%}   "
              f"({(tuned['domain_term_accuracy']-b['domain_term_accuracy'])*100:+.1f} pts)")
        print(f"  exact sentence match  {b['sentence_accuracy']:7.1%}  ->  "
              f"{tuned['sentence_accuracy']:7.1%}")

    for i in range(min(args.show, len(rows))):
        print("\n" + "-" * 70)
        print("REF :", rows[i]["text"])
        if base_hyps:
            print("BASE:", base_hyps[i].strip())
        print("LORA:", lora_hyps[i].strip())

    results["samples"] = [
        {"ref": rows[i]["text"],
         "base": base_hyps[i].strip() if base_hyps else None,
         "lora": lora_hyps[i].strip()}
        for i in range(min(15, len(rows)))
    ]
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
