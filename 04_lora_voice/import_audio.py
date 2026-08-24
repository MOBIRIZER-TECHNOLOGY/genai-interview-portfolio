"""
Import audio you already have (phone voice memos, meeting clips, call recordings)
into the dataset layout, converting to what Whisper needs.

    # 1. drop your files in a folder, write transcripts.tsv next to them
    # 2.
    python import_audio.py --src C:\\recordings --split eval

`transcripts.tsv` is one line per file, TAB separated:

    memo_001.m4a<TAB>Page on call, TLM-101 in Lyon is a SEV1.
    memo_002.m4a<TAB>Bristol reports the starvation guard on atlas-roll.

Transcripts must be in the **canonical written form you want the model to
produce** (`TLM-101`, not "TLM one oh one"), because that is the training target.

## What this does that matters

- **Resamples to 16 kHz mono.** Whisper's mel filterbank assumes 16 kHz. Feeding
  44.1 kHz produces plausible-but-worse output with nothing in the logs, so this
  converts explicitly and tells you when it did.
- **Reports clipping and level.** A dataset of clipped phone recordings will
  train a model that only works on clipped phone recordings.
- **Refuses silently-broken files** rather than writing a zero-length WAV that
  fails much later, in training, with a confusing error.

Any format librosa/soundfile can open works: wav, flac, ogg, mp3, m4a
(m4a/mp3 need ffmpeg on PATH).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SR = 16_000


def load_any(path: Path) -> tuple[np.ndarray, int]:
    """Load any supported audio file as mono float32, returning (wav, orig_sr)."""
    import librosa

    # sr=None keeps the native rate so we can report what we converted from
    wav, sr = librosa.load(str(path), sr=None, mono=True)
    return wav.astype(np.float32), int(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing audio + transcripts.tsv")
    ap.add_argument("--out", default=str(HERE / "data_real"))
    ap.add_argument("--split", default="eval", choices=["train", "eval"])
    ap.add_argument("--transcripts", default="transcripts.tsv")
    ap.add_argument("--peak-normalize", action="store_true",
                    help="scale each clip so its peak is 0.95 (use only if levels vary wildly)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    src = Path(args.src)
    tsv = src / args.transcripts
    if not tsv.exists():
        raise SystemExit(
            f"no {args.transcripts} in {src.resolve()}\n\n"
            "  Create it, one line per file, TAB separated:\n"
            "    memo_001.m4a\tPage on call, TLM-101 in Lyon is a SEV1."
        )

    pairs: list[tuple[str, str]] = []
    for lineno, line in enumerate(tsv.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" not in line:
            print(f"  ! {args.transcripts}:{lineno} has no TAB, skipping: {line[:60]}")
            continue
        fname, text = line.split("\t", 1)
        pairs.append((fname.strip(), text.strip()))

    print(f"{len(pairs)} transcript line(s) in {tsv}")

    out = Path(args.out)
    split_dir = out / args.split
    if not args.dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped, resampled = [], 0, 0
    for fname, text in pairs:
        path = src / fname
        if not path.exists():
            print(f"  ! missing: {fname}")
            skipped += 1
            continue

        try:
            wav, sr = load_any(path)
        except Exception as exc:
            print(f"  ! cannot read {fname}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            resampled += 1

        if wav.size < int(0.2 * SR):
            print(f"  ! {fname}: only {wav.size/SR:.2f}s, skipping")
            skipped += 1
            continue

        peak = float(np.max(np.abs(wav)))
        rms_db = 20 * np.log10(float(np.sqrt((wav**2).mean())) + 1e-12)
        clipped = int((np.abs(wav) > 0.999).sum())

        flags = []
        if clipped > 8:
            flags.append(f"CLIPPED({clipped})")
        if rms_db < -45:
            flags.append("VERY QUIET")
        if peak < 0.02:
            flags.append("NEAR SILENT")

        if args.peak_normalize and peak > 0:
            wav = wav / peak * 0.95

        name = f"{len(rows):04d}.wav"
        note = f"  [{sr} Hz -> {SR}]" if sr != SR else ""
        print(f"  {fname:<32} {wav.size/SR:>6.1f}s  rms {rms_db:>5.0f} dBFS"
              f"{note}  {' '.join(flags)}")

        if not args.dry_run:
            sf.write(split_dir / name, wav.astype(np.float32), SR)

        rows.append({
            "audio": f"{args.split}/{name}",
            "text": text,
            "seconds": round(len(wav) / SR, 2),
            "source": "real",
            "original_file": fname,
            "original_sr": sr,
        })

    if args.dry_run:
        print(f"\ndry run: would import {len(rows)}, skip {skipped}")
        return

    manifest = out / f"{args.split}.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_min = sum(r["seconds"] for r in rows) / 60
    print(f"\n{manifest}  {len(rows)} clips, {total_min:.1f} minutes"
          f"  ({resampled} resampled, {skipped} skipped)")
    print(f"\nNext:  python evaluate.py --data {out.name}")


if __name__ == "__main__":
    main()
