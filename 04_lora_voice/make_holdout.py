"""
Build a held-out eval set from a DIFFERENT speech engine than the one used for
training.

    python make_holdout.py --n 60          # Windows SAPI (David + Zira)

## Why this is worth having

Training data comes from SpeechT5, a neural TTS. Evaluating on SpeechT5 audio
cannot distinguish two very different outcomes:

  (a) the model learned the *vocabulary* -- generalises to any speech, or
  (b) the model learned *SpeechT5's acoustics* -- collapses on anything else.

Windows SAPI (`Microsoft David` / `Microsoft Zira`) is a completely different
synthesis family: concatenative/formant, built decades apart from SpeechT5, with
audibly different prosody, timbre and artefacts. It shares essentially nothing
with the training distribution except the English language.

So a LoRA that holds up on SAPI audio has learned something transferable. One
that collapses learned the TTS.

## What this is NOT

**This is not a substitute for real recordings.** SAPI is still synthetic: no
room, no microphone, no breath, no hesitation, no background noise, and a speaking
style no human uses. It closes part of the gap -- the "did it overfit to one
engine" part -- and leaves the rest open.

For genuinely real audio use `record.py` (microphone) or `import_audio.py`
(existing recordings). This exists because it runs in 90 seconds with no hardware.

Requires Windows + pywin32. On Linux/macOS use `espeak-ng` or `say` and adapt
`synth_sapi`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from make_dataset import SR, augment, make_pair  # noqa: E402

# SpeechAudioFormatType.SAFT16kHz16BitMono -- Whisper's required rate, no resample
SAFT_16KHZ_16BIT_MONO = 18
SSFM_CREATE_FOR_WRITE = 3


def synth_sapi(text: str, path: Path, voice_index: int, rate: int = 0) -> None:
    """Speak `text` into a 16 kHz mono WAV using Windows SAPI."""
    import win32com.client as win32

    speaker = win32.Dispatch("SAPI.SpVoice")
    voices = speaker.GetVoices()
    speaker.Voice = voices.Item(voice_index % voices.Count)
    speaker.Rate = rate                     # -10..10, 0 is default

    stream = win32.Dispatch("SAPI.SpFileStream")
    fmt = win32.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = SAFT_16KHZ_16BIT_MONO
    stream.Format = fmt
    stream.Open(str(path.resolve()), SSFM_CREATE_FOR_WRITE, False)
    try:
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
    finally:
        stream.Close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data_sapi"))
    ap.add_argument("--split", default="eval", choices=["train", "eval"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=4242,
                    help="matches record.py's default so the SAPI and real sets "
                         "cover the same sentences and are directly comparable")
    ap.add_argument("--clean", action="store_true", help="skip augmentation")
    ap.add_argument("--rates", nargs="+", type=int, default=[-2, 0, 2],
                    help="SAPI speaking rates to vary across clips")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    try:
        import win32com.client  # noqa: F401
    except ImportError:
        raise SystemExit("needs Windows + pywin32:  uv pip install pywin32")

    out = Path(args.out)
    split_dir = out / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    import win32com.client as win32

    voices = win32.Dispatch("SAPI.SpVoice").GetVoices()
    names = [voices.Item(i).GetDescription() for i in range(voices.Count)]
    print(f"SAPI voices: {voices.Count}")
    for n in names:
        print(f"  - {n}")
    print(f"rates: {args.rates} | augmentation: {'off' if args.clean else 'on'}\n")

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)

    pairs, seen = [], set()
    while len(pairs) < args.n:
        written, spoken = make_pair(rng)
        if written in seen:
            continue
        seen.add(written)
        pairs.append((written, spoken))

    rows = []
    for i, (written, spoken) in enumerate(pairs):
        name = f"{i:04d}.wav"
        path = split_dir / name
        voice_i = i % voices.Count
        rate = args.rates[i % len(args.rates)]

        synth_sapi(spoken, path, voice_i, rate)

        wav, sr = sf.read(path, dtype="float32")
        if sr != SR:
            raise RuntimeError(f"SAPI wrote {sr} Hz, expected {SR}")
        if not args.clean:
            wav = augment(wav, nrng)
            sf.write(path, wav, SR)

        rows.append({
            "audio": f"{args.split}/{name}",
            "text": written,
            "spoken": spoken,
            "seconds": round(len(wav) / SR, 2),
            "source": "sapi",
            "voice": names[voice_i].replace("Microsoft ", "").replace(" Desktop", ""),
            "rate": rate,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.n}")

    manifest = out / f"{args.split}.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_min = sum(r["seconds"] for r in rows) / 60
    print(f"\n{manifest}  {len(rows)} clips, {total_min:.1f} minutes")
    print(f"\nNext:  python evaluate.py --data {out.name}")
    print("       Compare the numbers against the in-distribution SpeechT5 eval.")
    print("       A big gap means the LoRA learned the engine, not the vocabulary.")


if __name__ == "__main__":
    main()
