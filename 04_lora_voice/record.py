"""
Record REAL audio for the Whisper project, one prompt at a time.

    python record.py --n 120                 # record 120 clips into data_real/
    python record.py --split eval --n 60     # a real held-out set
    python record.py --list-devices
    python record.py --device 2

## Why this exists

Everything in `make_dataset.py` is TTS-synthesised, and that has a failure mode
the synthetic eval cannot detect: the model may have learned **SpeechT5
acoustics** rather than the vocabulary, and could be *worse* than base Whisper on
a real microphone. The only honest test is real speech.

A real **eval** set is worth far more than a real training set. 60 recorded clips
used purely for evaluation tell you whether the synthetic training transferred.
Recording 60 clips takes about ten minutes.

## How a session works

For each prompt you see the target transcript and a pronunciation hint:

    [ 7/60]  Page on call, TLM-101 in Lyon is a SEV1.
             say: "tee ell em one oh one" ... "sev one"

    ENTER = start recording, ENTER again = stop
    r = retake   s = skip   p = play back   q = save and quit

Say it the way you naturally would — this is the point of real audio. Your
hesitations, your room, your microphone. Don't perform it.

Progress is written after every clip, so you can quit and resume any time; the
script picks up where you left off.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from make_dataset import SR, make_pair  # noqa: E402

# Terms that need a pronunciation hint, so every speaker says them consistently.
HINTS = {
    "TLM-101": "tee ell em one oh one", "TLM-204": "tee ell em two oh four",
    "TLM-330": "tee ell em three thirty", "TLM-402": "tee ell em four oh two",
    "DSP-500": "dee ess pee five hundred", "VIS-207": "vee eye ess two oh seven",
    "CON-401": "see oh en four oh one",
    "SEV1": "sev one", "SEV2": "sev two", "SEV3": "sev three",
    "nw-pallet-detect-v4": "en double you pallet detect vee four",
    "nw-barcode-ocr-v2": "en double you barcode oh see arr vee two",
    "nw-damage-clf-v1": "en double you damage see ell eff vee one",
}


def hint_for(text: str) -> str:
    found = [f'{k} = "{v}"' for k, v in HINTS.items() if k in text]
    return "   ".join(found)


# --------------------------------------------------------------- recording


def list_devices() -> None:
    import sounddevice as sd

    print("Input devices:\n")
    any_in = False
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            any_in = True
            print(f"  [{i}] {d['name']}   ({d['max_input_channels']} ch, "
                  f"default {d['default_samplerate']:.0f} Hz)")
    if not any_in:
        print("  NONE FOUND.\n")
        print("  On a remote desktop you must enable microphone redirection:")
        print("    Windows RDP  -> Show Options > Local Resources > Remote audio")
        print("                    > Settings > 'Record from this computer'")
        print("  Otherwise run this script on a machine with a physical mic, and")
        print("  copy the resulting data_real/ folder back.")


def record_clip(device: int | None, max_seconds: float) -> np.ndarray:
    """Push-to-talk: returns mono float32 at SR. ENTER stops."""
    import sounddevice as sd

    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"  ! {status}", file=sys.stderr)
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        device=device, callback=callback):
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass

    if not frames:
        return np.zeros(0, np.float32)
    wav = np.concatenate(frames, axis=0)[:, 0]
    return wav[: int(max_seconds * SR)]


def trim_silence(wav: np.ndarray, thresh_db: float = -42.0, pad_ms: int = 120) -> np.ndarray:
    """Trim leading/trailing silence, keeping a short pad.

    Whisper pads everything to 30 s anyway, so this is not about length -- it is
    about not training on ten seconds of you reaching for the keyboard.
    """
    if wav.size == 0:
        return wav
    frame = 256
    n = len(wav) // frame
    if n == 0:
        return wav
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    loud = np.where(db > thresh_db)[0]
    if loud.size == 0:
        return wav
    pad = int(pad_ms / 1000 * SR)
    start = max(0, loud[0] * frame - pad)
    end = min(len(wav), (loud[-1] + 1) * frame + pad)
    return wav[start:end]


def quality_report(wav: np.ndarray) -> tuple[str, bool]:
    """Return (message, ok). Catches the two mistakes that ruin a session."""
    if wav.size < int(0.3 * SR):
        return "TOO SHORT - nothing recorded?", False
    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt((wav**2).mean()))
    db = 20 * np.log10(rms + 1e-12)
    clipped = int((np.abs(wav) > 0.999).sum())

    msg = f"{len(wav)/SR:.1f}s  peak {peak:.2f}  rms {db:.0f} dBFS"
    if clipped > 8:
        return msg + f"  !! CLIPPED ({clipped} samples) - move back or lower gain", False
    if db < -45:
        return msg + "  !! VERY QUIET - move closer or raise gain", False
    if peak < 0.02:
        return msg + "  !! SILENT - wrong input device?", False
    return msg + "  ok", True


def play(wav: np.ndarray) -> None:
    try:
        import sounddevice as sd

        sd.play(wav, SR)
        sd.wait()
    except Exception as exc:
        print(f"  (playback unavailable: {exc})")


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data_real"))
    ap.add_argument("--split", default="eval", choices=["train", "eval"],
                    help="a REAL eval set is the higher-value thing to record first")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--seed", type=int, default=4242,
                    help="prompt seed; the default differs from make_dataset so "
                         "your real clips are not the same sentences as the synthetic set")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(f"{exc}\n\n  uv pip install sounddevice soundfile")

    if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
        print("No audio input device found.\n")
        list_devices()
        raise SystemExit(1)

    out = Path(args.out)
    split_dir = out / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / f"{args.split}.jsonl"

    # resume
    done: dict[str, dict] = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["text"]] = r
        print(f"resuming: {len(done)} clip(s) already recorded in {manifest}")

    rng = random.Random(args.seed)
    prompts, seen = [], set()
    while len(prompts) < args.n:
        written, _ = make_pair(rng)
        if written in seen:
            continue
        seen.add(written)
        prompts.append(written)

    todo = [p for p in prompts if p not in done]
    dev_name = sd.query_devices(args.device if args.device is not None else None,
                                kind="input")["name"]

    print("=" * 74)
    print(f"  Recording REAL audio  |  split={args.split}  |  {len(todo)} to go")
    print(f"  device: {dev_name}   {SR} Hz mono")
    print("=" * 74)
    print("  ENTER = start/stop    r = retake    s = skip    p = play    q = quit\n")
    print("  Say it naturally. Hesitations and room noise are the point.\n")

    rows = list(done.values())

    def flush() -> None:
        with open(manifest, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        for idx, text in enumerate(todo, 1):
            while True:
                print(f"[{idx}/{len(todo)}]  {text}")
                h = hint_for(text)
                if h:
                    print(f"          say: {h}")
                cmd = input("          ENTER to record > ").strip().lower()

                if cmd == "q":
                    raise KeyboardInterrupt
                if cmd == "s":
                    print("          skipped\n")
                    break

                print("          RECORDING... ENTER to stop")
                wav = record_clip(args.device, args.max_seconds)
                if not args.no_trim:
                    wav = trim_silence(wav)

                msg, ok = quality_report(wav)
                print(f"          {msg}")

                choice = input("          ENTER=keep  r=retake  p=play  s=skip  q=quit > ").strip().lower()
                while choice == "p":
                    play(wav)
                    choice = input("          ENTER=keep  r=retake  s=skip  q=quit > ").strip().lower()
                if choice == "q":
                    raise KeyboardInterrupt
                if choice == "r":
                    print()
                    continue
                if choice == "s":
                    print("          skipped\n")
                    break
                if not ok and choice != "":
                    continue

                name = f"{len(rows):04d}.wav"
                sf.write(split_dir / name, wav.astype(np.float32), SR)
                rows.append({
                    "audio": f"{args.split}/{name}",
                    "text": text,
                    "seconds": round(len(wav) / SR, 2),
                    "source": "real",
                })
                flush()
                print(f"          saved {name}\n")
                break

    except KeyboardInterrupt:
        print("\n\n  stopping early (progress is saved)")

    flush()
    total_min = sum(r["seconds"] for r in rows) / 60
    print(f"\n{manifest}  {len(rows)} clips, {total_min:.1f} minutes")
    print(f"\nNext:  python evaluate.py --data {out.name}")
    print(f"       (compares base Whisper vs your synthetic-trained LoRA on REAL audio)")


if __name__ == "__main__":
    main()
