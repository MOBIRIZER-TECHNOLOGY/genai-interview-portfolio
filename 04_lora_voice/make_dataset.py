"""
Build a speech dataset for domain adaptation of an ASR model — by synthesising
it with a TTS model.

    python make_dataset.py --train 240 --eval 60

## The task

General ASR models transcribe ordinary English well and mangle domain jargon.
Say "tee ell em three thirty" to Whisper and you get "TLM 330", "telem 330",
"TLM three thirty" — anything but the canonical `TLM-330`. Same for
`atlas-dispatch`, `nw-barcode-ocr-v2`, `TimescaleDB`.

So the label is not just "what words were said", it is **what the written form
should be**: spoken "tee ell em three thirty" -> written `TLM-330`. That's domain
vocabulary *and* inverse text normalisation in one task, and it's the real job an
ASR system does inside an ops tool.

## Why synthetic audio

Because the alternative is recording hundreds of utterances, and this needs to
run on your machine in ten minutes. TTS-generated training data for ASR domain
adaptation is a real, widely used technique — but it has a real limitation, and
the honest thing is to name it: **synthetic speech is too clean**. A model
trained only on TTS output learns TTS acoustics and can do *worse* on real
microphones.

Two mitigations here, both standard:
  1. **Speaker variety** — sample many different x-vector speaker embeddings.
  2. **Augmentation** — additive noise, gain, and a light room-ish decay, so the
     model can't rely on pristine input.

`--clean` turns augmentation off, which is worth running once to see the
difference in the eval numbers.

Output:
    data/train/*.wav + data/train.jsonl      {"audio": "...", "text": "TLM-330 ..."}
    data/eval/*.wav  + data/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

SR = 16_000  # Whisper's required sample rate

# --------------------------------------------------------------- vocabulary
# (written form, spoken form) -- the mapping the model has to learn.
CODES = [
    ("TLM-101", "tee ell em one oh one"),
    ("TLM-204", "tee ell em two oh four"),
    ("TLM-330", "tee ell em three thirty"),
    ("TLM-402", "tee ell em four oh two"),
    ("DSP-500", "dee ess pee five hundred"),
    ("VIS-207", "vee eye ess two oh seven"),
    ("CON-401", "see oh en four oh one"),
]

COMPONENTS = [
    ("atlas-dispatch", "atlas dispatch"),
    ("atlas-telemetry", "atlas telemetry"),
    ("atlas-vision", "atlas vision"),
    ("atlas-console", "atlas console"),
    ("atlas-sim", "atlas sim"),
    ("atlas-roll", "atlas roll"),
]

MODELS = [
    ("nw-pallet-detect-v4", "en double you pallet detect vee four"),
    ("nw-barcode-ocr-v2", "en double you barcode oh see arr vee two"),
    ("nw-damage-clf-v1", "en double you damage see ell eff vee one"),
]

SEVS = [("SEV1", "sev one"), ("SEV2", "sev two"), ("SEV3", "sev three")]
CELLS = ["Rotterdam", "Hamburg", "Memphis", "Lyon", "Bristol"]
JARGON = [
    ("shed mode", "shed mode"),
    ("the Rotterdam rule", "the rotterdam rule"),
    ("the starvation guard", "the starvation guard"),
    ("a hypertable chunk", "a hypertable chunk"),
    ("the auction loop", "the auction loop"),
    ("the manual inspection lane", "the manual inspection lane"),
]

TEMPLATES = [
    "We are seeing {code} on {comp} in {cell}.",
    "{comp} threw {code} again, escalating as {sev}.",
    "Raise a {sev} for {comp}, error {code}.",
    "{cell} reports {jargon} on {comp}.",
    "{model} confidence dropped, routing to {jargon}.",
    "{code} cleared on {comp} after a restart.",
    "{comp} is in {jargon} and this is a {sev}.",
    "Page on call, {code} in {cell} is a {sev}.",
    "{model} is failing on {comp} in {cell}.",
    "Check {jargon} before you file the {sev}.",
]


def make_pair(rng: random.Random) -> tuple[str, str]:
    """Return (written_text, spoken_text) for one utterance."""
    code_w, code_s = rng.choice(CODES)
    comp_w, comp_s = rng.choice(COMPONENTS)
    model_w, model_s = rng.choice(MODELS)
    sev_w, sev_s = rng.choice(SEVS)
    jar_w, jar_s = rng.choice(JARGON)
    cell = rng.choice(CELLS)
    tpl = rng.choice(TEMPLATES)

    written = tpl.format(code=code_w, comp=comp_w, model=model_w, sev=sev_w, jargon=jar_w, cell=cell)
    spoken = tpl.format(code=code_s, comp=comp_s, model=model_s, sev=sev_s, jargon=jar_s, cell=cell)
    return written, spoken.lower()


# ------------------------------------------------------------ augmentation


def augment(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light, realistic degradation so the model can't rely on pristine TTS audio."""
    # random gain (mic distance / level differences)
    wav = wav * rng.uniform(0.55, 1.25)

    # additive noise at a random SNR (room tone, fans, warehouse floor)
    snr_db = rng.uniform(12, 30)
    sig_pow = float(np.mean(wav**2)) + 1e-12
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    wav = wav + rng.normal(0, np.sqrt(noise_pow), wav.shape)

    # a single early reflection -- a crude but effective stand-in for room reverb
    if rng.random() < 0.5:
        delay = int(rng.uniform(0.010, 0.045) * SR)
        echo = np.zeros_like(wav)
        echo[delay:] = wav[:-delay] * rng.uniform(0.10, 0.28)
        wav = wav + echo

    peak = np.max(np.abs(wav))
    if peak > 0.99:
        wav = wav / peak * 0.99
    return wav.astype(np.float32)


# ---------------------------------------------------------------- speakers


def load_speakers(device: str, per_speaker: int = 12):
    """Load CMU Arctic x-vectors straight from the HF repo's zip.

    SpeechT5 takes speaker identity as a 512-dim x-vector *input*, not as model
    weights -- so we get many voices out of one TTS model for free.

    We read the zip directly rather than via `load_dataset`, because that repo
    ships a legacy loading script which `datasets` 3.0+ refuses to execute. This
    also lets us balance across *speakers* (jmk, rms, clb, slt, bdl, ...) instead
    of sampling 7931 utterance-level vectors that are mostly the same few voices.
    """
    import io
    import zipfile

    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset")
    zf = zipfile.ZipFile(path)

    by_speaker: dict[str, list[str]] = {}
    for name in zf.namelist():
        if not name.endswith(".npy"):
            continue
        # "spkrec-xvect/cmu_us_jmk_arctic-wav-arctic_a0219.npy" -> "jmk"
        stem = name.rsplit("/", 1)[-1]
        speaker = stem.split("-")[0].replace("cmu_us_", "").replace("_arctic", "")
        by_speaker.setdefault(speaker, []).append(name)

    vectors, ids = [], []
    for speaker, files in sorted(by_speaker.items()):
        for f in sorted(files)[:per_speaker]:
            vectors.append(np.load(io.BytesIO(zf.read(f))))
            ids.append(speaker)

    return torch.tensor(np.stack(vectors), dtype=torch.float32).to(device), ids


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--eval", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="skip augmentation")
    ap.add_argument("--voices-per-speaker", type=int, default=12,
                    help="x-vectors sampled per CMU Arctic speaker")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)

    print("loading SpeechT5 TTS (first run downloads ~600 MB) ...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()

    speakers, speaker_ids = load_speakers(device, per_speaker=args.voices_per_speaker)
    print(f"speakers: {len(speakers)} x-vectors from {len(set(speaker_ids))} distinct voices "
          f"({', '.join(sorted(set(speaker_ids)))})")
    print(f"device: {device} | augmentation: {'off' if args.clean else 'on'}")

    for split, n, seed in [("train", args.train, args.seed), ("eval", args.eval, args.seed + 9999)]:
        rng = random.Random(seed)
        nrng = np.random.default_rng(seed)
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Distinct utterances only -- duplicated audio inflates your training
        # count without adding information, and can leak train into eval.
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        while len(pairs) < n:
            w, s = make_pair(rng)
            if w in seen:
                continue
            seen.add(w)
            pairs.append((w, s))

        rows = []
        for i, (written, spoken) in enumerate(pairs):
            inputs = processor(text=spoken, return_tensors="pt").to(device)
            spk_i = int(nrng.integers(len(speakers)))
            spk = speakers[spk_i].unsqueeze(0)
            with torch.no_grad():
                wav = tts.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
            wav = wav.cpu().numpy().astype(np.float32)
            if not args.clean:
                wav = augment(wav, nrng)

            name = f"{i:04d}.wav"
            sf.write(split_dir / name, wav, SR)
            rows.append(
                {
                    "audio": f"{split}/{name}",
                    "text": written,
                    "spoken": spoken,
                    "speaker": speaker_ids[spk_i],
                    "seconds": round(len(wav) / SR, 2),
                }
            )
            if (i + 1) % 40 == 0:
                print(f"  {split}: {i+1}/{n}")

        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total_min = sum(r["seconds"] for r in rows) / 60
        print(f"{path}  {len(rows)} clips, {total_min:.1f} minutes of audio")

    sample = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    print("\nExample:")
    print(f"  spoken (to TTS):  {sample['spoken']}")
    print(f"  target transcript: {sample['text']}")
    print("\nNext:  python train_lora.py --epochs 6")


if __name__ == "__main__":
    main()
