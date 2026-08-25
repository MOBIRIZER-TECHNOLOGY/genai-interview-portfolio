"""
Does the adapter forget general English? Measure it, don't assume it.

    python probe_general_speech.py                    # synthesise + evaluate
    python probe_general_speech.py --skip-synth       # reuse data_general/

## Why this exists

`evaluate.py` measures the thing the adapter was trained for: domain WER and
domain-term accuracy, both on sentences full of `TLM-330` and `SEV1`. It cannot
see the cost side of the trade, and there is always a cost side.

Fine-tuning an ASR model on a narrow domain risks **catastrophic forgetting**:
the model gets very good at your jargon and quietly worse at ordinary speech.
The failure is invisible to a domain eval — every number goes up while the model
becomes useless for anything else. This project's README named that as a known
gap ("you should keep a small general-speech eval in the loop; this project
doesn't"). This closes it.

## Method

40 ordinary English sentences with **no domain vocabulary at all** — no error
codes, no service names, no severity labels — synthesised with the *same*
SpeechT5 voices used for training, so the only variable is the content of the
speech. Then base Whisper and the adapted model transcribe the same audio.

What the numbers mean:

- adapted WER ≈ base WER  -> the adapter is additive; it learned the domain
  without trading away general ability
- adapted WER >> base WER -> forgetting, quantified. The fix is mixing general
  speech into training, lowering the rank, or training fewer epochs

Either way it is a number rather than a hope, which is the whole point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = 16000

# Deliberately ordinary. Everyday vocabulary, varied length and structure, and
# not one token that appears in the domain training set.
SENTENCES = [
    "The weather this morning was cold and unusually bright.",
    "She asked whether the train would arrive before seven.",
    "There is a small bakery on the corner near the library.",
    "He forgot his umbrella again and walked home in the rain.",
    "The meeting has been moved to Thursday afternoon.",
    "My brother is learning to play the piano.",
    "We should probably leave before the traffic gets worse.",
    "The book was longer than I expected but worth finishing.",
    "Can you remind me to water the plants tonight?",
    "The children were playing football in the park.",
    "I would like a cup of tea with a little milk.",
    "That restaurant near the river has excellent fish.",
    "The film starts at half past eight.",
    "She has worked at the hospital for eleven years.",
    "It took nearly an hour to find a parking space.",
    "The garden looks much better since they cut the hedge.",
    "He speaks three languages and is learning a fourth.",
    "We watched the sun set behind the hills.",
    "The letter arrived two weeks after it was posted.",
    "I think the answer is somewhere in the second chapter.",
    "Please close the window before you go to bed.",
    "The bus was late so we walked to the station instead.",
    "Her new apartment has a balcony facing the sea.",
    "They built the bridge more than a century ago.",
    "The coffee machine in the kitchen is broken again.",
    "He wrote down the address on the back of an envelope.",
    "We are planning a short holiday in the spring.",
    "The museum is closed on Mondays during the winter.",
    "She found her keys under a pile of newspapers.",
    "The dog barked at every cyclist that passed.",
    "There was a long queue outside the theatre.",
    "I have not seen that photograph in years.",
    "The recipe calls for butter, flour and two eggs.",
    "His voice was quiet but everyone stopped to listen.",
    "The road was closed because of flooding.",
    "We should buy the tickets before they sell out.",
    "The lecture covered the history of the printing press.",
    "She left a note on the kitchen table.",
    "The storm knocked out power for most of the evening.",
    "It is much easier to walk than to find a taxi.",
]


def synthesise(outdir: Path) -> None:
    """Same TTS and the same voices as the training data -- content is the only variable."""
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    from make_dataset import load_speakers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"synthesising {len(SENTENCES)} general-English clips on {device} ...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()
    speakers, _speaker_ids = load_speakers(device)   # returns (tensor, ids)

    clips = outdir / "eval"
    clips.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i, text in enumerate(SENTENCES):
        inputs = processor(text=text.lower(), return_tensors="pt").to(device)
        spk = speakers[int(rng.integers(len(speakers)))].unsqueeze(0)
        with torch.no_grad():
            wav = tts.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
        name = f"{i:04d}.wav"
        sf.write(clips / name, wav.cpu().numpy().astype("float32"), SR)
        rows.append({"audio": f"eval/{name}", "text": text, "spoken": text.lower()})

    (outdir / "eval.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} clips -> {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data_general"))
    ap.add_argument("--lora", default=str(HERE / "lora-out"))
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--out", default=str(HERE / "eval_general_speech.json"))
    args = ap.parse_args()

    data = Path(args.data)
    if not args.skip_synth or not (data / "eval.jsonl").exists():
        synthesise(data)

    # reuse the project's own scoring so the numbers are comparable with evaluate.py
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from evaluate import report, score, transcribe

    rows = [json.loads(l) for l in (data / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model)

    base = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    hyps, secs = transcribe(base, processor, rows, data, 8)
    base_s = score(hyps, rows)
    report("BASE Whisper (general English)", base_s, secs)

    tuned = PeftModel.from_pretrained(base, args.lora).eval()
    hyps_l, secs_l = transcribe(tuned, processor, rows, data, 8)
    lora_s = score(hyps_l, rows)
    report("LoRA fine-tuned (general English)", lora_s, secs_l)

    d_wer = 100 * (lora_s["wer"] - base_s["wer"])
    print("\n" + "=" * 70)
    print(f"general-English WER   {100*base_s['wer']:5.1f}%  ->  {100*lora_s['wer']:5.1f}%"
          f"   ({d_wer:+.1f} pts)")
    if d_wer > 5:
        print("REGRESSION: the adapter is trading general ability for domain accuracy.")
        print("Fixes: mix general speech into training, lower the rank, or fewer epochs.")
    else:
        print("No meaningful regression -- the adapter is additive on this probe.")
    print("=" * 70)

    Path(args.out).write_text(json.dumps(
        {"n": len(rows), "base": base_s, "lora": lora_s,
         "wer_delta_pts": d_wer}, indent=1), encoding="utf-8")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
