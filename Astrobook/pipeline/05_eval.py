"""
Stage 5: does the adapter actually earn its keep?

There is no ground truth for "correct Jyotisha", so do NOT grade correctness.
Grade FAITHFULNESS: every test pair carries the chunk_id it was generated from,
so the source passage is the reference the answer must be supported by.

    # on the GPU box, after training:
    python pipeline/05_eval.py generate --adapter astro-lora --n 150
    # anywhere with an API key:
    python pipeline/05_eval.py judge

`generate` answers each held-out question TWICE -- once with the adapter, once
with the identical base model -- so the judge sees a real A/B, not a vibe check.
"""
import argparse, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CHUNKS, BUILD, SYSTEM_PROMPT, split_path

ANSWERS = os.path.join(BUILD, "eval_answers.jsonl")
SCORES = os.path.join(BUILD, "eval_scores.json")
JSTATE = os.path.join(BUILD, "judge_state.json")


# ---------------------------------------------------------------- generate
def generate(a):
    from unsloth import FastLanguageModel
    import torch

    rows = [json.loads(l) for l in open(split_path("test"), encoding="utf-8")]
    random.Random(0).shuffle(rows)
    rows = rows[:a.n]

    def answers_from(path, label):
        model, tok = FastLanguageModel.from_pretrained(
            model_name=path, max_seq_length=1024, dtype=None, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
        out = []
        for i, r in enumerate(rows):
            q = r["messages"][1]["content"]
            # transformers 5.x returns a BatchEncoding dict, not a tensor.
            enc = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": q}],
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            plen = enc["input_ids"].shape[-1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=400, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            out.append(tok.decode(gen[0][plen:],
                                  skip_special_tokens=True).strip())
            if (i + 1) % 25 == 0:
                print(f"  {label:<5} {i + 1}/{len(rows)}")
        del model
        torch.cuda.empty_cache()
        return out

    print(f"generating {len(rows)} x2 answers")
    tuned = answers_from(a.adapter, "tuned")
    base = answers_from(a.base, "base")

    with open(ANSWERS, "w", encoding="utf-8") as f:
        for r, t, b in zip(rows, tuned, base):
            f.write(json.dumps({
                "chunk_id": r["meta"]["chunk_id"], "source": r["meta"]["source"],
                "type": r["meta"]["type"], "question": r["messages"][1]["content"],
                "reference": r["messages"][2]["content"],
                "tuned": t, "base": b,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {ANSWERS}")


# ------------------------------------------------------------------- judge
JUDGE_SYSTEM = """You grade two candidate answers against the SOURCE PASSAGE they \
should be drawn from. You are grading FAITHFULNESS, not whether the astrology is \
true.

For each answer score:
  support   0-3  3 = every claim traceable to the passage. 2 = mostly, one loose
                 claim. 1 = substantially unsupported. 0 = contradicts the passage.
  fabricated true if it states a specific rule, verse number, or attribution the
                 passage does not contain. Confident invention is the failure mode
                 that matters most here -- flag it even when it sounds right.
  citation  0-2  2 = names the source text correctly. 1 = vague ("the classics
                 say"). 0 = names the wrong text, or none.
  style     0-2  2 = explains doctrine as doctrine. 0 = tells the reader what will
                 happen to them personally.

Then pick the better answer overall, or "tie". You do not know which system \
produced which; judge only what is written."""

SCORE_OBJ = {
    "type": "object",
    "properties": {
        "support": {"type": "integer", "minimum": 0, "maximum": 3},
        "fabricated": {"type": "boolean"},
        "citation": {"type": "integer", "minimum": 0, "maximum": 2},
        "style": {"type": "integer", "minimum": 0, "maximum": 2},
    },
    "required": ["support", "fabricated", "citation", "style"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "a": SCORE_OBJ,
        "b": SCORE_OBJ,
        "winner": {"type": "string", "enum": ["a", "b", "tie"]},
        "note": {"type": "string"},
    },
    "required": ["a", "b", "winner", "note"],
    "additionalProperties": False,
}


def judge(a):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    chunks = {}
    for line in open(CHUNKS, encoding="utf-8"):
        c = json.loads(line)
        chunks[c["id"]] = c["text"]
    rows = [json.loads(l) for l in open(ANSWERS, encoding="utf-8")]

    rng = random.Random(0)
    reqs, flip = [], {}
    for i, r in enumerate(rows):
        # Randomise slot assignment -- LLM judges have a real position bias.
        swap = rng.random() < 0.5
        flip[str(i)] = swap
        first, second = ((r["base"], r["tuned"]) if swap
                         else (r["tuned"], r["base"]))
        passage = chunks.get(r["chunk_id"], "")
        reqs.append(Request(
            custom_id=f"j{i:05d}",
            params=MessageCreateParamsNonStreaming(
                model=a.model, max_tokens=4000, system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content":
                           f"<passage>\n{passage}\n</passage>\n\n"
                           f"<question>{r['question']}</question>\n\n"
                           f"<answer_a>\n{first}\n</answer_a>\n\n"
                           f"<answer_b>\n{second}\n</answer_b>"}],
                output_config={"format": {"type": "json_schema",
                                          "schema": JUDGE_SCHEMA},
                               "effort": "medium"},
            )))

    batch = client.messages.batches.create(requests=reqs)
    json.dump({"batch_id": batch.id, "flip": flip}, open(JSTATE, "w"))
    print(f"judging {len(reqs)} pairs -> {batch.id}")

    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"  {b.processing_status}: {b.request_counts.processing} in flight")
        time.sleep(60)

    agg = {"tuned": [], "base": []}
    wins = {"tuned": 0, "base": 0, "tie": 0}
    for res in client.messages.batches.results(batch.id):
        if res.result.type != "succeeded":
            continue
        msg = res.result.message
        if msg.stop_reason == "refusal":
            continue
        try:
            text = next(x.text for x in msg.content if x.type == "text")
            d = json.loads(text)
        except (json.JSONDecodeError, StopIteration):
            continue
        idx = str(int(res.custom_id[1:]))
        swap = flip[idx]
        slot_a = "base" if swap else "tuned"
        slot_b = "tuned" if swap else "base"
        agg[slot_a].append(d["a"])
        agg[slot_b].append(d["b"])
        w = d["winner"]
        wins["tie" if w == "tie" else (slot_a if w == "a" else slot_b)] += 1

    def mean(rs, k):
        return sum(r[k] for r in rs) / max(1, len(rs))

    print(f"\n{'':<10}{'support/3':>11}{'cite/2':>9}{'style/2':>9}{'fabricated':>12}")
    for k in ("tuned", "base"):
        rs = agg[k]
        print(f"{k:<10}{mean(rs, 'support'):>11.2f}{mean(rs, 'citation'):>9.2f}"
              f"{mean(rs, 'style'):>9.2f}{100 * mean(rs, 'fabricated'):>11.1f}%")
    n = sum(wins.values())
    print(f"\nhead-to-head (n={n}): tuned {wins['tuned']} | base {wins['base']} "
          f"| tie {wins['tie']}")
    if wins["tuned"] <= wins["base"]:
        print("\n!! the adapter is not beating the base. Before retraining, check "
              "that stage-2 pairs are actually grounded -- bad data, not bad "
              "hyperparameters, is the usual cause.")

    json.dump({"agg": agg, "wins": wins}, open(SCORES, "w"), indent=2)
    print(f"wrote {SCORES}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--adapter", default="astro-lora")
    g.add_argument("--base",
                   default="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit")
    g.add_argument("--n", type=int, default=150)
    j = sub.add_parser("judge")
    j.add_argument("--model", default="claude-opus-5")
    args = p.parse_args()
    generate(args) if args.cmd == "generate" else judge(args)
