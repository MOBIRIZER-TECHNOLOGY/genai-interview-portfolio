"""
Validate the adapter against the UNTUNED BASE on held-out books.

    python pipeline/08_validate.py --adapter runs/qlora-nf4 --loss-n 300 --gen-n 20

Two measurements, both without a judge model or an API key:

1. HELD-OUT LOSS, base vs adapter, on identical examples with identical masking
   (assistant tokens only). This is the number missing from the earlier run:
   eval_loss compared the two ARMS to each other, never to doing nothing.

2. MECHANICAL STYLE METRICS over generated answers -- source attribution,
   doctrinal framing, personal prediction, preamble, markdown. These are exactly
   the behaviours the training data was built to install, and all are countable
   with regexes rather than opinions.

Uses peft's disable_adapter() so base and tuned are the SAME loaded weights.
Questions come from test.jsonl -- books held out of training entirely.
"""
import argparse, json, os, re, sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import SYSTEM_PROMPT, split_path

STOPWORDS = {"the", "and", "of", "vol", "pdf", "01", "02", "1", "2"}


def source_keys(source):
    """Distinctive words from a filename, for attribution matching."""
    stem = source[:-4] if source.endswith(".pdf") else source
    return [w for w in stem.split("-") if len(w) > 4 and w not in STOPWORDS]


# --- style probes -----------------------------------------------------------
RX_DOCTRINE = re.compile(
    r"\b(states?|holds?|describes?|asserts?|says?|indicates?|prescribes?|"
    r"according to|per)\b", re.I)
RX_PERSONAL = re.compile(
    r"\b(you will|you are likely|your (?:chart|life|horoscope|career|marriage)|"
    r"will bring you|you may experience)\b", re.I)
RX_HEDGE = re.compile(
    r"\b(it is important to (?:note|clarify)|however, it|that said|"
    r"let(?:'|’)s clarify|it should be noted|i (?:should|must) (?:note|clarify))\b",
    re.I)
RX_MARKDOWN = re.compile(r"(^|\n)\s*(#{1,4}\s|\*\*[A-Z])")
RX_SANSKRIT = re.compile(
    r"\b(bhava|lagna|lagn|rasi|rāśi|graha|dasha|dasa|antar|yoga|yog|"
    r"kendra|trikona|navamsa|karaka|mangal|shani|sani|budh|guru|shukra|sukr)\w*\b",
    re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/qlora-nf4")
    ap.add_argument("--loss-n", type=int, default=300)
    ap.add_argument("--gen-n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=200)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.adapter, "run_metrics.json")))
    base_id = cfg["model"]
    print(f"base    {base_id}")
    print(f"adapter {a.adapter}  ({cfg['mode']}, {cfg['global_steps']} steps)\n")

    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(split_path("test"), encoding="utf-8")]

    # ---------------------------------------------------- 1. held-out loss
    def example_loss(msgs):
        """NLL on assistant tokens only -- mirrors assistant_only_loss."""
        # return_dict=True always, then index. transformers 5.x returns a
        # BatchEncoding (a UserDict) which is NOT an instance of dict, so an
        # isinstance(x, dict) guard silently fails to unwrap it.
        full = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=True,
                                         add_generation_prompt=True,
                                         return_tensors="pt",
                                         return_dict=True)["input_ids"]
        full = full.to("cuda")
        plen = prompt.shape[-1]
        if full.shape[-1] <= plen or full.shape[-1] > 2048:
            return None
        labels = full.clone()
        labels[:, :plen] = -100
        with torch.no_grad():
            return model(input_ids=full, labels=labels).loss.item()

    def sweep(label):
        tot, n = 0.0, 0
        for r in rows[:a.loss_n]:
            l = example_loss(r["messages"])
            if l is not None and l == l:
                tot += l
                n += 1
        return tot / max(n, 1), n

    print("computing held-out loss on identical examples...")
    with model.disable_adapter():
        base_loss, n = sweep("base")
    tuned_loss, _ = sweep("tuned")
    print(f"  n = {n} held-out examples\n")

    # ------------------------------------------------- 2. generated answers
    def gen(question):
        enc = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": question}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        plen = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][plen:], skip_special_tokens=True).strip()

    # Spread the sample evenly across whatever held-out books actually exist.
    # The cap used to be `gen_n // 5`, hardcoding the five books the split
    # produced at the time. When 03_split.py was corrected to give val and test
    # different books, test dropped to four -- and this silently returned 16
    # answers for a requested 20, shrinking the evidence behind every style
    # metric without saying so. Derive the cap from the data, not from memory.
    n_books = len({r["meta"]["source"] for r in rows}) or 1
    per_book = max(1, -(-a.gen_n // n_books))       # ceil, so gen_n is reachable

    picked, seen = [], {}
    for r in rows:
        s = r["meta"]["source"]
        if seen.get(s, 0) >= per_book:
            continue
        seen[s] = seen.get(s, 0) + 1
        picked.append(r)
        if len(picked) >= a.gen_n:
            break

    def score(answers, sources):
        m = {"attributes": 0, "doctrinal": 0, "personal": 0, "hedges": 0,
             "markdown": 0, "sanskrit": 0, "words": 0}
        for ans, src in zip(answers, sources):
            keys = source_keys(src)
            m["attributes"] += any(k.lower() in ans.lower() for k in keys)
            m["doctrinal"] += bool(RX_DOCTRINE.search(ans))
            m["personal"] += bool(RX_PERSONAL.search(ans))
            m["hedges"] += bool(RX_HEDGE.search(ans))
            m["markdown"] += bool(RX_MARKDOWN.search(ans))
            m["sanskrit"] += bool(RX_SANSKRIT.search(ans))
            m["words"] += len(ans.split())
        n = max(len(answers), 1)
        return {k: (v / n if k != "words" else v / n) for k, v in m.items()}

    print(f"generating {len(picked)} answers x2 ...")
    srcs = [r["meta"]["source"] for r in picked]
    qs = [r["messages"][1]["content"] for r in picked]
    with model.disable_adapter():
        base_ans = [gen(q) for q in qs]
    tuned_ans = [gen(q) for q in qs]
    bm, tm = score(base_ans, srcs), score(tuned_ans, srcs)

    # ------------------------------------------------------------- report
    W = 34
    print("\n" + "=" * 68)
    print(f"{'HELD-OUT LOSS (lower is better)':<{W}}{'BASE':>14}{'TUNED':>14}")
    print("=" * 68)
    print(f"{'assistant-token NLL':<{W}}{base_loss:>14.4f}{tuned_loss:>14.4f}")
    delta = base_loss - tuned_loss
    print(f"{'improvement':<{W}}{'':>14}{delta:>+14.4f}")
    print(f"{'perplexity':<{W}}{pow(2.718281828, base_loss):>14.2f}"
          f"{pow(2.718281828, tuned_loss):>14.2f}")

    print("\n" + "=" * 68)
    print(f"{'STYLE (fraction of answers)':<{W}}{'BASE':>14}{'TUNED':>14}")
    print("=" * 68)
    for key, label, good_up in [
            ("attributes", "names the source text", True),
            ("doctrinal", "doctrinal framing verb", True),
            ("sanskrit", "uses Sanskrit terminology", True),
            ("personal", "personal prediction (bad)", False),
            ("hedges", "hedging preamble (bad)", False),
            ("markdown", "markdown headers (bad)", False)]:
        arrow = ""
        if (tm[key] > bm[key]) == good_up and abs(tm[key] - bm[key]) > 0.05:
            arrow = "  better"
        elif (tm[key] < bm[key]) == good_up and abs(tm[key] - bm[key]) > 0.05:
            arrow = "  worse"
        print(f"{label:<{W}}{bm[key]:>13.0%}{tm[key]:>13.0%}{arrow}")
    print(f"{'mean answer length (words)':<{W}}{bm['words']:>13.0f}{tm['words']:>13.0f}")

    json.dump({"base_loss": base_loss, "tuned_loss": tuned_loss, "n_loss": n,
               "base_style": bm, "tuned_style": tm, "n_gen": len(picked)},
              open(os.path.join(a.adapter, "validation.json"), "w"), indent=2)
    print(f"\nwrote {a.adapter}/validation.json")


if __name__ == "__main__":
    main()
