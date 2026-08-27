"""
RAG: retrieval supplies the facts, the adapter supplies the voice.

    python pipeline/10_rag.py --ask "What does Phaladeepika say about the birth of a son?"
    python pipeline/10_rag.py --compare -n 5      # 4-way ablation on held-out books

THE POINT. §15 showed the adapter learned style comprehensively and facts not at
all -- it answered "the 2nd, 5th, 7th, 8th, 10th, 11th lords" where Phaladeepika
says "the ascendant lord, 7th lord, 5th lord, Jupiter...". Fluent, correctly
shaped, wrong.

Retrieval fixes precisely that failure and nothing else. The adapter still
decides HOW to answer; the retrieved passage decides WHAT is true. Neither alone
is sufficient:

    base alone        rambling, hedged, invents sources
    base + RAG        correct facts, wrong register (markdown, preamble)
    adapter alone     right register, invented facts        <-- §15
    adapter + RAG     right register, grounded facts        <-- the goal

`--compare` runs all four on the same questions so the contribution of each part
is visible rather than asserted.
"""
import argparse, json, os, re, sys, textwrap

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import SYSTEM_PROMPT, split_path, build_context

RAG_SYSTEM = SYSTEM_PROMPT + (
    " You are given excerpts from the source texts. Answer ONLY from those "
    "excerpts, and name the text you are drawing on. If the excerpts do not "
    "contain the answer, say so plainly rather than supplying it from memory.")


def wrap(s, w=88, indent="    "):
    return "\n".join(textwrap.fill(p, w, initial_indent=indent,
                                   subsequent_indent=indent)
                     for p in s.split("\n") if p.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/qlora-nf4")
    ap.add_argument("--ask")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("-k", type=int, default=3, help="passages to retrieve")
    ap.add_argument("--max-new", type=int, default=220)
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "idx", os.path.join(HERE, "09_index.py"))
    idx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(idx)

    cfg = json.load(open(os.path.join(a.adapter, "run_metrics.json")))
    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    def gen(question, context=None):
        sys_p = RAG_SYSTEM if context else SYSTEM_PROMPT
        user = (f"<excerpts>\n{context}\n</excerpts>\n\n{question}"
                if context else question)
        enc = tok.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        plen = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][plen:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------ single ask
    if a.ask:
        hits = idx.search(a.ask, k=a.k)
        print(f"\nretrieved {len(hits)} passages:")
        for h in hits:
            print(f"  - {h['source']}  (rrf {h['rrf']:.4f})")
        ans = gen(a.ask, build_context(hits))
        print(f"\nQ: {a.ask}\n")
        print(wrap(ans))
        return

    # ------------------------------------------------------------- ablation
    rows = [json.loads(l) for l in open(split_path("test"), encoding="utf-8")]
    seen, picked = set(), []
    for r in rows:
        s = r["meta"]["source"]
        if s in seen:
            continue
        seen.add(s)
        picked.append(r)
        if len(picked) >= a.n:
            break

    for i, r in enumerate(picked, 1):
        q = r["messages"][1]["content"]
        ref = r["messages"][2]["content"]
        hits = idx.search(q, k=a.k)
        ctx = build_context(hits)
        got_right_book = any(h["source"] == r["meta"]["source"] for h in hits)

        print("=" * 92)
        print(f"Q{i}  [{r['meta']['source']}]")
        print(f"    {q}")
        print(f"    retrieval found the source book: "
              f"{'YES' if got_right_book else 'NO'}  "
              f"({', '.join(h['source'][:22] for h in hits)})")
        print("=" * 92)

        with model.disable_adapter():
            base_only = gen(q)
            base_rag = gen(q, ctx)
        tuned_only = gen(q)
        tuned_rag = gen(q, ctx)

        for label, text in [("BASE alone", base_only),
                            ("BASE + RAG", base_rag),
                            ("ADAPTER alone", tuned_only),
                            ("ADAPTER + RAG", tuned_rag),
                            ("REFERENCE (ground truth)", ref)]:
            print(f"\n  --- {label} ---")
            print(wrap(text[:700]))
        print()


if __name__ == "__main__":
    main()
