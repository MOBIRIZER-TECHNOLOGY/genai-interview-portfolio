"""
Talk to your tuned model.

    python chat.py                 # chat with the merged model
    python chat.py --rag           # same, but retrieve source passages first
    python chat.py --ask "..."     # one question, then exit

Loads models/astro-4b -- the standalone merged model. Type 'exit' to quit.

--rag is the recommended mode. The adapter supplies the voice; retrieval
supplies the facts. Without it the model answers in the right register but will
invent specifics, which is the documented limitation (see IMPLEMENTATION_PLAN
section 15).
"""
import argparse, os, sys, textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from config import SYSTEM_PROMPT, build_context

DEFAULT_MODEL = os.path.join(ROOT, "models", "astro-4b")

RAG_SYSTEM = SYSTEM_PROMPT + (
    " You are given excerpts from the source texts. Answer ONLY from those "
    "excerpts, and name the text you are drawing on. If the excerpts do not "
    "contain the answer, say so plainly rather than supplying it from memory.")


def wrap(s, w=90):
    return "\n".join(textwrap.fill(p, w) for p in s.split("\n") if p.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rag", action="store_true", help="retrieve before answering")
    ap.add_argument("--ask", help="single question, non-interactive")
    ap.add_argument("-k", type=int, default=3, help="passages to retrieve")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--full", action="store_true",
                    help="load in bf16 (8 GB) instead of 4-bit (~3 GB)")
    a = ap.parse_args()

    if not os.path.isdir(a.model):
        sys.exit(f"model not found: {a.model}\n"
                 "run: python pipeline/11_merge.py --adapter runs/qlora-2ep "
                 "--out models/astro-4b")

    search = None
    if a.rag:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "idx", os.path.join(ROOT, "pipeline", "09_index.py"))
        idx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(idx)
        search = idx.search

    print(f"loading {os.path.basename(a.model)} "
          f"({'bf16' if a.full else '4-bit'}){' + retrieval' if a.rag else ''} ...")
    quant = None if a.full else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, quantization_config=quant,
        device_map={"": 0} if torch.cuda.is_available() else {"": "cpu"})
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def answer(q):
        ctx = None
        if search:
            hits = search(q, k=a.k)
            ctx = build_context(hits)
            print("\n  sources: " + ", ".join(sorted({h["source"] for h in hits})))
        enc = tok.apply_chat_template(
            [{"role": "system", "content": RAG_SYSTEM if ctx else SYSTEM_PROMPT},
             {"role": "user",
              "content": f"<excerpts>\n{ctx}\n</excerpts>\n\n{q}" if ctx else q}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        plen = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][plen:], skip_special_tokens=True).strip()

    if a.ask:
        print("\n" + wrap(answer(a.ask)))
        return

    print("\nready. ask a Jyotisha question, or 'exit' to quit.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        print()
        print(wrap(answer(q)))
        print()


if __name__ == "__main__":
    main()
