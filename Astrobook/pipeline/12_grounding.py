"""
Is the model actually reading YOUR documents, or answering from pretraining?

    python pipeline/12_grounding.py -n 6

Two tests.

TEST A -- NEEDLE (decisive).
Insert a fact that cannot exist anywhere in Jyotisha literature or in the
model's pretraining -- an invented planet name, an absurd number -- into the
retrieved passage, then ask about it. If the model reports the needle, it is
demonstrably reading the supplied text. If it answers from priors instead, or
says nothing, retrieval is decorative. There is no ambiguity in this test: no
pretrained model has ever seen "Zorvaxa" ruling a bhava.

TEST B -- LEXICAL GROUNDING.
For real questions, measure what share of the answer's distinctive content words
actually occur in the retrieved passages. Compare the RAG answer against the
same model answering the same question with NO passages. If retrieval is doing
work, the RAG answer should overlap the sources far more.

Neither test needs a judge model or an API key.
"""
import argparse, importlib.util, json, os, re, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import SYSTEM_PROMPT, split_path, build_context

ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models", "astro-4b")

RAG_SYSTEM = SYSTEM_PROMPT + (
    " You are given excerpts from the source texts. Answer ONLY from those "
    "excerpts, and name the text you are drawing on. If the excerpts do not "
    "contain the answer, say so plainly rather than supplying it from memory.")

# Words that carry no evidence of grounding.
STOP = set("""a an the and or but if then than that this these those of in on at to for
with by from as is are was were be been being it its his her their there here what which
who whom when where how why will would shall should can could may might must not no nor
one two three four five six seven eight nine ten native person text states holds says
according also such other more most some any each both very much many"""
           .split())

# (label, planted sentence, question, PROOF TOKENS)
# Proof tokens must NOT appear in the question -- otherwise the model can echo
# them without reading anything. An earlier version checked for the invented
# NAME, which the question itself contained; the control correctly flagged that
# as an invalid test.
NEEDLES = [
    ("Zorvaxa",
     "The graha Zorvaxa, when placed in the 6th bhava, grants the native "
     "mastery over glassblowing.",
     "What does the graha Zorvaxa grant, and where must it be placed?",
     ["glassblowing"]),
    ("Quenlith",
     "Sage Quenlith declares that the 11th bhava governs the keeping of blue "
     "parrots above all else.",
     "According to Sage Quenlith, what specifically does the 11th bhava govern?",
     ["parrot"]),
    ("847",
     "The Chandra Kalika yoga is formed only when exactly 847 degrees separate "
     "the lagna from Mangal.",
     "How many degrees must separate the lagna from Mangal for Chandra Kalika "
     "yoga?",
     ["847"]),
]


def words(text):
    return {w for w in re.findall(r"[a-zāīūṛṅñṭḍṇśṣḥṃ]{4,}", text.lower())
            if w not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=200)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "idx", os.path.join(HERE, "09_index.py"))
    idx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(idx)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"": 0})
    model.eval()

    def gen(q, ctx=None):
        enc = tok.apply_chat_template(
            [{"role": "system", "content": RAG_SYSTEM if ctx else SYSTEM_PROMPT},
             {"role": "user",
              "content": f"<excerpts>\n{ctx}\n</excerpts>\n\n{q}" if ctx else q}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        plen = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][plen:], skip_special_tokens=True).strip()

    def context(hits, needle=None):
        parts = build_context(hits).split("\n\n") if hits else []
        if needle:
            # bury it in the middle, not at an edge where position alone helps
            parts.insert(len(parts) // 2, f"[Sanketanidhi]\n{needle}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------- TEST A
    print("=" * 78)
    print("TEST A -- NEEDLE: can the model report a fact that exists ONLY")
    print("                  in the passage we supplied?")
    print("=" * 78)
    hits = idx.search("bhava effects of grahas", k=a.k)
    found = 0
    for token, needle, question, proof in NEEDLES:
        ans = gen(question, context(hits, needle))
        hit = any(p.lower() in ans.lower() for p in proof)
        found += hit
        print(f"\n  needle '{token}'  -> {'REPEATED (grounded)' if hit else 'MISSED'}")
        print(f"    Q: {question}")
        print(f"    A: {' '.join(ans.split())[:260]}")
    print(f"\n  {found}/{len(NEEDLES)} needles reported back")

    # control: same questions, NO passage. Should be unable to answer.
    print("\n  control -- same questions with NO excerpts supplied:")
    leaked = 0
    for token, _, question, proof in NEEDLES:
        ans = gen(question)
        if any(p.lower() in ans.lower() for p in proof):
            leaked += 1
            print(f"    !! {proof} appeared WITHOUT the passage -- invalid test")
    if not leaked:
        print("    no proof token appears without the passage -- the facts are "
              "genuinely unknown to the model, so Test A is valid")

    # ------------------------------------------------------------- TEST B
    print("\n" + "=" * 78)
    print("TEST B -- LEXICAL GROUNDING on real held-out questions")
    print("=" * 78)
    rows = [json.loads(l) for l in open(split_path("test"), encoding="utf-8")]

    # One question per book caps the sample at the number of held-out books --
    # so `-n 20` silently returned 4 once 03_split.py was corrected and test
    # dropped to four books, and the lexical-lift number was being read off
    # four examples. Spread evenly instead: still no book dominates, but -n is
    # actually reachable. (Same defect 08_validate.py had with `gen_n // 5`.)
    n_books = len({r["meta"]["source"] for r in rows}) or 1
    per_book = max(1, -(-a.n // n_books))

    seen, picked = {}, []
    for r in rows:
        s = r["meta"]["source"]
        if seen.get(s, 0) >= per_book:
            continue
        seen[s] = seen.get(s, 0) + 1
        picked.append(r)
        if len(picked) >= a.n:
            break

    rag_scores, norag_scores, right_book = [], [], 0
    for r in picked:
        q = r["messages"][1]["content"]
        hits = idx.search(q, k=a.k)
        ctx = context(hits)
        right_book += any(h["source"] == r["meta"]["source"] for h in hits)
        src_words = words(ctx)
        q_words = words(q)

        for ans, bucket in ((gen(q, ctx), rag_scores), (gen(q), norag_scores)):
            aw = words(ans) - q_words          # ignore words echoed from the question
            bucket.append(len(aw & src_words) / max(len(aw), 1))

    def mean(x):
        return sum(x) / max(len(x), 1)

    print(f"\n  questions            {len(picked)}")
    print(f"  retrieved right book {right_book}/{len(picked)}")
    print(f"\n  share of answer's content words found in the retrieved passages:")
    print(f"    WITH retrieval     {mean(rag_scores):.1%}")
    print(f"    WITHOUT retrieval  {mean(norag_scores):.1%}")
    lift = mean(rag_scores) - mean(norag_scores)
    print(f"    lift               {lift:+.1%}")

    print("\n" + "=" * 78)
    verdict = (found >= 2 and not leaked and lift > 0.05)
    print("VERDICT: the model IS reading your documents." if verdict
          else "VERDICT: grounding is WEAK -- inspect the numbers above.")
    print("=" * 78)

    json.dump({"needles_found": found, "needles_total": len(NEEDLES),
               "control_leaks": leaked, "rag_overlap": mean(rag_scores),
               "norag_overlap": mean(norag_scores), "lift": lift,
               "right_book": right_book, "n": len(picked)},
              open(os.path.join(ROOT, "build", "grounding.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
