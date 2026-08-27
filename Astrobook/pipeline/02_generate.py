"""
Stage 2: chunks.jsonl -> synthetic instruction pairs, via the Batch API (50% off).

    python pipeline/02_generate.py submit                    # create batch, save id
    python pipeline/02_generate.py collect                   # poll + write pairs.jsonl

    python pipeline/02_generate.py submit --limit 40         # cheap smoke test first
    python pipeline/02_generate.py submit --model claude-sonnet-5

Cost, whole corpus (1,341 chunks x 8 pairs, batched at 50%):
    claude-opus-5    ~$55-65     default; best pair quality
    claude-sonnet-5  ~$20-25     roughly 3x cheaper, noticeably blander answers

Always run --limit 40 first and read the output. The prompt is the single biggest
lever on final model quality, and a bad prompt is much cheaper to find at $2.
"""
import argparse, json, os, sys, time
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CHUNKS, PAIRS, STATE, GEN_SYSTEM, GEN_SCHEMA,
                    PAIRS_PER_CHUNK, user_prompt, chunk_cid as cid)

def load_chunks(limit=None):
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    return rows[:limit] if limit else rows


def submit(model, limit):
    client = anthropic.Anthropic()
    chunks = load_chunks(limit)
    reqs = [
        Request(
            custom_id=cid(c["id"]),
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=16000,
                system=GEN_SYSTEM,
                messages=[{"role": "user",
                           "content": user_prompt(c, PAIRS_PER_CHUNK)}],
                output_config={
                    "format": {"type": "json_schema", "schema": GEN_SCHEMA},
                    "effort": "medium",
                },
            ),
        )
        for c in chunks
    ]
    batch = client.messages.batches.create(requests=reqs)
    json.dump({"batch_id": batch.id, "model": model, "n": len(reqs)},
              open(STATE, "w"), indent=2)
    print(f"submitted {len(reqs)} requests on {model} -> {batch.id}")
    print("most batches finish inside an hour; 24h is the hard ceiling")
    print("then: python pipeline/02_generate.py collect")


def collect():
    client = anthropic.Anthropic()
    st = json.load(open(STATE))
    bid = st["batch_id"]

    while True:
        b = client.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        print(f"  {b.processing_status}: {b.request_counts.processing} in flight, "
              f"{b.request_counts.succeeded} done")
        time.sleep(60)
    print(f"ended: {b.request_counts.succeeded} ok, {b.request_counts.errored} errored")

    # custom_id -> chunk, so every pair keeps a citable provenance trail
    src = {cid(c["id"]): c for c in load_chunks()}

    kept, ungrounded, failed = [], 0, 0
    for r in client.messages.batches.results(bid):
        if r.result.type != "succeeded":
            failed += 1
            continue
        msg = r.result.message
        if msg.stop_reason == "refusal":          # 200 OK, no usable content
            failed += 1
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            pairs = json.loads(text)["pairs"]
        except (json.JSONDecodeError, KeyError, TypeError):
            failed += 1
            continue
        c = src.get(r.custom_id, {})
        for p in pairs:
            if not p.get("grounded"):
                ungrounded += 1
                continue
            kept.append({
                "chunk_id": c.get("id"), "source": c.get("source"),
                "title": c.get("title"), "type": p["type"],
                "question": p["question"].strip(), "answer": p["answer"].strip(),
            })

    # Near-duplicate questions cluster hard across chunks of the same book
    # ("What does Saravali say about Mars?" x40). Dedupe on normalised text.
    seen, uniq, dupes = set(), [], 0
    for p in kept:
        k = " ".join(p["question"].lower().split())
        if k in seen:
            dupes += 1
            continue
        seen.add(k)
        uniq.append(p)

    with open(PAIRS, "w", encoding="utf-8") as f:
        for p in uniq:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"kept {len(uniq):,} | ungrounded {ungrounded:,} | dupes {dupes:,} | "
          f"failed reqs {failed}")
    print(f"wrote {PAIRS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect"])
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    submit(a.model, a.limit) if a.cmd == "submit" else collect()
