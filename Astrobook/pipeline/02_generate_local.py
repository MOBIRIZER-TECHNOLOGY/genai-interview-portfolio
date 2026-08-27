"""
Stage 2 (local): chunks.jsonl -> instruction pairs, via Ollama on your own GPU.

Free alternative to the Claude Batch path in 02_generate.py. Uses the SAME
prompt and schema (both live in config.py) so the two datasets stay comparable.

    python pipeline/02_generate_local.py --limit 5 --model qwen2.5:7b   # smoke
    python pipeline/02_generate_local.py --model qwen3:14b              # full run

RESUMABLE. Appends to build/pairs_raw.jsonl and skips chunks already done, so
you can Ctrl-C and restart without losing or duplicating work. Run
`--finalize` at the end to dedupe and write build/pairs.jsonl.
"""
import argparse, json, os, queue, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import (BUILD, CHUNKS, PAIRS, GEN_SYSTEM, GEN_SCHEMA,
                    PAIRS_PER_CHUNK, user_prompt)
from qc import sanitize_question, acceptable

RAW = os.path.join(BUILD, "pairs_raw.jsonl")
API = "http://localhost:11434/api/chat"

_write_lock = threading.Lock()
_stats = {"ok": 0, "fail": 0, "pairs": 0, "rejected": 0}


def call_ollama(model, chunk, n, timeout, num_ctx):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": GEN_SYSTEM},
                     {"role": "user", "content": user_prompt(chunk, n)}],
        "stream": False,
        "format": GEN_SCHEMA,          # Ollama enforces the JSON schema
        "think": False,                # qwen3 reasons by default; off = 3-4x faster
        # num_predict caps a runaway chunk. Measured: qwen3:14b spends ~1,050
        # tokens on 8 pairs, so 2,600 is generous. Without a cap, a model that
        # fails to close the JSON generates until the context fills and the
        # worker stalls for minutes producing nothing (qwen2.5:7b does exactly
        # this -- it is why that model returns zero usable pairs).
        "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": num_ctx,
                    "num_predict": 2600},
    }).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def worker(q, model, n, timeout, num_ctx, fh, t0, total):
    while True:
        try:
            chunk = q.get_nowait()
        except queue.Empty:
            return
        try:
            raw = call_ollama(model, chunk, n, timeout, num_ctx)
            pairs = json.loads(raw).get("pairs", [])
        except (urllib.error.URLError, json.JSONDecodeError, KeyError,
                TimeoutError, OSError) as e:
            with _write_lock:
                _stats["fail"] += 1
                print(f"  ! {chunk['id']}: {type(e).__name__}")
            q.task_done()
            continue

        kept = []
        for p in pairs:
            if not isinstance(p, dict) or not p.get("grounded"):
                _stats["rejected"] += 1
                continue
            aa = str(p.get("answer", "")).strip()
            qq = sanitize_question(str(p.get("question", "")).strip())
            ok, _why = acceptable(qq, aa)
            if not ok:
                _stats["rejected"] += 1
                continue
            kept.append({"chunk_id": chunk["id"], "source": chunk["source"],
                         "title": chunk["title"],
                         "type": p.get("type", "definitional"),
                         "question": qq, "answer": aa})

        with _write_lock:
            for k in kept:
                fh.write(json.dumps(k, ensure_ascii=False) + "\n")
            fh.flush()
            _stats["ok"] += 1
            _stats["pairs"] += len(kept)
            done = _stats["ok"] + _stats["fail"]
            if done % 10 == 0 or done == total:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (total - done) / max(rate, 1e-9) / 60
                print(f"  {done:>5}/{total}  {_stats['pairs']:>6} pairs  "
                      f"{rate*60:5.1f} chunk/min  ETA {eta:5.1f} min")
        q.task_done()


def finalize():
    """Dedupe near-identical questions and write the canonical pairs.jsonl."""
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    seen, uniq, dupes = set(), [], 0
    for p in rows:
        k = " ".join(p["question"].lower().split())
        if k in seen:
            dupes += 1
            continue
        seen.add(k)
        uniq.append(p)
    with open(PAIRS, "w", encoding="utf-8") as f:
        for p in uniq:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    import collections
    print(f"raw {len(rows):,} -> kept {len(uniq):,} (dropped {dupes:,} dupes)")
    print(f"sources: {len(set(p['source'] for p in uniq))}")
    for t, c in collections.Counter(p["type"] for p in uniq).most_common():
        print(f"  {t:<18}{c:>7,}  {100*c/max(1,len(uniq)):>5.1f}%")
    print(f"wrote {PAIRS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pairs", type=int, default=PAIRS_PER_CHUNK)
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel requests; needs OLLAMA_NUM_PARALLEL >= this")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--num-ctx", type=int, default=6144)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore prior progress")
    a = ap.parse_args()

    if a.finalize:
        return finalize()

    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    if a.limit:
        chunks = chunks[:a.limit]

    done = set()
    if os.path.exists(RAW) and not a.restart:
        for l in open(RAW, encoding="utf-8"):
            try:
                done.add(json.loads(l)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [c for c in chunks if c["id"] not in done]
    print(f"model {a.model} | {len(chunks):,} chunks | {len(done):,} already done "
          f"| {len(todo):,} to do | {a.workers} workers")
    if not todo:
        print("nothing to do -- run --finalize")
        return

    q = queue.Queue()
    for c in todo:
        q.put(c)
    t0 = time.time()
    with open(RAW, "a" if not a.restart else "w", encoding="utf-8") as fh:
        threads = [threading.Thread(target=worker,
                                    args=(q, a.model, a.pairs, a.timeout,
                                          a.num_ctx, fh, t0, len(todo)),
                                    daemon=True)
                   for _ in range(a.workers)]
        [t.start() for t in threads]
        [t.join() for t in threads]

    el = (time.time() - t0) / 60
    print(f"\ndone in {el:.1f} min | ok {_stats['ok']} | failed {_stats['fail']} "
          f"| pairs {_stats['pairs']:,} | rejected {_stats['rejected']:,}")
    print("next: python pipeline/02_generate_local.py --finalize")


if __name__ == "__main__":
    main()
