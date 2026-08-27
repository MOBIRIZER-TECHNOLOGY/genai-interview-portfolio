"""
HTTP service for the Astro adapter -- self-hosted, bound to a static IP.

    set ASTRO_API_KEY=<a long random string>
    python serve.py                       # 0.0.0.0:11435

    curl -H "Authorization: Bearer $ASTRO_API_KEY" ^
         -H "Content-Type: application/json" ^
         -d "{\"question\":\"What does Phaladeepika say about the 5th house?\"}" ^
         http://<your-ip>:11435/answer

Endpoints
    GET  /                 the demo page (demo.html), served same-origin
    GET  /healthz          model + index status. No auth, no model work.
    POST /search           retrieval only -- cheap, no GPU.
    POST /answer           retrieve + generate. `"stream": true` for SSE.

WHY THE API KEY IS MANDATORY
----------------------------
This binds to a public address. An unauthenticated inference endpoint is an
open GPU: anyone who finds the port can spend your card, and generation is the
one operation here that costs real money and real time. The service therefore
REFUSES TO START on a non-loopback bind without ASTRO_API_KEY set. That is
deliberate -- a warning would be ignored, and the failure mode is silent until
the electricity bill.

WHY REQUESTS ARE SERIALIZED
---------------------------
`model.generate()` is not safe to call concurrently on one model instance --
overlapping calls share KV cache allocation and will either corrupt each other's
output or OOM the card. A single lock makes concurrent callers queue instead.
That caps throughput at one request at a time, which is the honest limit of an
unbatched HuggingFace loop: measured 33 tok/s bf16, 26 tok/s in 4-bit. If you
need real concurrency the answer is vLLM with continuous batching, not removing
this lock.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TextIteratorStreamer)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from config import SYSTEM_PROMPT, build_context

# ------------------------------------------------------------------ config
HOST         = os.environ.get("ASTRO_HOST", "0.0.0.0")
PORT         = int(os.environ.get("ASTRO_PORT", "11435"))
# BACKEND
#   local   -- transformers loads weights into this process. The default, and
#             the only one that can serve the fine-tuned adapter.
#   ollama  -- proxy generation to a local Ollama server. Lets any Ollama model
#             (llama3.2, mistral, gemma...) answer through the SAME retrieval,
#             prompt and citation payload, with no HuggingFace licence step and
#             no second copy of the weights. Useful for comparing a stock model
#             against the tuned one on identical retrieved context.
BACKEND      = os.environ.get("ASTRO_BACKEND", "local")
OLLAMA_URL   = os.environ.get("ASTRO_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_DIR    = os.environ.get("ASTRO_MODEL", os.path.join(ROOT, "models", "astro-4b"))
API_KEY      = os.environ.get("ASTRO_API_KEY", "")
PRECISION    = os.environ.get("ASTRO_PRECISION", "bf16")     # bf16 | 4bit
MAX_NEW_CAP  = int(os.environ.get("ASTRO_MAX_NEW", "400"))
RATE_PER_MIN = int(os.environ.get("ASTRO_RATE_PER_MIN", "20"))
GEN_TIMEOUT  = int(os.environ.get("ASTRO_TIMEOUT", "180"))

RAG_SYSTEM = SYSTEM_PROMPT + (
    " You are given excerpts from the source texts. Answer ONLY from those "
    "excerpts, and name the text you are drawing on. If the excerpts do not "
    "contain the answer, say so plainly rather than supplying it from memory.")

STATE: dict = {}
GPU_LOCK = asyncio.Lock()


# ------------------------------------------------------------------- auth
def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def require_key(request: Request):
    """Constant-time key check, then a per-IP sliding window."""
    supplied = request.headers.get("x-api-key", "")
    auth = request.headers.get("authorization", "")
    if not supplied and auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    # compare_digest avoids leaking the key's length/prefix through timing
    if not (supplied and secrets.compare_digest(supplied, API_KEY)):
        raise HTTPException(status_code=401, detail="missing or invalid API key")

    ip = _client_ip(request)
    now = time.monotonic()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit: {RATE_PER_MIN} requests/minute per IP. "
                       f"Retry in {int(61 - (now - q[0]))}s.")
        q.append(now)
    return True


# ------------------------------------------------------------------ model
def _load():
    idx_path = os.path.join(ROOT, "pipeline", "09_index.py")
    spec = importlib.util.spec_from_file_location("idx", idx_path)
    idx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(idx)

    if BACKEND == "ollama":
        import httpx
        # Fail at startup, not on the first request. A proxy that starts
        # cleanly and 500s on every call is the worst of both worlds.
        try:
            tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
        except Exception as e:
            raise SystemExit(f"cannot reach Ollama at {OLLAMA_URL}: {e}")
        have = {m["name"] for m in tags.get("models", [])}
        if MODEL_DIR not in have and f"{MODEL_DIR}:latest" not in have:
            raise SystemExit(
                f"Ollama has no model '{MODEL_DIR}'. Pull it first:\n"
                f"  ollama pull {MODEL_DIR}\n"
                f"available: {', '.join(sorted(have)) or '(none)'}")
        STATE.update(tok=None, model=None, search=idx.search, dev="ollama",
                     loaded_s=0.0, vram_gb=0.0)
        print(f"backend ollama -> {OLLAMA_URL}, model {MODEL_DIR}")
        print("warming retrieval index ...")
        t0 = time.time()
        STATE["search"]("warmup query", k=1)
        print(f"  index warm in {time.time() - t0:.1f}s")
        return

    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(
            f"model not found: {MODEL_DIR}\n"
            "run: python pipeline/11_merge.py --adapter runs/qlora-2ep "
            "--out models/astro-4b")

    print(f"loading {MODEL_DIR}  ({PRECISION}) ...")
    t0 = time.time()
    quant = None
    if PRECISION == "4bit":
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    has_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, quantization_config=quant,
        device_map={"": 0} if has_cuda else {"": "cpu"})
    model.eval()

    STATE.update(
        tok=tok, model=model, search=idx.search,
        dev="cuda" if has_cuda else "cpu",
        loaded_s=round(time.time() - t0, 1),
        vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if has_cuda else 0.0,
    )
    print(f"  ready in {STATE['loaded_s']}s on {STATE['dev']} "
          f"({STATE['vram_gb']} GB)")

    print("warming retrieval index ...")
    t0 = time.time()
    STATE["search"]("warmup query", k=1)
    print(f"  index warm in {time.time() - t0:.1f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    STATE.clear()


app = FastAPI(title="Astro Jyotisha adapter", version="1.0", lifespan=lifespan)


# ----------------------------------------------------------------- schema
class AnswerIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=3, ge=1, le=8)
    max_new: int = Field(default=300, ge=16)
    rag: bool = True
    stream: bool = False


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)


# -------------------------------------------------------------- generation
def _encode(question: str, ctx: str | None):
    tok = STATE["tok"]
    enc = tok.apply_chat_template(
        [{"role": "system", "content": RAG_SYSTEM if ctx else SYSTEM_PROMPT},
         {"role": "user",
          "content": f"<excerpts>\n{ctx}\n</excerpts>\n\n{question}" if ctx
                     else question}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return {k: v.to(STATE["dev"]) for k, v in enc.items()}


def _generate(enc, max_new: int) -> str:
    tok, model = STATE["tok"], STATE["model"]
    plen = enc["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][plen:], skip_special_tokens=True).strip()


def _ollama_generate(question: str, ctx: str | None, max_new: int) -> str:
    """Same messages, same system prompt, same retrieved context -- only the
    model changes. That is what makes a comparison against the tuned adapter
    mean anything: identical input, one variable."""
    import httpx
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL_DIR,
            "messages": [
                {"role": "system",
                 "content": RAG_SYSTEM if ctx else SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"<excerpts>\n{ctx}\n</excerpts>\n\n{question}"
                            if ctx else question},
            ],
            "stream": False,
            # temperature 0 mirrors do_sample=False on the local path
            "options": {"num_predict": max_new, "temperature": 0},
        },
        timeout=GEN_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _cite(hits):
    return [{"source": h["source"], "title": h["title"], "chunk_id": h["id"],
             "rrf": round(h["rrf"], 4)} for h in hits]


# ----------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def demo():
    """The demo client, served from the app itself.

    Same-origin on purpose: the page fetches "healthz" and "answer" as RELATIVE
    paths, so it works unchanged on localhost, over the LAN, and through the
    public forward without a hardcoded host or a CORS rule. The API key is typed
    by the viewer and kept in their own browser -- it is never baked into the
    page, because this file is readable by anyone who can reach the port.
    """
    path = os.path.join(ROOT, "demo.html")
    if not os.path.exists(path):
        raise HTTPException(404, "demo.html not found next to serve.py")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/healthz")
def healthz():
    """Deliberately unauthenticated and model-free, so a load balancer or an
    uptime check can hit it without a key and without touching the GPU."""
    ok = bool(STATE.get("search")) and (
        BACKEND == "ollama" or bool(STATE.get("model")))
    return {"status": "ok" if ok else "loading",
            "model": MODEL_DIR if BACKEND == "ollama" else os.path.basename(MODEL_DIR),
            "backend": BACKEND,
            "precision": PRECISION, "device": STATE.get("dev"),
            "vram_gb": STATE.get("vram_gb"), "load_seconds": STATE.get("loaded_s")}


@app.post("/search")
def search(body: SearchIn, _=Depends(require_key)):
    """Retrieval only. No GPU generation, so it is not behind the lock."""
    hits = STATE["search"](body.query, k=body.k)
    return {"query": body.query, "hits": [
        {**_cite([h])[0], "text": " ".join(h["text"].split())[:1200],
         "dense": round(h["dense"], 4), "bm25": round(h["bm25"], 2)}
        for h in hits]}


@app.post("/answer")
async def answer(body: AnswerIn, _=Depends(require_key)):
    max_new = min(body.max_new, MAX_NEW_CAP)
    hits = STATE["search"](body.question, k=body.k) if body.rag else []
    ctx = build_context(hits) if hits else None

    if BACKEND == "ollama":
        t0 = time.time()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(_ollama_generate, body.question, ctx, max_new),
                timeout=GEN_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(504, f"generation exceeded {GEN_TIMEOUT}s")
        return {"question": body.question, "answer": text,
                "citations": _cite(hits), "grounded": bool(hits),
                "model": MODEL_DIR, "backend": "ollama",
                "seconds": round(time.time() - t0, 2)}

    enc = _encode(body.question, ctx)

    if body.stream:
        return StreamingResponse(_stream(enc, max_new, hits),
                                 media_type="text/event-stream")

    t0 = time.time()
    async with GPU_LOCK:
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(_generate, enc, max_new), timeout=GEN_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(504, f"generation exceeded {GEN_TIMEOUT}s")
    return {"question": body.question, "answer": text,
            "citations": _cite(hits), "grounded": bool(hits),
            "seconds": round(time.time() - t0, 2)}


async def _stream(enc, max_new: int, hits):
    """SSE. Citations are sent FIRST so the caller can render sources while the
    answer is still being written."""
    tok, model = STATE["tok"], STATE["model"]
    yield f"event: citations\ndata: {json.dumps(_cite(hits))}\n\n"

    async with GPU_LOCK:
        streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                        skip_special_tokens=True)
        thread = threading.Thread(
            target=model.generate,
            kwargs=dict(**enc, max_new_tokens=max_new, do_sample=False,
                        pad_token_id=tok.eos_token_id, streamer=streamer),
            daemon=True)
        thread.start()
        loop = asyncio.get_running_loop()
        it = iter(streamer)
        while True:
            piece = await loop.run_in_executor(None, lambda: next(it, None))
            if piece is None:
                break
            if piece:
                yield f"data: {json.dumps({'token': piece})}\n\n"
        thread.join(timeout=5)
    yield "event: done\ndata: {}\n\n"


# ------------------------------------------------------------------- main
def main():
    loopback = HOST in ("127.0.0.1", "localhost", "::1")
    if not API_KEY and not loopback:
        raise SystemExit(
            f"\nREFUSING TO START: binding {HOST}:{PORT} with no ASTRO_API_KEY.\n"
            "An unauthenticated inference endpoint on a routable address is an\n"
            "open GPU -- anyone who finds the port can spend your card.\n\n"
            "  PowerShell:  $env:ASTRO_API_KEY = -join ((48..57)+(97..122) | "
            "Get-Random -Count 40 | %{[char]$_})\n"
            "  then:        python serve.py\n\n"
            "To run without a key, bind loopback only: ASTRO_HOST=127.0.0.1\n")
    if API_KEY and len(API_KEY) < 24:
        raise SystemExit("ASTRO_API_KEY is shorter than 24 chars -- this is "
                         "reachable from the internet; use a long random key.")

    print(f"\nAstro service  ->  http://{HOST}:{PORT}")
    print(f"  auth        {'API key required' if API_KEY else 'NONE (loopback only)'}")
    print(f"  rate limit  {RATE_PER_MIN}/min per IP")
    print(f"  max_new cap {MAX_NEW_CAP} tokens, {GEN_TIMEOUT}s timeout\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
