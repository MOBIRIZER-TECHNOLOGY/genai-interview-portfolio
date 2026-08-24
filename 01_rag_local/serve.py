"""
Step 3 - serve the pipeline as an HTTP API with token streaming.

    python serve.py                 # http://localhost:8000/docs
    curl -N -X POST localhost:8000/ask/stream -H "content-type: application/json" \
         -d '{"question":"what is the Rotterdam rule?"}'

Why streaming matters and what it costs you: a 7B model on one consumer GPU
takes ~9 s to produce a full answer. Streaming does not make it faster, it makes
the *time to first token* ~1 s, and perceived latency is what users judge. The
catch is that you cannot validate the answer before the user sees the start of
it -- so this endpoint streams the tokens but sends the citation audit as a
trailing JSON event, and the client decides what to do if it comes back
ungrounded.

The model is loaded once at startup, not per request. Loading the embedder and
cross-encoder per request would add ~2 s and re-allocate VRAM every call.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.generate import DEFAULT_MODEL, SYSTEM_PROMPT, build_context, ollama_stream, verify_citations
from rag.pipeline import RagPipeline

HERE = Path(__file__).parent
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = HERE / "index"
    if not index.exists():
        raise RuntimeError(f"no index at {index} -- run: python ingest.py")
    print("loading pipeline (embedder + reranker onto GPU) ...")
    STATE["pipe"] = RagPipeline.load(index)
    print(f"ready: {len(STATE['pipe'].store)} chunks")
    yield
    STATE.clear()


app = FastAPI(
    title="Atlas RAG",
    description="Local hybrid-retrieval RAG over the Atlas corpus. No data leaves the machine.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is the Rotterdam rule?"])
    mode: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")
    top_k: int = Field(4, ge=1, le=12)
    model: str = DEFAULT_MODEL


@app.get("/health")
def health():
    pipe = STATE.get("pipe")
    return {
        "status": "ok" if pipe else "loading",
        "chunks": len(pipe.store) if pipe else 0,
        "embedder": pipe.store.model_name if pipe else None,
        "reranker": bool(pipe and pipe.reranker),
    }


@app.post("/search")
def search(req: AskRequest):
    """Retrieval only -- useful for debugging why an answer went wrong."""
    pipe = STATE["pipe"]
    pipe.top_k = req.top_k
    hits, r_ms, rr_ms = pipe.retrieve(req.question, mode=req.mode)
    return {
        "question": req.question,
        "retrieve_ms": round(r_ms, 1),
        "rerank_ms": round(rr_ms, 1),
        "hits": [
            {
                "chunk_id": h.chunk_id, "source": h.source, "breadcrumb": h.breadcrumb,
                "rrf_score": round(h.rrf_score, 5),
                "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 3),
                "body": h.body,
            }
            for h in hits
        ],
    }


@app.post("/ask")
def ask(req: AskRequest):
    pipe = STATE["pipe"]
    pipe.top_k = req.top_k
    pipe.llm_model = req.model
    try:
        result = pipe.ask(req.question, mode=req.mode)
    except Exception as exc:  # Ollama down is the common one
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc
    a = result.answer
    return {
        "question": req.question,
        "answer": a.text,
        "abstained": a.abstained,
        "grounded": a.grounded,
        "citations": a.citations,
        "invalid_citations": a.invalid_citations,
        "sources": [c for c in a.contexts if c["n"] in a.valid_citations],
        "timing_ms": {
            "retrieve": round(result.retrieve_ms),
            "rerank": round(result.rerank_ms),
            "generate": round(result.generate_ms),
            "total": round(result.total_ms),
        },
        "tokens": {"prompt": a.prompt_tokens, "generated": a.eval_tokens},
    }


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Server-sent-event style stream: `token` events, then one `done` event."""
    pipe = STATE["pipe"]
    pipe.top_k = req.top_k
    hits, r_ms, rr_ms = pipe.retrieve(req.question, mode=req.mode)
    context, manifest = build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context blocks:\n\n{context}\n\n---\n\nQuestion: {req.question}"},
    ]

    def gen():
        yield f"event: sources\ndata: {json.dumps(manifest)}\n\n"
        parts: list[str] = []
        for tok in ollama_stream(messages, model=req.model):
            parts.append(tok)
            yield f"event: token\ndata: {json.dumps(tok)}\n\n"
        full = "".join(parts)
        cited, valid, invalid = verify_citations(full, len(hits))
        audit = {
            "citations": cited,
            "invalid_citations": invalid,
            "abstained": full.strip().startswith("NOT_FOUND"),
            "grounded": full.strip().startswith("NOT_FOUND") or (bool(cited) and not invalid),
            "retrieve_ms": round(r_ms), "rerank_ms": round(rr_ms),
        }
        yield f"event: done\ndata: {json.dumps(audit)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
