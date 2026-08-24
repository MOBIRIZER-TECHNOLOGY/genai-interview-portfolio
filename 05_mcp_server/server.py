"""
An MCP server that exposes the Atlas knowledge base and the fine-tuned triage
model as tools an AI agent can call.

    python server.py                 # stdio (what Claude Code speaks)
    python server.py --http          # streamable HTTP on :8765
    python client_demo.py            # drive it without an AI client

## What MCP is, and why it exists

An LLM on its own is a text function. To be useful it needs to *do* things — read
your files, query your database, call your API. Before MCP, every AI client
invented its own plugin format, so every integration was written N times.

**MCP (Model Context Protocol)** is a JSON-RPC protocol that standardises the
connection. Write one server, and any MCP-speaking client (Claude Code, Claude
Desktop, an SDK agent, an IDE) can use it.

Three primitives, and picking the right one is most of the design work:

| Primitive | Controlled by | Use it for |
|---|---|---|
| **Tool** | the *model* decides to call it | actions and lookups: search, query, write |
| **Resource** | the *application/user* attaches it | context to read: files, records, schemas |
| **Prompt** | the *user* invokes it | reusable templates, e.g. a slash command |

The distinction that trips people up: a Resource is *passive context*, a Tool is
an *action the model chooses*. If the model needs to decide whether to fetch it,
it's a tool. If the user is attaching it, it's a resource.

## What this server exposes

Tools      search_atlas_docs, ask_atlas, list_atlas_documents, triage_incident
Resources  atlas://docs/{name}, atlas://index/stats
Prompts    incident_triage
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

# Quiet the dependency chatter BEFORE importing anything that logs. On stdio,
# stdout is the protocol stream and stderr is what the client surfaces to the
# user -- a server that dumps 200 lines of HF download logs on every start looks
# broken even when it works.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for noisy in ("httpx", "httpcore", "faiss", "faiss.loader", "sentence_transformers",
              "transformers", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from mcp.server import MCPServer  # noqa: E402

HERE = Path(__file__).parent
RAG_PROJECT = HERE.parent / "01_rag_local"
LORA_PROJECT = HERE.parent / "02_lora_text"
CORPUS = RAG_PROJECT / "corpus"
INDEX = RAG_PROJECT / "index"

sys.path.insert(0, str(RAG_PROJECT))

server = MCPServer(
    name="atlas-knowledge",
    version="1.0.0",
    log_level="WARNING",
    instructions=(
        "Knowledge and triage tools for the Atlas warehouse-robotics platform.\n"
        "- Use `ask_atlas` for a direct question you want answered with citations.\n"
        "- Use `search_atlas_docs` when you want the raw passages to reason over yourself.\n"
        "- Use `triage_incident` to convert a free-text operator report into a "
        "structured triage record.\n"
        "All data is local; nothing is sent to an external service."
    ),
)

# Heavy objects are created on first use, not at import. An MCP server that
# takes 8 seconds to start looks broken to the client -- stdio servers are
# spawned on demand and are expected to be responsive immediately.
_STATE: dict[str, Any] = {}


def _pipeline():
    if "pipe" not in _STATE:
        if not INDEX.exists():
            raise RuntimeError(
                f"No RAG index at {INDEX}. Run:  cd {RAG_PROJECT.name} && python ingest.py"
            )
        from rag.pipeline import RagPipeline

        _STATE["pipe"] = RagPipeline.load(INDEX)
    return _STATE["pipe"]


def _triage_model():
    if "triage" not in _STATE:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = LORA_PROJECT / "lora-out"
        if not adapter.exists():
            raise RuntimeError(
                f"No adapter at {adapter}. Run:  cd {LORA_PROJECT.name} && "
                "python make_dataset.py && python train_lora.py"
            )
        base = json.loads((adapter / "training_info.json").read_text(encoding="utf-8"))["base_model"]
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
        model = PeftModel.from_pretrained(model, str(adapter)).eval()
        _STATE["triage"] = (model, tok)
    return _STATE["triage"]


# ------------------------------------------------------------------- tools


@server.tool(
    description=(
        "Answer a question about the Atlas platform from the internal documentation, "
        "with verified citations. Returns NOT_FOUND if the docs do not cover it "
        "rather than guessing. Use this when you want an answer; use "
        "search_atlas_docs when you want the raw source passages."
    )
)
def ask_atlas(question: str, top_k: int = 4) -> dict:
    """Full RAG: hybrid retrieval -> rerank -> grounded generation."""
    pipe = _pipeline()
    pipe.top_k = max(1, min(top_k, 10))
    result = pipe.ask(question)
    a = result.answer
    return {
        "answer": a.text,
        "abstained": a.abstained,
        "grounded": a.grounded,
        "sources": [
            {"source": c["source"], "section": c["breadcrumb"]}
            for c in a.contexts
            if c["n"] in a.valid_citations
        ],
        "latency_ms": round(result.total_ms),
    }


@server.tool(
    description=(
        "Search the Atlas documentation and return the matching passages verbatim, "
        "without generating an answer. Use when you want to read the source text "
        "and reason about it yourself, or to check what the docs actually say."
    )
)
def search_atlas_docs(
    query: str,
    top_k: int = 5,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
) -> dict:
    """Retrieval only. `mode` exposes the individual retrieval arms for debugging."""
    pipe = _pipeline()
    pipe.top_k = max(1, min(top_k, 12))
    hits, retrieve_ms, rerank_ms = pipe.retrieve(query, mode=mode)
    return {
        "query": query,
        "mode": mode,
        "latency_ms": round(retrieve_ms + rerank_ms),
        "results": [
            {
                "source": h.source,
                "section": h.breadcrumb,
                "text": h.body,
                "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 4),
            }
            for h in hits
        ],
    }


@server.tool(
    description="List the Atlas documentation files available to search, with their section headings."
)
def list_atlas_documents() -> dict:
    """Cheap discovery call so an agent can orient before searching."""
    docs = []
    for path in sorted(CORPUS.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        docs.append(
            {
                "name": path.name,
                "uri": f"atlas://docs/{path.name}",
                "title": next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), path.stem),
                "sections": [l.lstrip("# ").strip() for l in lines if l.startswith("## ")],
                "bytes": path.stat().st_size,
            }
        )
    return {"count": len(docs), "documents": docs}


@server.tool(
    description=(
        "Convert a free-text operator incident report into a structured triage "
        "record (component, severity, error_code, page_oncall, action) using the "
        "LoRA-fine-tuned Atlas triage model. Requires project 02 to have been trained."
    )
)
def triage_incident(report: str) -> dict:
    """Runs the project-02 fine-tuned adapter. Local GPU inference."""
    import torch

    sys.path.insert(0, str(LORA_PROJECT))
    from make_dataset import SYSTEM

    model, tok = _triage_model()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": report}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **enc, max_new_tokens=110, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    raw = tok.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    try:
        return {"triage": json.loads(raw), "raw": raw}
    except json.JSONDecodeError:
        # Surface the failure rather than silently returning something plausible.
        return {"error": "model did not return valid JSON", "raw": raw}


# --------------------------------------------------------------- resources


@server.resource(
    "atlas://docs/{name}",
    description="Read one Atlas documentation file verbatim.",
    mime_type="text/markdown",
)
def read_doc(name: str) -> str:
    # The SDK's resource_security rejects traversal and absolute paths, but a
    # server should never rely solely on the framework for this.
    path = (CORPUS / name).resolve()
    if not path.is_relative_to(CORPUS.resolve()) or not path.exists():
        raise ValueError(f"unknown document: {name}")
    return path.read_text(encoding="utf-8")


@server.resource(
    "atlas://index/stats",
    description="Statistics about the RAG index: chunk count, embedding model, dimensions.",
    mime_type="application/json",
)
def index_stats() -> str:
    meta_path = INDEX / "meta.json"
    if not meta_path.exists():
        return json.dumps({"status": "not built", "hint": "run 01_rag_local/ingest.py"})
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["documents"] = len(list(CORPUS.glob("*.md")))
    meta["triage_adapter_trained"] = (LORA_PROJECT / "lora-out").exists()
    return json.dumps(meta, indent=2)


# ----------------------------------------------------------------- prompts


@server.prompt(
    name="incident_triage",
    description="Investigate an Atlas incident: triage it, then find the runbook steps.",
)
def incident_triage_prompt(report: str) -> str:
    return (
        f"An operator filed this Atlas incident report:\n\n{report}\n\n"
        "Do the following, in order:\n"
        "1. Call `triage_incident` to get the structured record.\n"
        "2. Call `ask_atlas` to find the runbook procedure for that component and severity.\n"
        "3. Summarise: what is broken, how severe it is, whether to page on-call, "
        "and the exact remediation steps -- citing the documents you used.\n"
        "If the documentation does not cover it, say so explicitly rather than improvising."
    )


# -------------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if args.http:
        server.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    else:
        # stdio: the client spawns this process and talks JSON-RPC over the pipes.
        # NOTHING may be printed to stdout -- it would corrupt the protocol stream.
        # Use stderr for logging.
        server.run(transport="stdio")
