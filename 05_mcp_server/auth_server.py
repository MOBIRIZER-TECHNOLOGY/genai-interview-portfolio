"""
The Atlas MCP server, with OAuth 2.1 required.

    python auth_server.py                  # http://localhost:8765/mcp
    python auth_client_demo.py             # full flow against it

Same tools as `server.py`, but every call must present a valid bearer token, and
each tool demands a specific scope:

    search_atlas_docs      atlas:read
    list_atlas_documents   atlas:read
    ask_atlas              atlas:ask
    triage_incident        atlas:triage      (GPU -- the expensive one)

## Why scopes per tool, not one scope per server

An MCP server is a bundle of capabilities with very different blast radii.
Reading documentation is cheap and harmless. `triage_incident` loads a model onto
the GPU and runs inference. A token minted for a docs-browsing integration should
not be able to occupy your GPU, and separating them costs one decorator.

This is the principle of least privilege applied at the tool boundary, and it is
the thing that makes "we exposed our internal systems over MCP" defensible.

## What the client sees

Unauthenticated request ->

    HTTP 401
    WWW-Authenticate: Bearer resource_metadata="http://localhost:8765/.well-known/..."

That header is the discovery mechanism (RFC 9728): the client follows it to find
the authorization server, registers itself, and runs the flow. Nothing needs to
be configured out of band.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
for noisy in ("httpx", "httpcore", "faiss", "faiss.loader", "sentence_transformers",
              "transformers", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from mcp.server import MCPServer  # noqa: E402
from mcp.server.auth.settings import (  # noqa: E402
    AuthSettings, ClientRegistrationOptions, RevocationOptions,
)
from pydantic import AnyHttpUrl  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from auth_provider import DEFAULT_SCOPES, SCOPES, AtlasOAuthProvider  # noqa: E402

RAG_PROJECT = HERE.parent / "01_rag_local"
LORA_PROJECT = HERE.parent / "02_lora_text"
CORPUS = RAG_PROJECT / "corpus"
INDEX = RAG_PROJECT / "index"
sys.path.insert(0, str(RAG_PROJECT))

ISSUER = os.environ.get("ATLAS_ISSUER", "http://localhost:8765")

provider = AtlasOAuthProvider()

server = MCPServer(
    name="atlas-knowledge-secure",
    version="1.0.0",
    log_level="WARNING",
    instructions=(
        "Authenticated Atlas knowledge and triage tools. Every call requires an "
        "OAuth 2.1 bearer token. Scopes: atlas:read (search/list), atlas:ask "
        "(RAG answers), atlas:triage (GPU triage model)."
    ),
    auth_server_provider=provider,
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(ISSUER),
        resource_server_url=AnyHttpUrl(ISSUER),
        # Server-wide floor. Per-tool scopes below are the real enforcement --
        # keeping this at the least-privileged scope means a docs token is not
        # rejected outright at the door.
        required_scopes=list(DEFAULT_SCOPES),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=list(SCOPES),
            default_scopes=list(DEFAULT_SCOPES),
        ),
        revocation_options=RevocationOptions(enabled=True),
    ),
)

_STATE: dict[str, Any] = {}


# --------------------------------------------------------------- authz


class Forbidden(Exception):
    """Raised when a valid token lacks the scope a tool requires."""


def require_scope(scope: str) -> list[str]:
    """Assert the caller's token carries `scope`; return its full scope list.

    The SDK has already authenticated the token by the time a tool body runs --
    this is *authorization*, the separate question of whether this particular
    caller may do this particular thing.
    """
    from mcp.server.auth.middleware.auth_context import get_access_token

    token = get_access_token()
    if token is None:
        raise Forbidden("no access token in request context")
    if scope not in token.scopes:
        raise Forbidden(
            f"this tool requires the '{scope}' scope; your token has "
            f"{sorted(token.scopes)}. Request it at authorization time."
        )
    return list(token.scopes)


def _pipeline():
    if "pipe" not in _STATE:
        if not INDEX.exists():
            raise RuntimeError(f"no RAG index at {INDEX}; run 01_rag_local/ingest.py")
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
            raise RuntimeError(f"no adapter at {adapter}; train project 02 first")
        base = json.loads((adapter / "training_info.json").read_text(encoding="utf-8"))["base_model"]
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
        _STATE["triage"] = (PeftModel.from_pretrained(model, str(adapter)).eval(), tok)
    return _STATE["triage"]


# --------------------------------------------------------------- tools


@server.tool(description="List Atlas documentation files. Requires scope: atlas:read")
def list_atlas_documents() -> dict:
    require_scope("atlas:read")
    docs = []
    for path in sorted(CORPUS.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        docs.append({
            "name": path.name,
            "title": next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), path.stem),
            "sections": [l.lstrip("# ").strip() for l in lines if l.startswith("## ")],
        })
    return {"count": len(docs), "documents": docs}


@server.tool(description="Search Atlas docs, returning passages verbatim. Requires scope: atlas:read")
def search_atlas_docs(query: str, top_k: int = 5,
                      mode: Literal["hybrid", "dense", "bm25"] = "hybrid") -> dict:
    require_scope("atlas:read")
    pipe = _pipeline()
    pipe.top_k = max(1, min(top_k, 12))
    hits, r_ms, rr_ms = pipe.retrieve(query, mode=mode)
    return {
        "query": query, "latency_ms": round(r_ms + rr_ms),
        "results": [{"source": h.source, "section": h.breadcrumb, "text": h.body} for h in hits],
    }


@server.tool(description="Answer a question with cited sources. Requires scope: atlas:ask")
def ask_atlas(question: str, top_k: int = 4) -> dict:
    require_scope("atlas:ask")
    pipe = _pipeline()
    pipe.top_k = max(1, min(top_k, 10))
    result = pipe.ask(question)
    a = result.answer
    return {
        "answer": a.text, "abstained": a.abstained, "grounded": a.grounded,
        "sources": [{"source": c["source"], "section": c["breadcrumb"]}
                    for c in a.contexts if c["n"] in a.valid_citations],
        "latency_ms": round(result.total_ms),
    }


@server.tool(description="Structured incident triage via the fine-tuned model (GPU). "
                         "Requires scope: atlas:triage")
def triage_incident(report: str) -> dict:
    require_scope("atlas:triage")
    import torch

    sys.path.insert(0, str(LORA_PROJECT))
    from make_dataset import SYSTEM

    model, tok = _triage_model()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": report}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=110, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        return {"triage": json.loads(raw)}
    except json.JSONDecodeError:
        return {"error": "model did not return valid JSON", "raw": raw}


@server.tool(description="Show the scopes on your current token. Requires any valid token.")
def whoami() -> dict:
    from mcp.server.auth.middleware.auth_context import get_access_token

    token = get_access_token()
    if token is None:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "client_id": token.client_id,
        "scopes": sorted(token.scopes),
        "expires_at": token.expires_at,
        "allowed_tools": sorted(
            t for t, s in (("list_atlas_documents", "atlas:read"),
                           ("search_atlas_docs", "atlas:read"),
                           ("ask_atlas", "atlas:ask"),
                           ("triage_incident", "atlas:triage"))
            if s in token.scopes
        ),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    # OAuth needs HTTP: the flow is browser redirects and back-channel POSTs,
    # none of which exist over stdio. stdio servers are trusted subprocesses and
    # need no auth; HTTP servers are reachable and therefore do.
    print(f"Atlas MCP (OAuth 2.1) on http://127.0.0.1:{args.port}/mcp")
    print(f"scopes: {', '.join(SCOPES)}")
    server.run(transport="streamable-http", host="127.0.0.1", port=args.port)
