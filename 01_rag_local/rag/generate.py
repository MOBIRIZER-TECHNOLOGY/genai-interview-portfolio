"""
Answer generation against a local Ollama model, with enforced citations and an
explicit abstain path.

Two design decisions worth defending in an interview:

1. **Numbered context blocks + "cite the block number".** Asking for citations
   in free text ("cite the source") produces plausible-looking made-up file
   names. Giving the model a numbered list and asking for `[2]` gives you a
   citation you can *mechanically verify* against the context you actually sent
   -- see `verify_citations`. A citation you cannot check is decoration.

2. **An explicit abstain instruction.** The default failure mode of RAG is not
   "no answer", it is a confident answer synthesised from the model's pretraining
   when retrieval missed. The prompt makes NOT_FOUND a first-class, named output.

Talking to Ollama over plain HTTP rather than a client library keeps the
dependency surface at `httpx` and makes the request shape visible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are a precise technical assistant for the Atlas platform.

Rules:
1. Answer ONLY from the numbered context blocks provided. They are the single
   source of truth, even if they contradict what you believe.
2. Cite the block number in square brackets after every factual claim, like [2].
   A sentence with a fact and no citation is a failure.
3. If the context does not contain the answer, reply with exactly:
   NOT_FOUND: <one sentence saying what is missing>
   Do not guess, and do not fall back on general knowledge.
4. Be concise. Quote exact numbers, error codes and identifiers verbatim."""

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    citations: list[int]
    valid_citations: list[int]
    invalid_citations: list[int]
    abstained: bool
    contexts: list[dict] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    eval_tokens: int = 0
    seconds: float = 0.0

    @property
    def grounded(self) -> bool:
        """True if the answer either abstained or cited only real blocks."""
        return self.abstained or (bool(self.citations) and not self.invalid_citations)


def build_context(hits: list) -> tuple[str, list[dict]]:
    """Render retrieved hits as numbered blocks; return the text and a manifest."""
    blocks, manifest = [], []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] source: {h.source}\n{h.breadcrumb}\n\n{h.body}")
        manifest.append(
            {
                "n": i,
                "chunk_id": h.chunk_id,
                "source": h.source,
                "breadcrumb": h.breadcrumb,
                "rerank_score": getattr(h, "rerank_score", None),
            }
        )
    return "\n\n---\n\n".join(blocks), manifest


def verify_citations(text: str, n_blocks: int) -> tuple[list[int], list[int], list[int]]:
    """Split the cited block numbers into (all, valid, hallucinated)."""
    cited = sorted({int(m) for m in CITATION.findall(text)})
    valid = [c for c in cited if 1 <= c <= n_blocks]
    invalid = [c for c in cited if c not in valid]
    return cited, valid, invalid


def ollama_chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    url: str = OLLAMA_URL,
    temperature: float = 0.0,
    num_ctx: int = 8192,
    timeout: float = 180.0,
) -> dict:
    """One non-streaming chat call. temperature=0 so eval runs are repeatable."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    r = httpx.post(f"{url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ollama_stream(messages: list[dict], model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
                  temperature: float = 0.0, num_ctx: int = 8192):
    """Yield answer tokens as they are produced (used by serve.py and ask.py)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    with httpx.stream("POST", f"{url}/api/chat", json=payload, timeout=180.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("done"):
                break
            tok = obj.get("message", {}).get("content", "")
            if tok:
                yield tok


def answer(question: str, hits: list, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL) -> Answer:
    context, manifest = build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context blocks:\n\n{context}\n\n---\n\nQuestion: {question}"},
    ]
    resp = ollama_chat(messages, model=model, url=url)
    text = resp["message"]["content"].strip()
    cited, valid, invalid = verify_citations(text, len(hits))
    return Answer(
        text=text,
        citations=cited,
        valid_citations=valid,
        invalid_citations=invalid,
        abstained=text.startswith("NOT_FOUND"),
        contexts=manifest,
        model=model,
        prompt_tokens=resp.get("prompt_eval_count", 0),
        eval_tokens=resp.get("eval_count", 0),
        seconds=resp.get("total_duration", 0) / 1e9,
    )
