"""
Entity/relation extraction: turn corpus chunks into knowledge-graph triples.

    python -m graphrag.extract          # corpus -> graph_data.json (cached)

## Why a graph at all

Vector RAG retrieves *chunks that resemble the question*. That fails on a
specific class of question — the **relational hop** — where the answer lives in
the join of two facts that never appear in the same chunk:

    "What is the response time for the severity that shed mode triggers?"

No chunk contains both "shed mode -> SEV3" and "SEV3 -> next business day"; the
embedding of the question resembles neither strongly. A graph stores the *edges*
explicitly, so answering becomes a two-hop walk instead of a similarity prayer.

## The honest cost, stated up front

Extraction is **one LLM call per chunk at index time**. On 30 chunks that is a
couple of minutes; on project 07's 13.6M chunks it would be ~50 days of local
inference. GraphRAG's economics are the reverse of vector RAG: expensive index,
cheap targeted queries. It is a technique for corpora that are small, stable and
entity-dense — exactly like an internal ops handbook, exactly unlike a web crawl.

Extraction quality also *is* system quality: a relation the LLM misses at index
time is unretrievable forever, with no error. That is why triples carry their
`source` chunk id — every graph fact remains auditable back to the text that
produced it, and the generator cites the chunk, not the triple.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "01_rag_local"))

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"
GRAPH_DATA = HERE / "graph_data.json"

EXTRACT_PROMPT = """Extract factual (subject, relation, object) triples from this text.

Rules:
- subjects/objects are SHORT canonical entity names, lowercase, exactly as the
  text names them (keep identifiers verbatim: "tlm-330", "sev2", "atlas-dispatch")
- relations are short verb phrases: "has_response_time", "triggers", "means",
  "retained_for", "fixed_by", "runs_on", "part_of"
- extract numbers and durations as objects verbatim ("15 min", "0.92", "90 days")
- named concepts are entities too: modes ("shed mode"), rules ("rotterdam rule"),
  windows ("freeze window"), lanes, guards -- link facts to THEM, not only to the
  service that owns them
- parenthetical examples create triples: "SEV3 ... (e.g. shed mode)" means
  ("shed mode", "classified_as", "sev3")
- table rows are triples: each cell relates to its row's subject
- only facts stated in the text; nothing inferred

Text:
{chunk}

Reply with JSON only:
{{"triples": [{{"s": "...", "r": "...", "o": "..."}}]}}"""


def _chat_json(prompt: str, url: str = OLLAMA_URL, model: str = MODEL) -> dict | None:
    r = httpx.post(f"{url}/api/chat", timeout=180, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",                     # constrain decoding to valid JSON
        "options": {"temperature": 0.0},
    })
    r.raise_for_status()
    text = r.json()["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else None


def normalize_entity(name: str) -> str:
    """Canonical node key. Deterministic — this is what joins triples together.

    A graph is only as connected as its entity resolution: if one chunk yields
    "SEV2" and another "sev 2", the hop between them silently does not exist.
    Lowercasing and space/hyphen folding fixes the common LLM inconsistencies;
    real systems add alias tables and embedding-based merging on top.
    """
    n = name.strip().lower()
    # possessives BEFORE quote-stripping: "shed mode's severity" must yield the
    # token "shed mode", not "shed modes" (which matches no node -- found by test)
    n = re.sub(r"'s(?=\s|$)", "", n)
    n = re.sub(r"[`\"']", "", n)
    # the extractor oscillates between "shed mode" and "shed_mode" across
    # chunks; without folding, the two variants become disconnected nodes and
    # every hop between them silently does not exist (observed: the crucial
    # shed_mode --classified_as--> sev3 edge was unreachable from "shed mode")
    n = n.replace("_", " ")
    n = re.sub(r"\s+", " ", n)
    # fold "tlm 330" -> "tlm-330" style identifiers
    n = re.sub(r"\b([a-z]{2,4}) (\d{3})\b", r"\1-\2", n)
    n = re.sub(r"\bsev (\d)\b", r"sev\1", n)
    return n


def extract_graph(force: bool = False, url: str = OLLAMA_URL, model: str = MODEL) -> dict:
    """Run extraction over the project-01 corpus chunks; cache the result.

    Cached because extraction is the expensive, non-deterministic step -- the
    graph build and every query on top of it are deterministic and re-runnable.
    """
    if GRAPH_DATA.exists() and not force:
        return json.loads(GRAPH_DATA.read_text(encoding="utf-8"))

    from rag.chunking import chunk_corpus

    corpus = ROOT / "01_rag_local" / "corpus"
    chunks = chunk_corpus(corpus)
    print(f"extracting triples from {len(chunks)} chunks with {model} ...")

    triples: list[dict] = []
    chunk_text: dict[str, str] = {}
    t0 = time.perf_counter()

    for i, c in enumerate(chunks, 1):
        chunk_text[c.id] = c.body
        out = _chat_json(EXTRACT_PROMPT.format(chunk=c.text), url, model)
        got = 0
        for t in (out or {}).get("triples", []):
            if not isinstance(t, dict):
                continue
            # the model sometimes emits null or non-string values for a field;
            # coerce defensively rather than crash 20 chunks into a run
            s, r, o = (str(t.get(k) or "").strip() for k in ("s", "r", "o"))
            if not (s and r and o):
                continue
            triples.append({
                "s": normalize_entity(s),
                "r": re.sub(r"\s+", "_", r.strip().lower()),
                "o": normalize_entity(o),
                "source": c.id,
            })
            got += 1
        print(f"  [{i:>2}/{len(chunks)}] {c.id:<28} {got} triples", flush=True)

    data = {
        "model": model,
        "n_chunks": len(chunks),
        "triples": triples,
        "chunk_text": chunk_text,
        "extract_seconds": round(time.perf_counter() - t0, 1),
    }
    GRAPH_DATA.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"\n{len(triples)} triples in {data['extract_seconds']}s -> {GRAPH_DATA.name}")
    return data


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()
    extract_graph(force=args.force)
