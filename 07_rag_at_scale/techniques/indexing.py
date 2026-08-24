"""
Indexing-side techniques: change what you store, not what you ask.

    from techniques.indexing import (semantic_chunk, contextual_prefix,
                                     late_chunk, matryoshka_truncate)

Query-side rewriting (see `query_side.py`) fixes the question. These fix the
*chunk*, and they are usually the better lever: a query rewrite costs an LLM call
on every request forever, while a better chunk is paid for once at index time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------- semantic chunking


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    reason: str = ""


def semantic_chunk(text: str, embedder, target_chars: int = 1400,
                   min_chars: int = 400, percentile: float = 90.0,
                   batch_size: int = 256) -> list[Chunk]:
    """Split where the *topic* changes, not every N characters.

    Fixed-size chunking cuts mid-argument: the sentence that states a conclusion
    ends up in a different chunk from the evidence for it, and neither retrieves
    well. Semantic chunking embeds each sentence, measures the cosine distance
    between consecutive sentences, and cuts at the peaks -- the points where the
    text moved on to something else.

    The threshold is a **percentile of the observed distances in this document**,
    not an absolute number. Absolute thresholds do not transfer: a dense
    technical page has uniformly high sentence-to-sentence distance, a narrative
    has uniformly low, and any fixed cut-off is wrong for one of them.

    Cost: one embedding pass over every sentence at index time. That is real at
    200 GB -- roughly 3-4x the embedding work -- which is why this project uses
    structural chunking for the full corpus and reserves semantic chunking for
    high-value documents.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return [Chunk(text, 0, len(text), "too short to split")]

    vecs = embedder.encode([s for s, _, _ in sentences], batch_size=batch_size,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
    # distance between neighbouring sentences; peaks are topic boundaries
    dists = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = float(np.percentile(dists, percentile))

    chunks: list[Chunk] = []
    start_i = 0
    for i, d in enumerate(dists):
        cur_start = sentences[start_i][1]
        cur_end = sentences[i][2]
        size = cur_end - cur_start
        if size < min_chars:
            continue
        if d >= threshold or size >= target_chars:
            chunks.append(Chunk(text[cur_start:cur_end], cur_start, cur_end,
                                "topic shift" if d >= threshold else "size cap"))
            start_i = i + 1

    if start_i < len(sentences):
        a, b = sentences[start_i][1], sentences[-1][2]
        if b - a > 60:
            chunks.append(Chunk(text[a:b], a, b, "tail"))
    return chunks or [Chunk(text, 0, len(text), "no boundary found")]


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n\n+")


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    out, pos = [], 0
    for m in _SENT.finditer(text):
        s = text[pos:m.start()].strip()
        if s:
            out.append((s, pos, m.start()))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        out.append((tail, pos, len(text)))
    return out


# ---------------------------------------------------- contextual retrieval


CONTEXT_PROMPT = """<document>
{document}
</document>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Write one short sentence that situates this chunk within the document, so it can
be understood on its own. Mention what the document is about and what this
specific part covers. Output only that sentence."""


def contextual_prefix(chunk: str, document: str, llm, max_doc_chars: int = 8000) -> str:
    """Anthropic's 'contextual retrieval': prepend LLM-written context to a chunk.

    A chunk that reads *"the threshold was raised to 0.92 after the Q2 review"*
    is nearly unretrievable: no document name, no product, no date. Embedded
    alone it matches almost nothing a user would actually ask.

    Prepending one generated sentence — *"This is from the atlas-vision service
    documentation, discussing barcode read confidence policy."* — puts the
    identifying vocabulary inside the embedded text. Anthropic reported ~35%
    reduction in retrieval failures from this, and ~49% combined with BM25.

    The cost is one LLM call **per chunk at index time**. At 143 M chunks that is
    completely infeasible — it would take months. This is a technique for
    corpora in the thousands-to-millions of chunks, or for a high-value subset.
    Prompt caching over the document makes it far cheaper, since the document is
    resent for every chunk in it.

    `project 01`'s heading breadcrumb is the free approximation of this idea:
    structural rather than generated, no LLM call, most of the benefit on
    well-structured documents.
    """
    doc = document[:max_doc_chars]
    context = llm(CONTEXT_PROMPT.format(document=doc, chunk=chunk)).strip()
    return f"{context}\n\n{chunk}"


# ------------------------------------------------------------ late chunking


def late_chunk(text: str, spans: list[tuple[int, int]], model, tokenizer=None,
               max_length: int = 8192) -> np.ndarray:
    """Embed the WHOLE document once, then pool per chunk.

    Normal chunking embeds each chunk in isolation, so a chunk containing "it"
    or "the service" has no idea what the referent is. Late chunking runs the
    transformer over the full document first -- so every token attends to the
    whole context -- and only *then* mean-pools the token vectors within each
    chunk's span.

    The chunk vector therefore carries contextual information the isolated chunk
    never had, at no extra LLM cost: one forward pass per document instead of one
    per chunk, which is usually *cheaper* than standard chunking.

    The constraint is the model's context window. It needs a long-context
    embedding model (jina-embeddings-v2/v3, nomic-embed-text) -- BGE's 512 tokens
    is too short for this to buy anything. That is why this is a function you can
    call rather than the default path in `build_index.py`.
    """
    import torch

    tokenizer = tokenizer or model.tokenizer
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].numpy()

    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        token_vecs = model(**enc).last_hidden_state[0].float().cpu().numpy()

    out = []
    for (a, b) in spans:
        # tokens whose character span overlaps this chunk
        mask = (offsets[:, 0] < b) & (offsets[:, 1] > a) & (offsets[:, 1] > offsets[:, 0])
        sel = token_vecs[mask]
        v = sel.mean(axis=0) if len(sel) else np.zeros(token_vecs.shape[1], np.float32)
        n = np.linalg.norm(v)
        out.append(v / n if n > 0 else v)
    return np.asarray(out, dtype=np.float32)


# -------------------------------------------------------------- Matryoshka


def matryoshka_truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Truncate embeddings to `dim` and renormalise.

    Matryoshka Representation Learning trains a model so that the *first* k
    dimensions of its output are themselves a usable embedding, for several
    nested k. You get one model that emits 768 dims, and can use 768, 256, 128 or
    64 by slicing -- no distillation, no second model.

    Why it matters at scale: it turns dimensionality into a **runtime** knob.
    Search a 64-dim index to shortlist cheaply, then rescore with the full 768.
    That is the same two-stage shape as binary-then-int8 in `scale/quantize.py`,
    along a different axis, and the two compose.

    Truncation is only valid for a model **trained** with an MRL objective --
    nomic-embed-text-v1.5, mxbai-embed-large-v1, OpenAI text-embedding-3. Slicing
    a normal embedding model destroys it, because there is nothing making the
    early dimensions more important than the late ones.
    """
    if dim > vectors.shape[-1]:
        raise ValueError(f"cannot truncate {vectors.shape[-1]} dims to {dim}")
    v = vectors[..., :dim]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def matryoshka_memory_table(n_vectors: int, dims: tuple[int, ...] = (768, 512, 256, 128, 64)) -> list[dict]:
    """What each nested dimensionality costs, at float32 and binary."""
    rows = []
    for d in dims:
        rows.append({
            "dim": d,
            "float32_gb": round(n_vectors * d * 4 / 1e9, 2),
            "binary_gb": round(n_vectors * d / 8 / 1e9, 3),
            "vs_full": f"{dims[0] / d:.0f}x smaller",
        })
    return rows
