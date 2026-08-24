"""
ColBERT-style late interaction: one vector per *token*, not per chunk.

    from techniques.late_interaction import ColBERTScorer
    scorer = ColBERTScorer()
    scores = scorer.score(query, passages)

## The idea

A normal bi-encoder squashes a whole passage into **one** vector. Everything the
passage says has to survive that compression, so a long passage that answers your
question in one clause gets averaged into vagueness — the signal is diluted by
the other 300 tokens.

A cross-encoder (project 01's reranker) avoids that by reading query and passage
together, but costs a full transformer pass **per pair**, so it can only ever be
a reranker over a handful of candidates.

**Late interaction is the middle.** Keep one vector per token on both sides, and
score with MaxSim: for each query token, find its best-matching passage token,
and sum those maxima.

    score(q, d) = sum over query tokens i of  max over passage tokens j of  (q_i . d_j)

Every query token gets to find its own evidence, anywhere in the passage. A rare
term matches the one place it appears instead of being averaged away. And because
document token vectors are computed offline, scoring is just matrix
multiplication — no transformer pass at query time.

## The cost, which is the whole reason this isn't the default

Storage. One vector per token instead of per chunk:

    chunk-level : 1 vector  per ~350 tokens
    token-level : 350 vectors per ~350 tokens        -> ~100-300x more vectors

At our scale that is fatal: 465 M chunks would become ~160 **billion** token
vectors. ColBERTv2/PLAID make it survivable with aggressive residual compression
(centroid + 2-bit residual, ~36x), but it is still an order of magnitude more
storage than a flat bi-encoder index.

**Where it actually belongs in a 200 GB system:** not as the primary index, but
as a *second-stage reranker* over the few hundred candidates the binary index
returns — cheaper than a cross-encoder, more accurate than the bi-encoder score.
That is how this module is meant to be used, and `rerank()` below is that path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LateInteractionHit:
    index: int
    score: float
    matched_tokens: list[tuple[str, str, float]]   # (query token, best doc token, sim)


class ColBERTScorer:
    """MaxSim scoring on top of any HF encoder.

    Uses a plain encoder rather than a real ColBERT checkpoint by default, so it
    runs with what this project already has. A true ColBERT model is *trained*
    with the MaxSim objective and will score noticeably better -- swap
    `colbert-ir/colbertv2.0` in via `--model` to use one. The mechanics
    demonstrated here are identical.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str | None = None, dim: int | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = dim          # optional projection to shrink storage

    def _token_vectors(self, texts: list[str], max_length: int = 256):
        """Return (vectors [n, T, d], attention masks [n, T], token strings)."""
        import torch

        enc = self.tok(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc).last_hidden_state
        # L2-normalise per token so the dot product is a cosine
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if self.dim:
            out = torch.nn.functional.normalize(out[..., : self.dim], p=2, dim=-1)
        toks = [self.tok.convert_ids_to_tokens(ids) for ids in enc["input_ids"].cpu()]
        return out.float().cpu().numpy(), enc["attention_mask"].cpu().numpy(), toks

    # ------------------------------------------------------------ scoring

    def score(self, query: str, passages: list[str], explain: bool = False,
              max_length: int = 256) -> list[LateInteractionHit]:
        """MaxSim score of `query` against each passage, best first."""
        qv, qm, qt = self._token_vectors([query], max_length=64)
        dv, dm, dt = self._token_vectors(passages, max_length=max_length)

        q_valid = qm[0].astype(bool)
        # drop special tokens: [CLS]/[SEP] match everything and add constant noise
        q_keep = np.array([v and t not in self.tok.all_special_tokens
                           for v, t in zip(q_valid, qt[0])])
        Q = qv[0][q_keep]
        q_tokens = [t for t, keep in zip(qt[0], q_keep) if keep]

        hits: list[LateInteractionHit] = []
        for i in range(len(passages)):
            d_valid = dm[i].astype(bool)
            D = dv[i][d_valid]
            if len(D) == 0 or len(Q) == 0:
                hits.append(LateInteractionHit(i, 0.0, []))
                continue

            sim = Q @ D.T                     # [q_tokens, d_tokens]
            best = sim.max(axis=1)            # each query token's best evidence
            total = float(best.sum())

            matched = []
            if explain:
                arg = sim.argmax(axis=1)
                d_toks = [t for t, v in zip(dt[i], d_valid) if v]
                matched = [
                    (q_tokens[j], d_toks[arg[j]], float(best[j]))
                    for j in np.argsort(-best)[:6]
                ]
            hits.append(LateInteractionHit(i, total, matched))

        return sorted(hits, key=lambda h: -h.score)

    def rerank(self, query: str, passages: list[str], top_k: int = 10) -> list[LateInteractionHit]:
        """The production-shaped use: rerank a candidate set from the vector index.

        Cheaper than a cross-encoder (no per-pair transformer pass if document
        vectors are precomputed) and more expressive than a single-vector score.
        """
        return self.score(query, passages)[:top_k]

    # -------------------------------------------------------- storage math

    @staticmethod
    def storage_estimate(n_chunks: int, tokens_per_chunk: int = 350,
                         dim: int = 128, bits: int = 2) -> dict:
        """Why late interaction cannot be the primary index at this scale."""
        n_tokens = n_chunks * tokens_per_chunk
        return {
            "n_chunks": n_chunks,
            "n_token_vectors": n_tokens,
            "chunk_level_binary_gb": round(n_chunks * dim / 8 / 1e9, 2),
            "token_level_float16_gb": round(n_tokens * dim * 2 / 1e9, 1),
            "token_level_compressed_gb": round(n_tokens * dim * bits / 8 / 1e9, 1),
            "blowup_vs_chunk_binary": f"{(n_tokens * bits) / n_chunks:.0f}x",
        }
