"""
Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion.

Why both? They fail in opposite directions, which is exactly what you want in
an ensemble:

  - **Dense** understands paraphrase. "how long do we keep camera footage"
    retrieves the retention table even though it shares no rare words with it.
    But it is weak on rare literal tokens -- ask for `TLM-330` and a 384-dim
    embedding may not have a distinct direction for that string at all.
  - **BM25** nails exact rare tokens (`TLM-330`, `ATLAS_STARVATION_ROUNDS`,
    `nw-barcode-ocr-v2`) because rare terms get the highest IDF weight. But it
    scores zero for a paraphrase that shares no vocabulary.

Fusion method is **RRF** rather than a weighted score blend, because dense
cosine scores (~0.5-0.9) and BM25 scores (unbounded, corpus dependent) are on
incomparable scales. Normalising them requires a per-corpus calibration that
silently rots. RRF only uses *rank*, so it needs no calibration at all:

    rrf(d) = sum over retrievers of  1 / (k + rank(d))          k = 60

k=60 is the constant from the original Cormack et al. paper. It damps the
influence of the very top rank just enough that one retriever cannot dominate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .store import SearchHit, VectorStore

TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, keeping hyphen/underscore/dot-joined identifiers whole.

    This matters: a plain `\\w+` tokenizer shatters `TLM-330` into `tlm` + `330`
    and `atlas-dispatch` into two common words, throwing away the exact rare
    token that made BM25 worth having.
    """
    return TOKEN.findall(text.lower())


@dataclass
class FusedHit:
    chunk_id: str
    body: str
    source: str
    breadcrumb: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None = None


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder, rrf_k: int = 60):
        from rank_bm25 import BM25Okapi

        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.records = store.records
        self.by_id = {r["id"]: r for r in self.records}
        # BM25 indexes the raw body, not the breadcrumb-prefixed text -- the
        # breadcrumb repeats the file name in every chunk of a document, which
        # would inflate term frequency for those words across the corpus.
        self._bm25 = BM25Okapi([tokenize(r["body"]) for r in self.records])

    # ------------------------------------------------------------- pieces

    def dense(self, query: str, k: int) -> list[SearchHit]:
        qv = self.embedder.encode_queries([query])
        return self.store.search(qv, k=k)

    def bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.records[i]["id"], float(scores[i])) for i in order if scores[i] > 0]

    # -------------------------------------------------------------- fusion

    def search(self, query: str, k: int = 8, candidates: int = 20) -> list[FusedHit]:
        """Retrieve `candidates` from each arm, fuse, return the top `k`."""
        dense_hits = self.dense(query, candidates)
        bm25_hits = self.bm25(query, candidates)

        dense_rank = {h.chunk_id: i for i, h in enumerate(dense_hits)}
        bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25_hits)}

        fused: dict[str, float] = {}
        for cid, rank in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for cid, rank in bm25_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        out: list[FusedHit] = []
        for cid, score in ordered:
            r = self.by_id[cid]
            out.append(
                FusedHit(
                    chunk_id=cid,
                    body=r["body"],
                    source=r["source"],
                    breadcrumb=r["breadcrumb"],
                    rrf_score=score,
                    dense_rank=dense_rank.get(cid),
                    bm25_rank=bm25_rank.get(cid),
                )
            )
        return out

    def search_dense_only(self, query: str, k: int = 8) -> list[FusedHit]:
        """Ablation arm -- used by the evaluation harness to prove hybrid helps."""
        return [
            FusedHit(
                chunk_id=h.chunk_id,
                body=h.body,
                source=h.source,
                breadcrumb=h.breadcrumb,
                rrf_score=h.score,
                dense_rank=h.rank,
                bm25_rank=None,
            )
            for h in self.dense(query, k)
        ]

    def search_bm25_only(self, query: str, k: int = 8) -> list[FusedHit]:
        """Ablation arm."""
        out = []
        for rank, (cid, score) in enumerate(self.bm25(query, k)):
            r = self.by_id[cid]
            out.append(
                FusedHit(
                    chunk_id=cid,
                    body=r["body"],
                    source=r["source"],
                    breadcrumb=r["breadcrumb"],
                    rrf_score=score,
                    dense_rank=None,
                    bm25_rank=rank,
                )
            )
        return out
