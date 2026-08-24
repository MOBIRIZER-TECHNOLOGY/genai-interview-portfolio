"""
Cross-encoder reranking.

The retriever is a **bi-encoder**: query and passage are embedded separately and
compared with a dot product. That is what makes it fast -- passages are embedded
once, offline. It is also what makes it imprecise: the model never sees the
query and the passage together, so it cannot reason about whether *this specific
question* is answered by *this specific text*.

A **cross-encoder** feeds `[query, passage]` through the transformer jointly and
outputs one relevance score. Far more accurate, and far too slow to run over a
whole corpus -- it is O(N) forward passes per query. So the standard shape is:

    retrieve 20 cheaply  ->  rerank those 20 precisely  ->  keep the top 4

That two-stage pattern is the single highest-leverage upgrade to a naive RAG
pipeline, and this project measures the lift in `eval/evaluate.py`.
"""

from __future__ import annotations

DEFAULT_RERANKER = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None):
        from sentence_transformers import CrossEncoder
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=self.device, max_length=512)

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list:
        """Score every hit against the query and return them best-first.

        Mutates `rerank_score` on each hit so the caller can show both the
        retrieval rank and the reranked rank side by side.
        """
        if not hits:
            return []
        pairs = [(query, h.body) for h in hits]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        ordered = sorted(hits, key=lambda h: -h.rerank_score)
        return ordered[:top_k] if top_k else ordered
