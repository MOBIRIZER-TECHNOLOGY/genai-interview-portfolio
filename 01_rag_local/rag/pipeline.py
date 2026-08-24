"""
The whole RAG pipeline behind one object, so `ask.py`, `serve.py` and the
evaluation harness all exercise exactly the same code path.

    RagPipeline.load("index/")
      .ask("what is the Rotterdam rule?")

Shape of a query:

    question
       |
       +-- dense retrieve (top 20) ----+
       |                               +--> RRF fuse --> rerank (cross-encoder)
       +-- BM25 retrieve  (top 20) ----+                      |
                                                    top 4 blocks
                                                              |
                                              numbered-context prompt -> Ollama
                                                              |
                                                 answer + verified citations
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .embed import Embedder
from .generate import Answer, answer as generate_answer, DEFAULT_MODEL
from .rerank import Reranker
from .retrieve import HybridRetriever
from .store import VectorStore


@dataclass
class RagResult:
    question: str
    answer: Answer
    hits: list
    retrieve_ms: float
    rerank_ms: float
    generate_ms: float

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.rerank_ms + self.generate_ms


class RagPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        llm_model: str = DEFAULT_MODEL,
        candidates: int = 20,
        top_k: int = 4,
    ):
        self.store = store
        self.embedder = embedder
        self.retriever = HybridRetriever(store, embedder)
        self.reranker = reranker
        self.llm_model = llm_model
        self.candidates = candidates
        self.top_k = top_k

    @classmethod
    def load(
        cls,
        index_dir: Path | str,
        use_reranker: bool = True,
        llm_model: str = DEFAULT_MODEL,
        **kw,
    ) -> "RagPipeline":
        index_dir = Path(index_dir)
        store = VectorStore.load(index_dir)
        # The query embedder MUST be the same model the index was built with.
        # Mixing them produces vectors in a different space -- retrieval still
        # "works", it just returns nonsense, which is a nasty silent failure.
        embedder = Embedder(store.model_name)
        reranker = Reranker() if use_reranker else None
        return cls(store, embedder, reranker, llm_model=llm_model, **kw)

    # ------------------------------------------------------------ retrieval

    def retrieve(self, question: str, mode: str = "hybrid") -> tuple[list, float, float]:
        t0 = time.perf_counter()
        if mode == "hybrid":
            hits = self.retriever.search(question, k=self.candidates, candidates=self.candidates)
        elif mode == "dense":
            hits = self.retriever.search_dense_only(question, k=self.candidates)
        elif mode == "bm25":
            hits = self.retriever.search_bm25_only(question, k=self.candidates)
        else:
            raise ValueError(f"unknown retrieval mode: {mode!r}")
        t1 = time.perf_counter()

        if self.reranker:
            hits = self.reranker.rerank(question, hits, top_k=self.top_k)
        else:
            hits = hits[: self.top_k]
        t2 = time.perf_counter()

        return hits, (t1 - t0) * 1000, (t2 - t1) * 1000

    # --------------------------------------------------------------- answer

    def ask(self, question: str, mode: str = "hybrid") -> RagResult:
        hits, r_ms, rr_ms = self.retrieve(question, mode=mode)
        t0 = time.perf_counter()
        ans = generate_answer(question, hits, model=self.llm_model)
        g_ms = (time.perf_counter() - t0) * 1000
        return RagResult(question, ans, hits, r_ms, rr_ms, g_ms)
