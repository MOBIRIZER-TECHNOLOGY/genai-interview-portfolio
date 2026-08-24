"""
Dense embeddings on the local GPU.

Model: BAAI/bge-small-en-v1.5 (133 MB, 384 dims). Small on purpose -- for a
corpus this size the bottleneck is never embedding quality, and a small model
keeps the whole demo runnable in seconds. Swap `--model` for bge-base or
bge-large and nothing else changes.

Two things here are easy to get wrong and both cost real accuracy:

1. **Asymmetric prefixes.** BGE was trained with an instruction prefix on the
   *query* side only. Embedding a query the same way you embed a passage
   measurably drops recall. `encode_queries` adds the prefix, `encode_passages`
   does not.
2. **Normalisation.** We L2-normalise, which makes inner product == cosine
   similarity, so FAISS `IndexFlatIP` gives cosine ranking for free.
"""

from __future__ import annotations

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        # renamed in sentence-transformers 6.x; keep working on 3.x/4.x too
        getter = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = getter()

    def _encode(self, texts: list[str], batch_size: int, show: bool) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # -> inner product == cosine
            show_progress_bar=show,
        )
        return vecs.astype("float32")

    def encode_passages(self, texts: list[str], batch_size: int = 64, show: bool = True) -> np.ndarray:
        return self._encode(texts, batch_size, show)

    def encode_queries(self, texts: list[str], batch_size: int = 32, show: bool = False) -> np.ndarray:
        return self._encode([QUERY_PREFIX + t for t in texts], batch_size, show)
