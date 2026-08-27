"""
Build the retrieval index over chunks.jsonl.

    python pipeline/09_index.py            # build
    python pipeline/09_index.py --query "What is Arudha Pada?"

HYBRID retrieval: dense embeddings + BM25, fused with Reciprocal Rank Fusion.

Why both. This corpus is full of rare, highly specific Sanskrit terms --
Arudha, Chandal Yog, Ashtakavarga, Prishtodaya. Dense embedding models were not
trained on much Jyotisha and blur those into nearby concepts; BM25 matches them
exactly because they are rare tokens with high IDF. Conversely BM25 fails on
paraphrase ("what happens when Saturn is in the 7th" vs a passage phrased
"Sani in Kalatra Bhava"), which is exactly where dense wins. Each covers the
other's blind spot.

Writes build/index.npz (embeddings) + build/bm25.json (term stats).
"""
import argparse, json, math, os, re, sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import BUILD, CHUNKS, RETRIEVAL_CHUNKS

EMB_PATH = os.path.join(BUILD, "index.npz")
BM25_PATH = os.path.join(BUILD, "bm25.json")
MODEL = "BAAI/bge-small-en-v1.5"        # 384-dim, ~130 MB, strong for its size

TOKEN = re.compile(r"[a-z0-9Ā-ſḀ-ỿ]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


# --------------------------------------------------------------------- BM25
class BM25:
    """Standard Okapi BM25. Pure python -- no index server, no extra service."""

    def __init__(self, docs=None, state=None):
        if state:
            self.df = state["df"]
            self.idf = state["idf"]
            self.doc_len = state["doc_len"]
            self.avgdl = state["avgdl"]
            self.N = state["N"]
            self.tf = [Counter(d) for d in state["tf_keys"]]
            return
        toks = [tokenize(d) for d in docs]
        self.N = len(toks)
        self.doc_len = [len(t) for t in toks]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self.tf = [Counter(t) for t in toks]
        self.df = Counter()
        for c in self.tf:
            self.df.update(c.keys())
        # +0.5/+0.5 smoothing keeps idf positive for terms in most documents
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for w, n in self.df.items()}

    def scores(self, query, k1=1.5, b=0.75):
        q = tokenize(query)
        out = np.zeros(self.N, dtype=np.float32)
        for w in q:
            if w not in self.idf:
                continue
            idf = self.idf[w]
            for i, c in enumerate(self.tf):
                f = c.get(w)
                if not f:
                    continue
                dl = self.doc_len[i]
                out[i] += idf * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * dl / max(self.avgdl, 1e-9)))
        return out

    def save(self, path):
        json.dump({"df": self.df, "idf": self.idf, "doc_len": self.doc_len,
                   "avgdl": self.avgdl, "N": self.N,
                   "tf_keys": [dict(c) for c in self.tf]},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @staticmethod
    def load(path):
        return BM25(state=json.load(open(path, encoding="utf-8")))


# -------------------------------------------------------------------- build
def build():
    from sentence_transformers import SentenceTransformer

    # Index the RETRIEVAL chunking (~350 tok), never the training chunking
    # (~1,800 tok). See RETRIEVAL_TARGET in 01_extract.py: the embedder caps at
    # 512 tokens, so indexing the training chunks silently dropped 68.3% of the
    # corpus. Falls back only if the file has not been built yet.
    src = RETRIEVAL_CHUNKS if os.path.exists(RETRIEVAL_CHUNKS) else CHUNKS
    if src is CHUNKS:
        print("!! retrieval_chunks.jsonl missing -- indexing the 1,800-token "
              "training chunks, which the embedder will truncate. "
              "Run 01_extract.py.")
    chunks = [json.loads(l) for l in open(src, encoding="utf-8")]
    # Index TITLE + TEXT, not text alone. The book name lives in metadata, so a
    # query naming a source ("what does Phaladeepika say about...") could not
    # match it at all -- retrieval returned the right topic from the wrong book.
    texts = [f"{c['title']}. {c['title']}. {c['text']}" for c in chunks]
    print(f"{len(chunks):,} chunks (indexed as title + text)")

    print(f"embedding with {MODEL} ...")
    m = SentenceTransformer(MODEL, device="cuda")
    emb = m.encode(texts, batch_size=64, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True)
    np.savez_compressed(EMB_PATH, emb=emb.astype(np.float32),
                        ids=np.array([c["id"] for c in chunks]))
    print(f"  embeddings {emb.shape} -> {EMB_PATH} "
          f"({os.path.getsize(EMB_PATH)/1e6:.1f} MB)")

    print("building BM25 ...")
    BM25(docs=texts).save(BM25_PATH)
    print(f"  -> {BM25_PATH} ({os.path.getsize(BM25_PATH)/1e6:.1f} MB)")


# ------------------------------------------------------------------ retrieve
_cache = {}


def load_index():
    if _cache:
        return _cache
    from sentence_transformers import SentenceTransformer
    z = np.load(EMB_PATH, allow_pickle=True)
    _cache["emb"] = z["emb"]
    _cache["ids"] = list(z["ids"])
    src = RETRIEVAL_CHUNKS if os.path.exists(RETRIEVAL_CHUNKS) else CHUNKS
    _cache["chunks"] = {c["id"]: c for c in
                        (json.loads(l) for l in open(src, encoding="utf-8"))}
    _cache["bm25"] = BM25.load(BM25_PATH)
    _cache["model"] = SentenceTransformer(MODEL, device="cuda")
    return _cache


def search(query, k=4, k_rrf=60):
    """Reciprocal Rank Fusion of dense and BM25 rankings.

    RRF scores by 1/(k+rank) from each list rather than by raw score, which
    avoids having to normalise a cosine similarity against a BM25 score -- two
    quantities on entirely different scales.
    """
    ix = load_index()
    qv = ix["model"].encode([query], normalize_embeddings=True,
                            convert_to_numpy=True)[0]
    dense = ix["emb"] @ qv                       # cosine, embeddings are unit
    lex = ix["bm25"].scores(query)

    fused = defaultdict(float)
    for rank, i in enumerate(np.argsort(-dense)[:50]):
        fused[int(i)] += 1.0 / (k_rrf + rank)
    for rank, i in enumerate(np.argsort(-lex)[:50]):
        fused[int(i)] += 1.0 / (k_rrf + rank)

    top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    out = []
    for i, score in top:
        c = ix["chunks"][ix["ids"][i]]
        out.append({"id": c["id"], "source": c["source"], "title": c["title"],
                    "text": c["text"], "rrf": score,
                    "dense": float(dense[i]), "bm25": float(lex[i])})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query")
    ap.add_argument("-k", type=int, default=4)
    a = ap.parse_args()
    if a.query:
        for i, h in enumerate(search(a.query, a.k), 1):
            print(f"\n[{i}] {h['source']}  rrf={h['rrf']:.4f} "
                  f"dense={h['dense']:.3f} bm25={h['bm25']:.1f}")
            print("    " + " ".join(h["text"][:300].split()) + " ...")
    else:
        build()
