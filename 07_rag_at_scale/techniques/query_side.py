"""
Query-side techniques: rewrite the question before you search.

Every one of these exists because of the same problem: **the user's question and
the answering passage often share no vocabulary.** A short question sits in a
different region of embedding space than a long expository paragraph, and a
question that contains three sub-questions matches none of them well.

    from techniques.query_side import hyde, multi_query, decompose, route

All of these cost an LLM call before you can search. That is a real latency and
money cost — typically 200-800 ms with a local 7B — and the honest framing is
that they are worth it when retrieval is the bottleneck and not otherwise.
Measure retrieval recall first; if it is already 0.95 these will not help you.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"


def _chat(prompt: str, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
          temperature: float = 0.0, max_tokens: int = 400) -> str:
    r = httpx.post(f"{url}/api/chat", timeout=120, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ---------------------------------------------------------------- HyDE


HYDE_PROMPT = """Write a short factual passage (3-4 sentences) that would answer this question.
Write it as if it were an excerpt from a technical document or encyclopedia.
Do not say "I don't know" -- if you are unsure, write a plausible passage anyway.
Do not preface it or explain. Output only the passage.

Question: {question}"""


@dataclass
class HydeResult:
    question: str
    hypothetical: str
    seconds: float


def hyde(question: str, model: str = DEFAULT_MODEL, **kw) -> HydeResult:
    """Hypothetical Document Embeddings.

    Instead of embedding the *question*, ask the LLM to hallucinate an *answer*
    and embed that. The generated passage is often factually wrong -- that is
    fine and not the point. It is wrong in the right *shape*: it has the
    vocabulary, register and length of a real answering passage, so it lands
    much closer to the true answer in embedding space than the question did.

    You are trading a question-to-passage similarity (asymmetric, hard) for a
    passage-to-passage similarity (symmetric, easy).

    When it fails: highly specific factual lookups where the model invents a
    confident wrong entity and drags retrieval toward it. `TLM-330` is a good
    example -- the model has no idea, so its hypothetical is noise. Use HyDE for
    conceptual questions, keep BM25 in the mix for identifier lookups.
    """
    import time

    t0 = time.perf_counter()
    doc = _chat(HYDE_PROMPT.format(question=question), model=model, **kw)
    return HydeResult(question, doc, time.perf_counter() - t0)


# --------------------------------------------------------- multi-query


MULTI_PROMPT = """Generate {n} different search queries that would find documents answering this question.
Vary the phrasing, vocabulary and specificity -- use synonyms and related technical terms.
Output one query per line, no numbering, no explanation.

Question: {question}"""


@dataclass
class MultiQueryResult:
    original: str
    variants: list[str]
    seconds: float


def multi_query(question: str, n: int = 4, model: str = DEFAULT_MODEL,
                **kw) -> MultiQueryResult:
    """Generate several paraphrases and union their results.

    A single embedding is one point in space. If the phrasing is unlucky, that
    point sits away from the answer and there is no recovery. Several paraphrases
    give several points; fusing their result lists (RRF, as in project 01) means
    any one of them finding the answer is enough.

    This is a **recall** technique and it costs precision -- you are pulling in
    more candidates, some irrelevant. It pairs naturally with a reranker, which
    is what restores precision afterwards.
    """
    import time

    t0 = time.perf_counter()
    out = _chat(MULTI_PROMPT.format(n=n, question=question), model=model, **kw)
    variants = [
        re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        for line in out.splitlines() if line.strip()
    ]
    variants = [v for v in variants if len(v) > 3][:n]
    if question not in variants:
        variants.insert(0, question)      # never lose the original phrasing
    return MultiQueryResult(question, variants, time.perf_counter() - t0)


# --------------------------------------------------------- decomposition


DECOMPOSE_PROMPT = """Break this question into the minimal set of independent sub-questions
that must each be answered to answer the whole thing.
If the question is already simple and needs no decomposition, output it unchanged as a single line.
Output one sub-question per line, no numbering, no explanation.

Question: {question}"""


@dataclass
class Decomposition:
    question: str
    sub_questions: list[str]
    is_compound: bool
    seconds: float


def decompose(question: str, model: str = DEFAULT_MODEL, **kw) -> Decomposition:
    """Split a multi-hop question into independently answerable parts.

    "How does the auction bid formula relate to the SEV2 threshold?" needs two
    different documents. Embedding it as one query retrieves passages that are
    mediocre for both halves rather than good for either -- the embedding is an
    average of two topics and points at neither.

    Note this only handles *parallel* decomposition. Genuinely sequential
    multi-hop ("who wrote the paper that introduced the technique used by X")
    needs iterative retrieve-then-reformulate, which is an agent loop, not a
    single rewrite.
    """
    import time

    t0 = time.perf_counter()
    out = _chat(DECOMPOSE_PROMPT.format(question=question), model=model, **kw)
    subs = [
        re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        for line in out.splitlines() if line.strip()
    ]
    subs = [s for s in subs if len(s) > 5]
    return Decomposition(question, subs or [question], len(subs) > 1,
                         time.perf_counter() - t0)


# --------------------------------------------------------------- routing


ROUTE_PROMPT = """Classify this query into exactly one category.

lookup      - asks for a specific value, code, identifier or threshold (e.g. "what does TLM-330 mean")
conceptual  - asks how or why something works, needs explanation
comparison  - asks to relate or contrast two or more things
unanswerable- small talk, opinion, or clearly outside a technical documentation corpus

Output only the category word.

Query: {question}"""

STRATEGY = {
    "lookup": {
        "retrieval": "bm25-weighted hybrid",
        "hyde": False,
        "multi_query": False,
        "top_k": 3,
        "why": "exact rare tokens; BM25's IDF beats dense here and HyDE would hallucinate a wrong code",
    },
    "conceptual": {
        "retrieval": "dense-weighted hybrid",
        "hyde": True,
        "multi_query": True,
        "top_k": 6,
        "why": "paraphrase-heavy; HyDE closes the question-to-passage gap",
    },
    "comparison": {
        "retrieval": "hybrid",
        "hyde": False,
        "multi_query": True,
        "top_k": 8,
        "why": "needs coverage of several entities; decompose first, widen top_k",
    },
    "unanswerable": {
        "retrieval": "none",
        "hyde": False,
        "multi_query": False,
        "top_k": 0,
        "why": "abstain without spending a retrieval or a generation",
    },
}


@dataclass
class Route:
    question: str
    category: str
    strategy: dict = field(default_factory=dict)
    seconds: float = 0.0


def route(question: str, model: str = DEFAULT_MODEL, **kw) -> Route:
    """Pick a retrieval strategy per query instead of using one for everything.

    The techniques above are not free and they are not universally good: HyDE
    helps conceptual questions and actively *hurts* identifier lookups. A router
    turns that from a fixed trade-off into a per-query decision.

    The honest caveat: this adds an LLM call to every query, on the critical
    path, to save work later. At high QPS you would distil this classifier into
    something tiny -- a fine-tuned 0.5B (project 02's exact pattern) or even
    logistic regression on the query embedding, which is a millisecond.
    """
    import time

    t0 = time.perf_counter()
    raw = _chat(ROUTE_PROMPT.format(question=question), model=model, max_tokens=10, **kw)
    cat = next((c for c in STRATEGY if c in raw.lower()), "conceptual")
    return Route(question, cat, STRATEGY[cat], time.perf_counter() - t0)


# ------------------------------------------------------------ orchestration


def prepare_queries(question: str, model: str = DEFAULT_MODEL) -> dict:
    """Route, then apply whichever rewrites that route calls for.

    Returns everything needed to run retrieval, including the per-stage timings
    so you can see what the rewriting actually cost.
    """
    r = route(question, model=model)
    queries = [question]
    timings = {"route_s": r.seconds}
    extras: dict = {}

    if r.category == "comparison":
        d = decompose(question, model=model)
        timings["decompose_s"] = d.seconds
        extras["sub_questions"] = d.sub_questions
        queries = d.sub_questions

    if r.strategy.get("multi_query"):
        mq = multi_query(question, model=model)
        timings["multi_query_s"] = mq.seconds
        queries = list(dict.fromkeys(queries + mq.variants))

    if r.strategy.get("hyde"):
        hy = hyde(question, model=model)
        timings["hyde_s"] = hy.seconds
        extras["hypothetical"] = hy.hypothetical
        queries.append(hy.hypothetical)

    return {
        "question": question,
        "category": r.category,
        "strategy": r.strategy,
        "queries": queries,
        "timings": timings,
        "total_rewrite_s": round(sum(timings.values()), 3),
        **extras,
    }
