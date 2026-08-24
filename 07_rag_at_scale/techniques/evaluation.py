"""
RAGAS-style evaluation: judge the answer, not just the retrieval.

    from techniques.evaluation import evaluate_answer, evaluate_batch

Project 01 measures recall@k, MRR, citation validity and abstention. Those are
cheap, deterministic, and they cover retrieval well. What they cannot see is
whether each *sentence of the answer* is actually supported by the context —
a model can cite block [2] correctly and still assert something block [2] never
said.

That needs a judge. These four metrics are the RAGAS set, and each one isolates a
different failure:

| metric | question it answers | what a low score means |
|---|---|---|
| **faithfulness** | is every claim supported by the retrieved context? | the model is inventing; fix the prompt or the model |
| **answer relevance** | does the answer address the question asked? | the model drifted; fix the prompt |
| **context precision** | is the retrieved context mostly relevant? | retrieving junk; fix reranking or top_k |
| **context recall** | does the context contain everything needed? | retrieval missed; fix chunking or the retriever |

The diagnostic value is in the **pairs**. Low faithfulness with high context
recall means the model is ignoring good context. Low faithfulness with low
context recall means retrieval starved it and the model guessed. Same symptom,
opposite fix.

## The honest caveat about LLM judges

An LLM-as-judge is a measurement instrument with its own error rate. It is
biased toward verbose answers, it is inconsistent near the boundary, and it
cannot exceed its own competence — a 7B judge cannot reliably grade claims it
does not understand. Treat these as *relative* signals for comparing two systems,
not absolute truth. Where a deterministic metric exists (project 01's mechanical
citation verification), prefer it: it is free and it does not hallucinate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import httpx

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"


def _judge(prompt: str, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
           max_tokens: int = 500) -> str:
    r = httpx.post(f"{url}/api/chat", timeout=180, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # temperature 0: a judge that returns different scores for the same input
        # is not a measurement instrument.
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _extract_json(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if m:
        text = m.group(1)
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        obj = json.loads(text[a:b + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------- faithfulness

FAITHFULNESS_PROMPT = """You are verifying whether an answer is supported by its context.

CONTEXT:
{context}

ANSWER:
{answer}

Break the answer into individual factual claims. For each claim decide whether the
CONTEXT supports it. A claim is supported only if the context states it or directly
implies it -- general world knowledge does NOT count as support.

Reply with JSON only:
{{"claims": [{{"claim": "...", "supported": true/false, "why": "..."}}]}}"""


ANSWER_RELEVANCE_PROMPT = """Given this answer, write the {n} questions it most directly answers.
Do not use outside knowledge; base them only on the answer text.
Output one question per line, no numbering.

ANSWER:
{answer}"""


CONTEXT_PRECISION_PROMPT = """For each numbered context block, decide whether it was USEFUL
for answering the question. A block is useful only if it contributes information the
answer needs -- being on the same general topic is not enough.

QUESTION: {question}

{blocks}

Reply with JSON only:
{{"verdicts": [{{"block": 1, "useful": true/false}}, ...]}}"""


CONTEXT_RECALL_PROMPT = """Compare a reference answer against the retrieved context.

QUESTION: {question}

CONTEXT:
{context}

REFERENCE ANSWER:
{reference}

For each sentence of the reference answer, decide whether the CONTEXT contains the
information needed to produce it.

Reply with JSON only:
{{"sentences": [{{"sentence": "...", "in_context": true/false}}]}}"""


@dataclass
class RagasScores:
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    n_claims: int = 0
    unsupported_claims: list[str] = None
    detail: dict = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def overall(self) -> float:
        vals = [v for v in (self.faithfulness, self.answer_relevance,
                            self.context_precision, self.context_recall)
                if v is not None]
        return sum(vals) / len(vals) if vals else 0.0


def faithfulness(answer: str, contexts: list[str], model: str = DEFAULT_MODEL) -> tuple[float, list[str], int]:
    """Fraction of the answer's claims that the context supports."""
    ctx = "\n\n---\n\n".join(contexts)
    out = _extract_json(_judge(FAITHFULNESS_PROMPT.format(context=ctx, answer=answer), model))
    if not out or "claims" not in out or not out["claims"]:
        return 0.0, [], 0
    claims = out["claims"]
    unsupported = [c.get("claim", "") for c in claims if not c.get("supported")]
    return (len(claims) - len(unsupported)) / len(claims), unsupported, len(claims)


def answer_relevance(question: str, answer: str, embedder, model: str = DEFAULT_MODEL,
                     n: int = 3) -> float:
    """Reverse-engineer questions from the answer; compare to the real one.

    If the answer actually addresses the question, the questions it *implies*
    should be close to the question asked. This catches an answer that is
    faithful to the context but about something else entirely.
    """
    import numpy as np

    out = _judge(ANSWER_RELEVANCE_PROMPT.format(answer=answer, n=n), model)
    generated = [re.sub(r"^\s*[-*\d.)]+\s*", "", l).strip()
                 for l in out.splitlines() if l.strip()][:n]
    if not generated:
        return 0.0
    vecs = embedder.encode([question] + generated, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    return float(np.mean(vecs[0] @ vecs[1:].T))


def context_precision(question: str, contexts: list[str], model: str = DEFAULT_MODEL) -> float:
    """Fraction of retrieved blocks that were actually useful."""
    blocks = "\n\n".join(f"[{i}] {c[:900]}" for i, c in enumerate(contexts, 1))
    out = _extract_json(_judge(
        CONTEXT_PRECISION_PROMPT.format(question=question, blocks=blocks), model))
    if not out or "verdicts" not in out or not out["verdicts"]:
        return 0.0
    v = out["verdicts"]
    return sum(1 for x in v if x.get("useful")) / len(v)


def context_recall(question: str, contexts: list[str], reference: str,
                   model: str = DEFAULT_MODEL) -> float:
    """Fraction of the reference answer that the context could support."""
    ctx = "\n\n---\n\n".join(contexts)
    out = _extract_json(_judge(CONTEXT_RECALL_PROMPT.format(
        question=question, context=ctx, reference=reference), model))
    if not out or "sentences" not in out or not out["sentences"]:
        return 0.0
    s = out["sentences"]
    return sum(1 for x in s if x.get("in_context")) / len(s)


def evaluate_answer(question: str, answer: str, contexts: list[str],
                    reference: str | None = None, embedder=None,
                    model: str = DEFAULT_MODEL) -> RagasScores:
    """All applicable metrics for one (question, answer, context) triple.

    `reference` is optional -- context_recall needs a gold answer, the other
    three are reference-free, which is what makes them usable on production
    traffic where you have no labels.
    """
    f, unsupported, n_claims = faithfulness(answer, contexts, model)
    scores = RagasScores(
        faithfulness=f,
        context_precision=context_precision(question, contexts, model),
        n_claims=n_claims,
        unsupported_claims=unsupported,
        detail={},
    )
    if embedder is not None:
        scores.answer_relevance = answer_relevance(question, answer, embedder, model)
    if reference:
        scores.context_recall = context_recall(question, contexts, reference, model)
    return scores


def evaluate_batch(rows: list[dict], embedder=None, model: str = DEFAULT_MODEL,
                   verbose: bool = True) -> dict:
    """rows: [{question, answer, contexts, reference?}, ...]"""
    out = []
    for i, r in enumerate(rows, 1):
        s = evaluate_answer(r["question"], r["answer"], r["contexts"],
                            r.get("reference"), embedder, model)
        out.append({"question": r["question"], **s.to_dict()})
        if verbose:
            print(f"  [{i}/{len(rows)}] faith {s.faithfulness:.2f} "
                  f"prec {s.context_precision:.2f} "
                  f"{'rel %.2f ' % s.answer_relevance if s.answer_relevance is not None else ''}"
                  f"{'rec %.2f ' % s.context_recall if s.context_recall is not None else ''}"
                  f"| {r['question'][:48]}")
            for u in (s.unsupported_claims or [])[:2]:
                print(f"        UNSUPPORTED: {u[:88]}")

    def mean(key: str) -> float | None:
        vals = [r[key] for r in out if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": len(out),
        "faithfulness": mean("faithfulness"),
        "answer_relevance": mean("answer_relevance"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "rows": out,
    }
