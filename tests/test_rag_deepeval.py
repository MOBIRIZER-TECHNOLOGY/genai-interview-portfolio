"""
DeepEval quality gates for the RAG pipeline (project 01).

    pytest tests/test_rag_deepeval.py -m llm -v

These are **regression gates**, not benchmarks. `01_rag_local/eval/evaluate.py`
prints numbers; these assert thresholds and fail the build when answer quality
drops. That distinction is the reason this file exists.

## What each metric catches

| metric | fails when |
|---|---|
| **Faithfulness** | the answer asserts things the retrieved context does not support |
| **AnswerRelevancy** | the answer is true but does not address the question |
| **ContextualPrecision** | retrieval returned mostly noise |
| **ContextualRecall** | retrieval missed information the answer needed |
| **Hallucination** | the answer contradicts the provided context |

The diagnostic value is in the **pairs**: low faithfulness with *high* contextual
recall means the model ignored good context (fix the prompt); with *low* recall
it means retrieval starved it (fix retrieval). Same symptom, opposite fix.

## Two deliberate choices

**Local judge.** `OllamaJudge` runs `qwen2.5:7b` at temperature 0. No API key, no
per-test cost, nothing leaves the machine — consistent with every other model
here. See `tests/ollama_judge.py`.

**Thresholds below what the system currently scores.** The measured pipeline hits
100% fact recall and 100% citation validity on its own eval set, but these gates
sit at 0.6-0.7. A gate pinned to today's exact score fails on harmless variance
and gets disabled within a week. A gate set where *real* degradation lives keeps
working.

## The honest caveat

An LLM judge is an instrument with its own error rate: biased toward verbose
answers, inconsistent near boundaries, and capped by its own competence — a 7B
judge cannot grade claims it does not understand. Use these to compare runs of
your own system, not to certify absolute quality. Where a deterministic check
exists, prefer it: project 01's mechanical citation verification is free, exact,
and cannot hallucinate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_rag_local"))
sys.path.insert(0, str(ROOT / "tests"))

from ollama_judge import OllamaJudge, ollama_available  # noqa: E402

INDEX = ROOT / "01_rag_local" / "index"

pytestmark = [
    pytest.mark.llm,
    pytest.mark.slow,
    pytest.mark.skipif(not ollama_available(), reason="Ollama or the judge model is unavailable"),
    pytest.mark.skipif(not INDEX.exists(), reason="no RAG index; run 01_rag_local/ingest.py"),
]


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def judge():
    return OllamaJudge()


@pytest.fixture(scope="module")
def pipeline():
    from rag.pipeline import RagPipeline

    return RagPipeline.load(INDEX)


@pytest.fixture(scope="module")
def answered(pipeline):
    """Run each question once and reuse across metrics.

    Module-scoped on purpose: generation is the expensive part, and every metric
    below should judge the *same* output. Re-running per metric would also mean
    judging different text each time, which makes the scores incomparable.
    """
    questions = [
        "What is the Rotterdam rule?",
        "What does error code TLM-330 mean and what should I do?",
        "What barcode confidence threshold triggers a retry?",
        "How long are vision frames retained, and why that long?",
    ]
    out = {}
    for q in questions:
        r = pipeline.ask(q)
        out[q] = {
            "answer": r.answer.text,
            "contexts": [h.body for h in r.hits],
            "grounded": r.answer.grounded,
            "abstained": r.answer.abstained,
        }
    return out


def _case(q, rec):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(input=q, actual_output=rec["answer"], retrieval_context=rec["contexts"])


# ------------------------------------------------------------ the gates


def test_faithfulness(answered, judge):
    """No claim may go beyond what the retrieved context supports."""
    from deepeval import assert_test
    from deepeval.metrics import FaithfulnessMetric

    metric = FaithfulnessMetric(threshold=0.7, model=judge, async_mode=False,
                                include_reason=True)
    for q, rec in answered.items():
        assert_test(_case(q, rec), [metric])


def test_answer_relevancy(answered, judge):
    """The answer must address the question actually asked."""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric

    metric = AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False)
    for q, rec in answered.items():
        assert_test(_case(q, rec), [metric])


def test_contextual_precision(answered, judge):
    """Retrieved blocks should be mostly useful, not topically adjacent."""
    from deepeval import assert_test
    from deepeval.metrics import ContextualPrecisionMetric
    from deepeval.test_case import LLMTestCase

    metric = ContextualPrecisionMetric(threshold=0.6, model=judge, async_mode=False)
    expected = {
        "What is the Rotterdam rule?":
            "A task losing 12 consecutive auctions has its priority multiplied by 1.5 "
            "and is pinned to the next round.",
        "What barcode confidence threshold triggers a retry?":
            "Barcode reads below 0.92 confidence are retried up to 3 times.",
    }
    for q, gold in expected.items():
        rec = answered[q]
        assert_test(
            LLMTestCase(input=q, actual_output=rec["answer"], expected_output=gold,
                        retrieval_context=rec["contexts"]),
            [metric],
        )


def test_hallucination_against_context(answered, judge):
    """The answer must not contradict the context it was given.

    HallucinationMetric uses `context` (ground truth) rather than
    `retrieval_context`, and it is scored INVERSELY -- lower is better -- so the
    threshold is an upper bound.
    """
    from deepeval import assert_test
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase

    metric = HallucinationMetric(threshold=0.4, model=judge, async_mode=False)
    for q, rec in answered.items():
        assert_test(
            LLMTestCase(input=q, actual_output=rec["answer"], context=rec["contexts"]),
            [metric],
        )


def test_geval_citation_discipline(answered, judge):
    """Custom G-Eval rubric for the behaviour this pipeline is built around.

    Off-the-shelf metrics do not know that this system is *supposed* to cite
    block numbers and abstain when unsupported. G-Eval lets you grade the
    property you actually care about instead of a generic proxy.
    """
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    metric = GEval(
        name="CitationDiscipline",
        criteria=(
            "Determine whether every factual claim in the output is attributed to a "
            "numbered context block using [n] notation, and whether numbers, error "
            "codes and identifiers are quoted exactly rather than paraphrased. "
            "An answer that states facts without any [n] citation should score low."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6,
        model=judge,
        async_mode=False,
    )
    for q, rec in answered.items():
        assert_test(_case(q, rec), [metric])


# ------------------------------------------------- deterministic guards
# Cheap, exact, no judge. These would catch a real regression even if every
# LLM-based gate above were disabled.


def test_grounded_or_abstained(answered):
    """Every answer either cites only real blocks, or abstains. No middle ground."""
    for q, rec in answered.items():
        assert rec["grounded"], f"ungrounded answer for {q!r}: {rec['answer'][:120]}"


def test_abstains_on_unanswerable(pipeline):
    """The failure mode that matters most: inventing an answer, fluently.

    Deterministic and free -- no judge involved.
    """
    for q in ("What is the Atlas engineering salary band?",
              "How do I configure the Atlas Kafka consumer group offsets?"):
        r = pipeline.ask(q)
        assert r.answer.abstained, f"should have abstained on {q!r}, said: {r.answer.text[:140]}"


def test_no_hallucinated_citation_numbers(answered):
    """A [7] when only 4 blocks were sent is a citation that cannot be verified."""
    for q, rec in answered.items():
        assert rec["grounded"] or rec["abstained"]
