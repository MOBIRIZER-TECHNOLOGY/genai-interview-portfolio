"""
DeepEval quality gates for the RAG pipeline (project 01).

    pytest tests/test_rag_deepeval.py -m llm -v

These are **regression gates**, not benchmarks. `01_rag_local/eval/evaluate.py`
prints numbers; these assert thresholds and fail the build when quality drops.

## What this layer earned its place by finding

A real hallucination that the deterministic checks structurally could not catch:

    answer:  "Vision frames are retained for 14 days hot and
              indefinitely cold [1]."
    corpus:  | Vision frames | 14 days | none | 14 days |

Cold storage is **none**. The model invented indefinite cold retention — and
cited block `[1]`, which genuinely *was* sent. So `verify_citations()` passed it.
Mechanical verification confirms a citation points at a real block; only a judge
can tell you the sentence misrepresents that block.

## Calibrate the instrument before trusting it

An LLM judge has its own error rate, and with a local `qwen2.5:7b` that rate is
high enough to matter. Measured on the real pipeline:

| question | faithfulness | relevancy | G-Eval citation |
|---|---:|---:|---:|
| Rotterdam rule (correct, cited `[2]`) | 1.00 | **0.00** | **0.00** |
| TLM-330 (correct, cited `[1]`) | 0.50 | 0.67 | 0.90 |
| barcode 0.92 (correct, cited `[1]`) | 1.00 | 1.00 | **0.00** |
| vision frames (**hallucinated**) | **0.33** | 0.25 | 1.00 |

Faithfulness tracks reality: 1.00 on two correct answers, 0.33 on the real
defect. The others do not — relevancy scored **0.00** on a direct, correct,
cited answer, and the G-Eval citation metric scored 0.00 on answers containing
`[2]` and `[1]` while scoring 1.00 on another that also contains `[1]`.
Self-contradictory, so it measures nothing.

So this file:

1. **Gates on faithfulness only**, after a calibration test proves the judge
   separates a known-good from a known-bad answer.
2. **Reports relevancy and contextual precision as advisory** — visible, never
   failing. A gate that fails a perfect answer gets switched off within a week,
   taking the useful gates with it.
3. **Dropped the G-Eval citation metric entirely.** That property is already
   verified exactly and for free by `verify_citations()`. An unreliable LLM gate
   over a mechanically checkable property is strictly worse than the mechanical
   check alone.

To promote the advisory metrics, use a stronger judge
(`DEEPEVAL_JUDGE=llama3.1:70b`, or a frontier API model) and re-run calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_rag_local"))
sys.path.insert(0, str(ROOT / "tests"))

from ollama_judge import make_judge, ollama_available  # noqa: E402

INDEX = ROOT / "01_rag_local" / "index"

pytestmark = [
    pytest.mark.llm,
    pytest.mark.slow,
    pytest.mark.skipif(not ollama_available(), reason="Ollama or the judge model is unavailable"),
    pytest.mark.skipif(not INDEX.exists(), reason="no RAG index; run 01_rag_local/ingest.py"),
]

QUESTIONS = [
    "What is the Rotterdam rule?",
    "What does error code TLM-330 mean and what should I do?",
    "What barcode confidence threshold triggers a retry?",
    "How long are vision frames retained, and why that long?",
]


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def judge():
    return make_judge()


@pytest.fixture(scope="module")
def pipeline():
    from rag.pipeline import RagPipeline

    return RagPipeline.load(INDEX)


@pytest.fixture(scope="module")
def answered(pipeline):
    """Ask each question once and reuse across metrics.

    Module-scoped deliberately: generation is the expensive part, and every
    metric must judge the *same* output. Re-generating per metric would make the
    scores incomparable.
    """
    out = {}
    for q in QUESTIONS:
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

    return LLMTestCase(input=q, actual_output=rec["answer"],
                       retrieval_context=rec["contexts"])


# ------------------------------------------------- calibrate the judge


CALIB_CONTEXT = [
    "Barcode reads below 0.92 confidence are re-attempted up to 3 times with a "
    "different exposure. After 3 failures the pallet is routed to the manual "
    "inspection lane."
]
CALIB_Q = "What barcode confidence threshold triggers a retry?"
CALIB_GOOD = "Reads below 0.92 confidence are retried up to 3 times [1]."
CALIB_BAD = ("Reads below 0.55 confidence are retried up to 12 times, and the robot "
             "is then decommissioned by the night shift.")


def calibration_gap(metric_factory) -> float:
    """How far a metric separates a known-good answer from a known-bad one.

    An LLM judge is a measurement instrument and instruments need calibration. A
    metric scoring both the same is measuring nothing, and gating on it produces
    noise -- noisy gates get disabled, taking the good gates with them.
    """
    from deepeval.test_case import LLMTestCase

    m = metric_factory()
    m.measure(LLMTestCase(input=CALIB_Q, actual_output=CALIB_GOOD,
                          retrieval_context=CALIB_CONTEXT))
    good = m.score
    m.measure(LLMTestCase(input=CALIB_Q, actual_output=CALIB_BAD,
                          retrieval_context=CALIB_CONTEXT))
    return good - m.score


def test_judge_is_calibrated_for_faithfulness(judge):
    """Fail loudly if the instrument we gate on has stopped discriminating.

    Runs before the gate. If this fails, the gate's verdicts are meaningless and
    the judge is what needs fixing -- not the gate.
    """
    from deepeval.metrics import FaithfulnessMetric

    gap = calibration_gap(
        lambda: FaithfulnessMetric(threshold=0.7, model=judge, async_mode=False))
    assert gap >= 0.5, (
        f"judge separates good from bad by only {gap:.2f}; it cannot be trusted "
        "to gate faithfulness. Use a stronger DEEPEVAL_JUDGE.")


# ------------------------------------------------------------- the gate


def test_faithfulness(answered, judge):
    """No claim may exceed what the retrieved context supports.

    The one metric this judge grades usefully, and the one that justifies the
    whole layer: it caught the "indefinitely cold" hallucination, which carried
    a valid citation and therefore passed every deterministic check.

    Two judge failure modes measured on verified-correct answers, both scoring
    0.50 when they should score high:

    - **partial context use.** The TLM-330 answer is fully supported by block
      [1], but blocks [2]-[4] are retrieval noise about other topics. The judge
      penalised the answer for correctly ignoring them.
    - **negation.** After the prompt fix the answer says vision frames are
      "not retained cold"; the judge read that as "retained cold" and called it
      a contradiction.

    Both are limitations of a 7B judge, not of the pipeline -- which is why the
    threshold below is derived from measured separation.
    """
    from deepeval.metrics import FaithfulnessMetric

    # Threshold DERIVED from measurement, not aspiration.
    #
    #   known hallucination ("indefinitely cold")   0.33
    #   known-correct answers                       0.50 - 1.00
    #
    # 0.7 produced 2 false positives out of 4 on verified-correct answers, and a
    # gate that fails correct work gets switched off. 0.40 sits inside the only
    # separation this judge actually demonstrates. The margin is thin (0.17) and
    # that is a statement about the 7B judge, not about the pipeline -- a
    # stronger DEEPEVAL_JUDGE would support a stricter gate.
    FAITHFULNESS_FLOOR = 0.40

    metric = FaithfulnessMetric(threshold=FAITHFULNESS_FLOOR, model=judge,
                                async_mode=False, include_reason=True)
    failures = []
    for q, rec in answered.items():
        metric.measure(_case(q, rec))
        if metric.score < FAITHFULNESS_FLOOR:
            failures.append(
                f"{q!r} scored {metric.score:.2f} -- {str(metric.reason)[:200]}")

    assert not failures, "unfaithful answers:\n  " + "\n  ".join(failures)


# --------------------------------------------------------- advisory only


@pytest.mark.advisory
def test_answer_relevancy_advisory(answered, judge, capsys):
    """Reported, never gated: measured 0.00 on a correct, cited answer."""
    from deepeval.metrics import AnswerRelevancyMetric

    def factory():
        return AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False)

    metric = factory()
    rows = []
    for q, rec in answered.items():
        metric.measure(_case(q, rec))
        rows.append((metric.score, q))

    with capsys.disabled():
        print("\n  answer relevancy (ADVISORY, not gated):")
        for score, q in rows:
            print(f"    {score:.2f}  {q[:58]}")
        print(f"    calibration gap: {calibration_gap(factory):+.2f} "
              f"(needs >= 0.50 to become a gate)")


@pytest.mark.advisory
def test_contextual_precision_advisory(answered, judge, capsys):
    """Retrieval precision, reported not gated."""
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
    rows = []
    for q, gold in expected.items():
        rec = answered[q]
        metric.measure(LLMTestCase(input=q, actual_output=rec["answer"],
                                   expected_output=gold,
                                   retrieval_context=rec["contexts"]))
        rows.append((metric.score, q))

    with capsys.disabled():
        print("\n  contextual precision (ADVISORY, not gated):")
        for score, q in rows:
            print(f"    {score:.2f}  {q[:58]}")


# NOTE: the G-Eval "citation discipline" metric was REMOVED, deliberately.
#
# It scored 0.00 on answers visibly containing [2] and [1], while scoring 1.00 on
# another answer that also contains [1] -- self-contradictory, so it measures
# nothing. The property it graded is already checked exactly and for free by
# verify_citations() in test_crash_safety.py.
#
# Where a deterministic check exists, prefer it.


# ------------------------------------------------- deterministic guards
# Cheap, exact, no judge. These would catch a real regression even if every
# LLM-based check above were disabled.


def test_grounded_or_abstained(answered):
    """Every answer either cites only real blocks, or abstains."""
    for q, rec in answered.items():
        assert rec["grounded"], f"ungrounded answer for {q!r}: {rec['answer'][:120]}"


def test_abstains_on_unanswerable(pipeline):
    """The failure mode that matters most: inventing an answer, fluently."""
    for q in ("What is the Atlas engineering salary band?",
              "How do I configure the Atlas Kafka consumer group offsets?"):
        r = pipeline.ask(q)
        assert r.answer.abstained, (
            f"should have abstained on {q!r}, said: {r.answer.text[:140]}")


def test_no_invented_cold_retention(answered):
    """Deterministic regression test for the hallucination DeepEval found.

    The corpus says vision-frame cold storage is "none" (deleted after 14 days).
    The model claimed "indefinitely cold" -- with a valid [1] citation, so every
    mechanical check passed it. Fixed by a system-prompt rule forbidding the
    model from turning "none" into a duration.

    Pinned here deterministically: now that the specific failure is known, it
    costs nothing to check exactly and needs no judge.
    """
    ans = answered["How long are vision frames retained, and why that long?"]["answer"].lower()
    assert "14 day" in ans, "the real retention period must be stated"
    for invented in ("indefinite", "indefinitely", "7 year", "10 year"):
        assert invented not in ans, (
            f"invented cold-retention claim {invented!r} is back: {ans[:160]}")


def test_no_hallucinated_citation_numbers(answered):
    """A [7] when only 4 blocks were sent is a citation that cannot be verified."""
    for q, rec in answered.items():
        assert rec["grounded"] or rec["abstained"]
