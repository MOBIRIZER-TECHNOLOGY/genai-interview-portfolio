"""
Chunking tests — every one of these is a regression test for a real bug.

The headline: `chunk_text` once produced **42x too many chunks** on real data,
mean length 178 characters against a 1400-character target. It ran for 50
minutes per shard and nothing looked wrong — chunks existed, contained text, and
embedded fine. It surfaced only as "this is taking suspiciously long".

`test_short_document_yields_one_chunk` reproduces it in under a millisecond.
That gap — 50 minutes of confusion versus one instant assertion — is the whole
argument for this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "07_rag_at_scale"))
sys.path.insert(0, str(ROOT / "01_rag_local"))

from build_index import chunk_text  # noqa: E402


TARGET = 1400
OVERLAP = 240


def spans_to_lengths(spans):
    return [b - a for a, b in spans]


# --------------------------------------------------------- the 42x bug


def test_short_document_yields_one_chunk():
    """THE regression test.

    A document shorter than the chunk size must produce exactly one chunk.

    The bug: once `end` reached len(text) the separator search was skipped, so
    `end` stopped moving while `pos = max(pos + 1, end - overlap)` advanced one
    character per iteration. A 500-char document produced ~180 near-identical
    chunks.
    """
    spans = chunk_text("x" * 500, TARGET, OVERLAP)
    assert len(spans) == 1, f"expected 1 chunk, got {len(spans)} (the 42x bug is back)"
    assert spans[0] == (0, 500)


@pytest.mark.parametrize("length", [61, 100, 250, 500, 900, 1399])
def test_any_document_under_target_is_one_chunk(length):
    """Sweep the whole sub-target range, not just one convenient value."""
    spans = chunk_text("x" * length, TARGET, OVERLAP)
    assert len(spans) == 1


def test_document_just_over_target_does_not_explode():
    """The boundary where the tail is shorter than the overlap.

    A 1500-char doc leaves a ~100-char tail after the first chunk -- shorter
    than the 240-char overlap. That is precisely the state where the old
    forward-progress guard failed.
    """
    spans = chunk_text("word " * 300, TARGET, OVERLAP)   # 1500 chars
    assert len(spans) <= 3, f"expected <=3 chunks for 1500 chars, got {len(spans)}"


def test_chunks_are_near_target_length_on_realistic_text():
    """Mean chunk length must be close to the target, not a fraction of it.

    Measured before the fix: 178 chars. After: 1209. This asserts the property
    rather than the exact number, so it survives tuning.
    """
    doc = ("The dispatch service assigns tasks to robots using a sealed-bid reverse "
           "auction that runs every one hundred and fifty milliseconds. ") * 60
    lengths = spans_to_lengths(chunk_text(doc, TARGET, OVERLAP))
    mean = sum(lengths) / len(lengths)
    assert mean > TARGET * 0.6, f"mean chunk {mean:.0f} chars, far below target {TARGET}"


def test_no_runaway_chunk_count():
    """Chunk count must stay within a small factor of text_len / stride."""
    doc = "word " * 4000                       # 20,000 chars
    stride = TARGET - OVERLAP
    spans = chunk_text(doc, TARGET, OVERLAP)
    upper = (len(doc) / stride) * 1.5
    assert len(spans) <= upper, f"{len(spans)} chunks exceeds sane bound {upper:.0f}"


# ------------------------------------------------------- basic contracts


def test_below_minimum_yields_nothing():
    assert chunk_text("", TARGET, OVERLAP) == []
    assert chunk_text("x" * 30, TARGET, OVERLAP) == []


def test_spans_are_ordered_and_in_bounds():
    doc = "word " * 2000
    spans = chunk_text(doc, TARGET, OVERLAP)
    for a, b in spans:
        assert 0 <= a < b <= len(doc)
    for (a1, _), (a2, _) in zip(spans, spans[1:]):
        assert a2 > a1, "chunk starts must strictly advance"


def test_full_coverage_of_document():
    """Every character must appear in at least one chunk -- no silent gaps."""
    doc = "word " * 2000
    spans = chunk_text(doc, TARGET, OVERLAP)
    covered = [False] * len(doc)
    for a, b in spans:
        for i in range(a, b):
            covered[i] = True
    # the final <60-char tail is deliberately dropped as a degenerate chunk
    assert all(covered[: len(doc) - 60])


def test_overlap_is_actually_applied():
    doc = "word " * 2000
    spans = chunk_text(doc, TARGET, OVERLAP)
    assert len(spans) > 1
    a2 = spans[1][0]
    b1 = spans[0][1]
    assert a2 < b1, "consecutive chunks must overlap"


def test_terminates_on_pathological_input():
    """No separators at all, and a repeated single character.

    Both are inputs where a boundary-search bug turns into an infinite loop.
    `chunk_text` must always terminate, so a plain call with a timeout-free
    assertion is the test.
    """
    for doc in ("x" * 5000, "\n" * 5000, "." * 5000):
        spans = chunk_text(doc, TARGET, OVERLAP)
        assert len(spans) < 100


# ------------------------------------------ project 01's markdown chunker


def test_markdown_chunker_prefixes_breadcrumb(tmp_path):
    """Project 01 chunks must carry their heading breadcrumb into the embedding.

    That prefix is what makes a bare table row retrievable; losing it is a
    silent quality regression with no error.
    """
    from rag.chunking import chunk_markdown

    doc = tmp_path / "sample.md"
    doc.write_text(
        "# Atlas Overview\n\n"
        "Intro paragraph about the platform.\n\n"
        "## Severity ladder\n\n"
        "| SEV1 | robots halted | 5 min |\n",
        encoding="utf-8",
    )
    chunks = chunk_markdown(doc, max_tokens=320, overlap_tokens=60)
    assert chunks
    for c in chunks:
        assert c.text.startswith(c.breadcrumb), "embedded text must lead with the breadcrumb"
        assert "sample.md" in c.breadcrumb
    assert any("Severity ladder" in c.breadcrumb for c in chunks)
