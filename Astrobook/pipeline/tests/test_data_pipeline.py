"""
Invariants over the built dataset. These are the silent-failure guards: none of
them would throw during a run, they would just quietly degrade the adapter.

    python pipeline/tests/test_data_pipeline.py

Skips gracefully if build/ artifacts are absent.
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from config import CHUNKS, PAIRS, split_path, chunk_cid
from qc import sanitize_question, acceptable


def _load(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# ----------------------------------------------------------------- chunking
def test_no_chunk_exceeds_cap():
    rows = _load(CHUNKS)
    if not rows:
        print("      (skip: no chunks.jsonl)"); return
    HARD_MAX = 2600
    over = [r["id"] for r in rows if r["approx_tokens"] > HARD_MAX + 50]
    assert not over, f"{len(over)} chunks over cap, e.g. {over[:3]}"


def test_few_chunks_pinned_at_cap():
    """The 54% bug: splitter finds no boundaries and slices mid-sentence."""
    rows = _load(CHUNKS)
    if not rows:
        print("      (skip)"); return
    pinned = sum(1 for r in rows if r["approx_tokens"] >= 2560)
    frac = pinned / len(rows)
    assert frac < 0.05, f"{100*frac:.1f}% pinned at cap -- splitter cascade failed"


def test_few_chunks_start_mid_sentence():
    rows = _load(CHUNKS)
    if not rows:
        print("      (skip)"); return
    low = sum(1 for r in rows if r["text"][:1].islower())
    frac = low / len(rows)
    assert frac < 0.03, f"{100*frac:.1f}% start lowercase -- bogus unit boundaries"


def test_no_private_use_glyphs():
    """Broken subsetted fonts emit PUA codepoints that become training noise."""
    rows = _load(CHUNKS)
    if not rows:
        print("      (skip)"); return
    bad = [r["id"] for r in rows if re.search(r"[-]", r["text"])]
    assert not bad, f"{len(bad)} chunks contain PUA glyphs, e.g. {bad[:3]}"


# ------------------------------------------------------------------ custom id
def test_custom_id_is_injective_and_valid():
    """Batch custom_id must round-trip 1:1 and satisfy the API charset."""
    rows = _load(CHUNKS)
    if not rows:
        print("      (skip)"); return
    ids = [r["id"] for r in rows]
    cids = [chunk_cid(i) for i in ids]
    dupes = [c for c, n in collections.Counter(cids).items() if n > 1]
    assert not dupes, f"custom_id collisions: {dupes[:3]}"
    bad = [c for c in cids if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", c)]
    assert not bad, f"invalid custom_id charset/length: {bad[:3]}"


# ------------------------------------------------------------------------ qc
def test_qc_strips_and_drops_passage_references():
    assert sanitize_question("What is X according to the passage?") == "What is X?"
    assert sanitize_question("What does the passage say about X?") is None
    assert sanitize_question("How is X calculated?") == "How is X calculated?"


def test_qc_rejects_degenerate_answers():
    ok, _ = acceptable("What is the Arudha Pada?", "the same words " * 12)
    assert not ok, "degenerate repetition was accepted"


def test_no_pair_references_a_passage():
    """The trained set must contain no question that needs unseen context."""
    rows = _load(PAIRS)
    if not rows:
        print("      (skip: no pairs.jsonl)"); return
    rx = re.compile(r"\b(passage|excerpt|text above|provided text)\b", re.I)
    bad = [p["question"] for p in rows if rx.search(p["question"])]
    assert not bad, f"{len(bad)} leaked, e.g. {bad[:2]}"


# --------------------------------------------------------------------- split
def test_split_has_no_book_leakage():
    """THE eval-validity invariant: no source book in both train and test."""
    tr, te = _load(split_path("train")), _load(split_path("test"))
    if not tr or not te:
        print("      (skip: no splits)"); return
    train_books = {r["meta"]["source"] for r in tr}
    test_books = {r["meta"]["source"] for r in te}
    overlap = train_books & test_books
    assert not overlap, f"LEAKAGE: {overlap} appear in both train and test"


def test_split_val_and_test_disjoint_from_train():
    tr, va = _load(split_path("train")), _load(split_path("val"))
    if not tr or not va:
        print("      (skip)"); return
    assert not ({r["meta"]["source"] for r in tr} &
                {r["meta"]["source"] for r in va}), "val leaks into train"


def test_split_val_and_test_use_different_books():
    """The invariant the other two miss: val and test must also be independent
    OF EACH OTHER.

    Both books-in-both-splits and, more subtly, the same source CHUNK feeding a
    question on each side. The first version of 03_split.py passed both leakage
    tests above while val and test shared 232 of 235 chunks, because it split
    each held-out book's rows in half. Anything selected on val was then being
    selected on test, and nothing in the suite objected.
    """
    va, te = _load(split_path("val")), _load(split_path("test"))
    if not va or not te:
        print("      (skip)"); return

    shared_books = ({r["meta"]["source"] for r in va} &
                    {r["meta"]["source"] for r in te})
    assert not shared_books, (
        f"val and test share book(s) {sorted(shared_books)} -- a model selected "
        "on val is then selected on test")

    shared_chunks = ({r["meta"]["chunk_id"] for r in va} &
                     {r["meta"]["chunk_id"] for r in te})
    assert not shared_chunks, (
        f"val and test share {len(shared_chunks)} source chunk(s) -- different "
        "questions, same passages, so the two sets are not independent")


def test_split_preserves_all_pairs():
    pairs = _load(PAIRS)
    parts = [_load(split_path(n)) for n in ("train", "val", "test")]
    if not pairs or any(p is None for p in parts):
        print("      (skip)"); return
    total = sum(len(p) for p in parts)
    assert total == len(pairs), f"{total} split rows vs {len(pairs)} pairs"


def test_chat_format_is_wellformed():
    rows = _load(split_path("train"))
    if not rows:
        print("      (skip)"); return
    for r in rows[:500]:
        m = r["messages"]
        assert [x["role"] for x in m] == ["system", "user", "assistant"], m
        assert all(x["content"].strip() for x in m), "empty message content"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
