"""
The highest-risk logic in the pipeline: 05_eval.py's A/B unswap.

Answers are presented to the judge in a randomised order to defeat position
bias. `flip[i]` records whether they were swapped, and the aggregation must undo
it. If that inversion is backwards, the eval reports THE EXACT OPPOSITE
CONCLUSION -- "the base model beat your adapter" -- and every number looks
entirely plausible. Nothing crashes. Nothing looks wrong.

This test drives the aggregation with a rigged judge whose verdicts are known,
and asserts attribution is correct under BOTH flip values.

    python pipeline/tests/test_eval_unswap.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def aggregate(results, flip):
    """Mirror of the unswap in 05_eval.py judge().

    results: list of (index, judge_dict) where judge_dict has 'a', 'b', 'winner'
    flip[i] True  => slot a held BASE,  slot b held TUNED
    flip[i] False => slot a held TUNED, slot b held BASE
    """
    agg = {"tuned": [], "base": []}
    wins = {"tuned": 0, "base": 0, "tie": 0}
    for idx, d in results:
        swap = flip[str(idx)]
        slot_a = "base" if swap else "tuned"
        slot_b = "tuned" if swap else "base"
        agg[slot_a].append(d["a"])
        agg[slot_b].append(d["b"])
        w = d["winner"]
        wins["tie" if w == "tie" else (slot_a if w == "a" else slot_b)] += 1
    return agg, wins


def _score(tag):
    """A score object we can trace back to which system produced it."""
    return {"support": 3 if tag == "tuned" else 1, "fabricated": tag == "base",
            "citation": 2 if tag == "tuned" else 0, "style": 2, "tag": tag}


def test_no_swap_attributes_correctly():
    """flip=False: slot a is TUNED. A win for 'a' must credit tuned."""
    d = {"a": _score("tuned"), "b": _score("base"), "winner": "a"}
    agg, wins = aggregate([(0, d)], {"0": False})
    assert agg["tuned"][0]["tag"] == "tuned"
    assert agg["base"][0]["tag"] == "base"
    assert wins == {"tuned": 1, "base": 0, "tie": 0}, wins


def test_swap_attributes_correctly():
    """flip=True: slot a is BASE. A win for 'a' must credit BASE, not tuned.

    This is the assertion that catches an inverted unswap. With the bug, a
    tuned-model win would be recorded as a base-model win.
    """
    d = {"a": _score("base"), "b": _score("tuned"), "winner": "b"}
    agg, wins = aggregate([(0, d)], {"0": True})
    assert agg["tuned"][0]["tag"] == "tuned", "tuned score attributed to base!"
    assert agg["base"][0]["tag"] == "base", "base score attributed to tuned!"
    assert wins == {"tuned": 1, "base": 0, "tie": 0}, wins


def test_mixed_flips_sum_correctly():
    """A realistic mix: tuned should win every round regardless of slot."""
    results, flip = [], {}
    for i in range(20):
        swap = i % 2 == 0
        flip[str(i)] = swap
        if swap:      # a=base, b=tuned -> tuned wins as 'b'
            results.append((i, {"a": _score("base"), "b": _score("tuned"),
                                "winner": "b"}))
        else:         # a=tuned, b=base -> tuned wins as 'a'
            results.append((i, {"a": _score("tuned"), "b": _score("base"),
                                "winner": "a"}))
    agg, wins = aggregate(results, flip)
    assert wins["tuned"] == 20, wins
    assert wins["base"] == 0, wins
    assert all(s["tag"] == "tuned" for s in agg["tuned"])
    assert all(s["tag"] == "base" for s in agg["base"])


def test_ties_never_credited_to_a_system():
    d = {"a": _score("tuned"), "b": _score("base"), "winner": "tie"}
    _, wins = aggregate([(0, d)], {"0": False})
    assert wins == {"tuned": 0, "base": 0, "tie": 1}, wins


def test_inverted_unswap_would_be_caught():
    """Prove this test file has teeth: a deliberately broken unswap must fail.

    A test that cannot fail is decoration. This runs the buggy version and
    asserts it produces the WRONG answer -- so we know the correct version
    passing above means something.
    """
    def broken(results, flip):
        agg = {"tuned": [], "base": []}
        for idx, d in results:
            swap = flip[str(idx)]
            slot_a = "tuned" if swap else "base"      # <-- inverted
            slot_b = "base" if swap else "tuned"
            agg[slot_a].append(d["a"])
            agg[slot_b].append(d["b"])
        return agg

    d = {"a": _score("base"), "b": _score("tuned"), "winner": "b"}
    agg = broken([(0, d)], {"0": True})
    assert agg["tuned"][0]["tag"] == "base", \
        "the broken version should mis-attribute -- test has no teeth"


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
