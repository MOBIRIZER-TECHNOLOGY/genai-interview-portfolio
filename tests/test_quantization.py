"""
Quantisation tests — codec correctness, and the two measurement traps.

The binary + int8 cascade is the claim project 07 rests on (32x memory reduction
at 0.985 recall). These tests protect both the codec and the *methodology*,
because the methodology is where both real mistakes happened:

- validating on synthetic isotropic vectors, which understates binary recall
  by 3.5x and would have condemned a working technique
- leaving the query inside the corpus, which caps recall@10 at exactly 0.9 in a
  way that looks like a plausible plateau rather than a bug
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "07_rag_at_scale"))

from scale.quantize import (  # noqa: E402
    binary_encode, calibrate_int8, hamming_search, int8_decode, int8_encode,
    memory_report,
)

DIM = 384


@pytest.fixture(scope="module")
def clustered():
    """Anisotropic, clustered unit vectors -- a stand-in for real embeddings.

    Deliberately NOT isotropic noise. See test_synthetic_isotropic_understates_recall.
    """
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(40, DIM))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Noise scale must be well below 1/sqrt(DIM) = 0.051, or the noise norm
    # (scale * sqrt(DIM)) swamps the unit-norm centre and the "clusters" are
    # isotropic in disguise. An earlier version used scale=0.6 -> noise norm
    # 11.8 vs centre 1.0, and the fixture tested nothing it claimed to.
    X = centers[rng.integers(0, 40, 4000)] + rng.normal(scale=0.015, size=(4000, DIM))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


# ------------------------------------------------------------- binary


def test_binary_encode_shape_and_dtype(clustered):
    codes = binary_encode(clustered)
    assert codes.shape == (len(clustered), DIM // 8)
    assert codes.dtype == np.uint8


def test_binary_encode_is_the_sign_bit():
    v = np.array([[1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]], dtype=np.float32)
    # packbits is MSB-first: 1,0,1,0,1,0,1,0 -> 0b10101010
    assert binary_encode(v)[0][0] == 0b10101010


def test_binary_encode_handles_1d_input():
    assert binary_encode(np.ones(DIM, dtype=np.float32)).shape == (1, DIM // 8)


def test_binary_gives_32x_reduction(clustered):
    assert binary_encode(clustered).nbytes * 32 == clustered.nbytes


def test_hamming_finds_itself(clustered):
    codes = binary_encode(clustered)
    dists, idxs = hamming_search(codes[:5], codes, k=1)
    assert list(idxs[:, 0]) == [0, 1, 2, 3, 4]
    assert (dists[:, 0] == 0).all(), "a vector's Hamming distance to itself must be 0"


def test_hamming_distance_is_symmetric_and_bounded(clustered):
    codes = binary_encode(clustered[:50])
    d, _ = hamming_search(codes[:1], codes, k=50)
    assert (d >= 0).all() and (d <= DIM).all()


# --------------------------------------------------------------- int8


def test_int8_roundtrip_error_is_below_one_step(clustered):
    """Mean error must sit near step/2, not near the data's own scale.

    The max error is NOT the metric: the 99.9-percentile calibration
    deliberately clips ~0.2% of values, so max error is large by design. Testing
    max would fail on correct code -- mean is the honest check.
    """
    cal = calibrate_int8(clustered[:2000])
    back = int8_decode(int8_encode(clustered, cal), cal)
    err = np.abs(back - clustered)
    step = cal.np_ranges.mean() / 255
    assert err.mean() < step, f"mean err {err.mean():.6f} exceeds one step {step:.6f}"


def test_int8_codes_span_the_range(clustered):
    cal = calibrate_int8(clustered[:2000])
    codes = int8_encode(clustered, cal)
    assert codes.dtype == np.int8
    assert codes.min() < -100 and codes.max() > 100, "calibration is wasting resolution"


def test_int8_calibration_is_global_not_per_vector(clustered):
    """Per-vector scaling would make two vectors' codes incomparable.

    Encoding a subset must give byte-identical codes to encoding the whole set.
    """
    cal = calibrate_int8(clustered[:2000])
    whole = int8_encode(clustered, cal)
    part = int8_encode(clustered[:100], cal)
    assert np.array_equal(whole[:100], part)


def test_int8_calibration_roundtrips_through_json(tmp_path, clustered):
    cal = calibrate_int8(clustered[:1000])
    p = tmp_path / "cal.json"
    cal.save(p)
    from scale.quantize import Int8Calibration

    loaded = Int8Calibration.load(p)
    assert np.allclose(loaded.np_mins, cal.np_mins)
    assert np.allclose(loaded.np_ranges, cal.np_ranges)


def test_dead_dimension_does_not_divide_by_zero():
    """A constant dimension gives max == min; the guard must keep range at 1.0."""
    X = np.random.default_rng(0).normal(size=(200, DIM)).astype(np.float32)
    X[:, 7] = 0.5                                  # constant column
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cal = calibrate_int8(X)
    assert np.isfinite(int8_encode(X, cal)).all()


# ------------------------------------------------- the two-stage cascade


def _recall(pred, exact, k):
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(pred, exact)]))


def test_rescore_beats_binary_alone(clustered):
    """The whole point of stage 2: it must materially improve on stage 1."""
    k, n_q = 10, 40
    rng = np.random.default_rng(1)
    qi = rng.choice(len(clustered), n_q, replace=False)
    Q = clustered[qi]

    sims = Q @ clustered.T
    for i, q in enumerate(qi):
        sims[i, q] = -1e9                          # exclude self (see below)
    exact = np.argsort(-sims, axis=1)[:, :k]

    cal = calibrate_int8(clustered[:2000])
    B, I8 = binary_encode(clustered), int8_encode(clustered, cal)

    _, bi = hamming_search(binary_encode(Q), B, k + 1)
    binary_only = np.array([[j for j in row if j != q][:k] for row, q in zip(bi, qi)])

    _, C = hamming_search(binary_encode(Q), B, 400)
    rescored = np.empty((n_q, k), np.int64)
    for i in range(n_q):
        c = np.array([j for j in C[i] if j != qi[i]])
        s = int8_decode(I8[c], cal) @ Q[i]
        rescored[i] = c[np.argsort(-s)[:k]]

    r_bin, r_res = _recall(binary_only, exact, k), _recall(rescored, exact, k)
    assert r_res > r_bin, f"rescore ({r_res:.3f}) must beat binary alone ({r_bin:.3f})"

    # Assert on QUALITY, not raw recall.
    #
    # A raw recall floor is the wrong gate on clustered data: tight clusters
    # make the exact top-10 largely arbitrary among near-identical vectors, so
    # recall is depressed by tie-shuffling even when every returned document is
    # equally relevant. Quality ratio -- mean similarity of what we returned
    # over mean similarity of the exact top-k -- is the property that actually
    # matters, and it is what the project's headline claim reports (0.985
    # recall at ratio 1.0000).
    quality = float(np.mean([
        sims[i][rescored[i]].mean() / sims[i][exact[i]].mean() for i in range(n_q)
    ]))
    assert quality > 0.99, f"rescored results are materially worse: ratio {quality:.4f}"


def test_self_in_corpus_caps_recall_at_exactly_0_9(clustered):
    """TRAP 2, pinned as a test.

    If the query is still indexed it is its own nearest neighbour and
    permanently occupies one of k slots -- capping recall@10 at exactly 0.9.
    The number is plausible enough to ship, which is what made it dangerous.
    """
    k, n_q = 10, 30
    rng = np.random.default_rng(2)
    qi = rng.choice(len(clustered), n_q, replace=False)
    Q = clustered[qi]

    sims = Q @ clustered.T
    for i, q in enumerate(qi):
        sims[i, q] = -1e9
    exact = np.argsort(-sims, axis=1)[:, :k]

    # deliberately do NOT exclude self from the results
    B = binary_encode(clustered)
    _, naive = hamming_search(binary_encode(Q), B, k)

    assert _recall(naive, exact, k) <= 0.9 + 1e-9, (
        "leaving the query in the corpus must not exceed 0.9 -- if it does, the "
        "ground truth is also contaminated"
    )


def test_synthetic_isotropic_understates_recall(clustered):
    """TRAP 1, pinned as a test.

    Isotropic Gaussians are the worst case for binary quantisation: independent
    dimensions mean the sign pattern carries little neighbour information.
    Measured 0.18 synthetic vs 0.62 on real embeddings. Validating on
    np.random.normal would have thrown away a working 32x technique.
    """
    k, n_q = 10, 30
    rng = np.random.default_rng(3)
    S = rng.normal(size=(4000, DIM)).astype(np.float32)
    S /= np.linalg.norm(S, axis=1, keepdims=True)

    def binary_only_recall(X):
        qi = rng.choice(len(X), n_q, replace=False)
        Q = X[qi]
        sims = Q @ X.T
        for i, q in enumerate(qi):
            sims[i, q] = -1e9
        exact = np.argsort(-sims, axis=1)[:, :k]
        _, bi = hamming_search(binary_encode(Q), binary_encode(X), k + 1)
        pred = np.array([[j for j in row if j != q][:k] for row, q in zip(bi, qi)])
        return _recall(pred, exact, k)

    assert binary_only_recall(S) < binary_only_recall(clustered), (
        "isotropic vectors must score WORSE than clustered ones; if not, the "
        "clustered fixture has lost its structure and trap 1 is untested"
    )


# ------------------------------------------------------- memory report


def test_memory_report_handles_empty_index():
    """Regression: memory_report(0, dim) divided by zero.

    An empty index is legitimate -- a build that aborts before its first commit
    leaves n_chunks=0 with dim already set.
    """
    r = memory_report(0, DIM)
    assert r["binary_reduction"] == "n/a"
    assert r["binary_gb"] == 0


def test_memory_report_arithmetic():
    r = memory_report(1_000_000, DIM)
    assert r["binary_reduction"] == "32x"
    # memory_report rounds to 2 dp, so compare at that precision
    assert r["float32_gb"] == pytest.approx(1.54, abs=0.01)
    assert r["int8_gb"] == pytest.approx(0.38, abs=0.01)
    assert r["binary_gb"] == pytest.approx(0.048, abs=0.001)
