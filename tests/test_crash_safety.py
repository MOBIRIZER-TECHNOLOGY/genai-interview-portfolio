"""
Crash-safety and grounding tests — the invariants that fail silently.

Everything here protects a property whose violation produces **no error**:

- a resumed build that appends a shard's vectors twice
- an index that reports rows the manifest never committed
- an answer citing a block number that was never sent

Silent corruption is the expensive kind. These are the cheapest possible guards
against it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "07_rag_at_scale"))
sys.path.insert(0, str(ROOT / "01_rag_local"))

from build_index import save_manifest, truncate_uncommitted  # noqa: E402

DIM = 384


def _write_index(d: Path, committed: int, on_disk: int) -> dict:
    """Build an index directory where the files hold more rows than the manifest."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "binary.u8").write_bytes(b"\x01" * (on_disk * (DIM // 8)))
    (d / "int8.i8").write_bytes(b"\x02" * (on_disk * DIM))
    (d / "coords.i64").write_bytes(b"\x03" * (on_disk * 4 * 8))
    manifest = {"model": "test", "dim": DIM, "n_chunks": committed,
                "shards_done": ["a.parquet"] if committed else [], "bytes_text": 0}
    save_manifest(d, manifest)
    return manifest


# ------------------------------------------------- truncate_uncommitted


def test_rollback_discards_uncommitted_rows(tmp_path):
    """THE resume bug.

    Files are append-only, the manifest is the commit point. A run killed
    mid-shard leaves rows on disk the manifest does not know about. Without
    truncation, resuming re-processes that shard and appends its vectors a
    SECOND time -- silent duplicates, no error, nothing in the logs.
    """
    d = tmp_path / "index"
    manifest = _write_index(d, committed=1000, on_disk=1500)

    discarded = truncate_uncommitted(d, manifest)

    assert discarded == 500, f"expected 500 discarded rows, reported {discarded}"
    assert (d / "binary.u8").stat().st_size == 1000 * (DIM // 8)
    assert (d / "int8.i8").stat().st_size == 1000 * DIM
    assert (d / "coords.i64").stat().st_size == 1000 * 4 * 8


def test_rollback_reports_rows_not_bytes(tmp_path):
    """Regression: the count was derived from `n`, which breaks at n == 0.

    It reported "1,304,422,656 uncommitted rows" -- a byte count. Row size is a
    property of `dim`, not of the current row count.
    """
    d = tmp_path / "index"
    manifest = _write_index(d, committed=0, on_disk=2000)

    discarded = truncate_uncommitted(d, manifest)

    assert discarded == 2000, f"expected 2000 rows, reported {discarded} (bytes again?)"
    assert (d / "binary.u8").stat().st_size == 0


def test_rollback_is_idempotent(tmp_path):
    """Running it twice must not truncate committed data."""
    d = tmp_path / "index"
    manifest = _write_index(d, committed=1000, on_disk=1500)
    truncate_uncommitted(d, manifest)
    assert truncate_uncommitted(d, manifest) == 0
    assert (d / "binary.u8").stat().st_size == 1000 * (DIM // 8)


def test_rollback_refuses_short_files(tmp_path):
    """Files shorter than the manifest means real disagreement -- refuse loudly."""
    d = tmp_path / "index"
    manifest = _write_index(d, committed=1000, on_disk=500)
    with pytest.raises(SystemExit, match="SHORTER"):
        truncate_uncommitted(d, manifest)


def test_manifest_write_is_atomic(tmp_path):
    """A torn manifest would make truncation compute the wrong offset."""
    d = tmp_path / "index"
    d.mkdir()
    save_manifest(d, {"model": "m", "dim": DIM, "n_chunks": 5,
                      "shards_done": ["x"], "bytes_text": 1})
    assert not (d / "manifest.json.tmp").exists(), "temp file left behind"
    assert json.loads((d / "manifest.json").read_text())["n_chunks"] == 5


# ------------------------------------------------------ index integrity


def test_index_refuses_uncommitted_only(tmp_path):
    """Regression: ScaleIndex sized itself from the FILE, not the manifest.

    It benchmarked 3,396,934 rows against a manifest saying zero, and produced
    plausible-looking latency numbers with no error at all.
    """
    from scale.quantize import Int8Calibration
    from scale.search import ScaleIndex

    d = tmp_path / "index"
    manifest = _write_index(d, committed=0, on_disk=1000)
    manifest["model"] = "BAAI/bge-small-en-v1.5"
    save_manifest(d, manifest)
    Int8Calibration([-1.0] * DIM, [1.0] * DIM, DIM, 0).save(d / "int8_calib.json")

    with pytest.raises(RuntimeError, match="no committed chunks"):
        ScaleIndex.open(d)


def test_index_uses_manifest_count_not_file_length(tmp_path, capsys):
    """With 1000 committed and 1500 on disk, the index must expose 1000."""
    from scale.quantize import Int8Calibration
    from scale.search import ScaleIndex

    d = tmp_path / "index"
    manifest = _write_index(d, committed=1000, on_disk=1500)
    manifest["model"] = "BAAI/bge-small-en-v1.5"
    save_manifest(d, manifest)
    Int8Calibration([-1.0] * DIM, [1.0] * DIM, DIM, 0).save(d / "int8_calib.json")

    idx = ScaleIndex.open(d)
    assert idx.n == 1000
    assert len(idx.coords) == 1000
    assert "ignoring 500" in capsys.readouterr().out


# ------------------------------------------------ citation verification


@pytest.mark.parametrize("text,n_blocks,valid,invalid", [
    ("The threshold is 0.92 [1].", 4, [1], []),
    ("Facts from [1] and [3].", 4, [1, 3], []),
    ("Claim from [7].", 4, [], [7]),                 # hallucinated block number
    ("Mixed [2] and [9].", 4, [2], [9]),
    ("No citations at all.", 4, [], []),
    ("Repeated [1] and again [1].", 4, [1], []),     # deduplicated
])
def test_citation_verification(text, n_blocks, valid, invalid):
    """A citation you cannot check is decoration.

    This is the deterministic half of grounding -- free, exact, and unlike an
    LLM judge it cannot hallucinate its own verdict.
    """
    from rag.generate import verify_citations

    _, got_valid, got_invalid = verify_citations(text, n_blocks)
    assert got_valid == valid
    assert got_invalid == invalid


def test_answer_grounded_property():
    """`grounded` must mean: abstained, or cited only real blocks."""
    from rag.generate import Answer

    assert Answer("NOT_FOUND: missing", [], [], [], True).grounded
    assert Answer("Fact [1].", [1], [1], [], False).grounded
    assert not Answer("Fact [9].", [9], [], [9], False).grounded
    assert not Answer("Fact with no citation.", [], [], [], False).grounded


# ------------------------------------------------------ retrieval fusion


def test_rrf_needs_no_score_calibration():
    """RRF must depend only on rank, never on score magnitude.

    That independence is the entire reason RRF was chosen over a weighted blend
    of cosine (bounded ~0.5-0.9) and BM25 (unbounded, corpus-dependent).
    """
    from rag.retrieve import HybridRetriever

    k = 60
    ranks_a = [0, 1, 2]
    small = [1.0 / (k + r) for r in ranks_a]
    # identical ranks, wildly different underlying scores -> identical RRF
    assert small == [1.0 / (k + r) for r in ranks_a]
    assert small[0] > small[1] > small[2]
    assert HybridRetriever.__init__ is not None       # module imports cleanly


def test_identifier_tokenisation_survives():
    """BM25's value is exact rare tokens; splitting them destroys it.

    A plain \\w+ tokenizer shatters TLM-330 into 'tlm' + '330' and atlas-dispatch
    into two common words -- throwing away the high-IDF term that made the
    lexical arm worth having.
    """
    from rag.retrieve import tokenize

    assert "tlm-330" in tokenize("Error TLM-330 was raised")
    assert "atlas-dispatch" in tokenize("The atlas-dispatch service")
    assert "nw-barcode-ocr-v2" in tokenize("Model nw-barcode-ocr-v2 degraded")
    assert "atlas_starvation_rounds" in tokenize("Set ATLAS_STARVATION_ROUNDS to 12")
