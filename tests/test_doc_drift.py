"""
Pin the numbers in the READMEs to the numbers the suite actually produces.

## Why this file exists

Every other test here guards *code* drift. Nothing guarded *doc* drift, and it
happened three times: the test count went stale in the root README (105), in
`tests/README.md` (141) and in a commit message (141) while the suite collected
143 -- each figure correct when written, none re-derived afterwards. A portfolio
whose headline numbers disagree with itself is worse than one with no numbers,
because the reader cannot tell which claim was measured.

The failure mode is specific: prose counts have no compiler. So compile them
here. These tests read the claims out of the Markdown and check them against
live collection, against each other, and against the files on disk.

Deliberately NOT checked: the coverage percentage against a live `coverage`
run. That would mean running the whole suite under coverage from inside the
suite -- minutes, and recursive. Cross-document agreement plus the
statements/missed arithmetic catches the drift that actually occurs (one place
updated, the others forgotten) for milliseconds.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
ROOT_README = ROOT / "README.md"
TESTS_README = TESTS / "README.md"

# Markers CI excludes; the documented counts are all for this subset.
DETERMINISTIC = "not llm and not gpu"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# Each claim is anchored to its surrounding words rather than matched loosely,
# because both READMEs also contain historical figures that must NOT be picked
# up: "56% when coverage was first checked", "11 tests fire", "9 tests fire".
COUNT_CLAIMS = [
    (ROOT_README, r"\*\*(\d+) deterministic tests at \d+% coverage\*\*"),
    (ROOT_README, r"(\d+) deterministic tests \(\d+% coverage\)"),
    (TESTS_README, r"#\s*(\d+) tests, ~\d+ s, \d+% coverage"),
    (TESTS_README, r"## Layer 1 — deterministic \((\d+) tests, \d+% coverage\)"),
]

COVERAGE_CLAIMS = [
    (ROOT_README, r"\*\*\d+ deterministic tests at (\d+)% coverage\*\*"),
    (ROOT_README, r"\d+ deterministic tests \((\d+)% coverage\)"),
    (TESTS_README, r"#\s*\d+ tests, ~\d+ s, (\d+)% coverage"),
    (TESTS_README, r"## Layer 1 — deterministic \(\d+ tests, (\d+)% coverage\)"),
]


def _claims(claims: list[tuple[Path, str]]) -> dict[str, int]:
    """Extract each documented figure, keyed by where it lives."""
    out: dict[str, int] = {}
    for path, pattern in claims:
        m = re.search(pattern, _read(path))
        assert m, (
            f"claim not found in {path.name}: /{pattern}/\n"
            "Either the doc was reworded or the claim was deleted -- update this "
            "pattern deliberately, do not delete the check."
        )
        out[f"{path.name}:/{pattern[:34]}.../"] = int(m.group(1))
    return out


@pytest.fixture(scope="module")
def collected_count() -> int:
    """How many tests the deterministic suite really collects, right now."""
    # NOT `-q`: this repo's conftest replaces the quiet reporter with per-file
    # totals and prints no summary line at all, which silently starved the
    # parser below. Verbose collection gives both a summary and node ids.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-m", DETERMINISTIC,
         "--collect-only"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    # normalise Windows separators once, so no pattern below needs a backslash
    # class (exactly the escaping this repo has got wrong before, silently)
    out = (proc.stdout + proc.stderr).replace("\\", "/")

    # Three parsers, most authoritative first. Each fallback exists because a
    # reporting plugin can remove the one above it -- and a count check that
    # silently returns nothing is worse than no check.
    m = re.search(r"(\d+)(?:/\d+)? tests? collected", out)      # this repo's
    if m:
        return int(m.group(1))
    m = re.search(r"collected (\d+) items?", out)               # stock pytest
    if m:
        return int(m.group(1))
    nodes = len(re.findall(r"^tests/\S+\.py::", out, re.M))     # node ids
    if nodes:
        return nodes
    per_file = [int(n) for n in re.findall(r"^tests/\S+\.py: (\d+)$", out, re.M)]
    assert per_file, f"could not determine collected count from:\n{out[-2000:]}"
    return sum(per_file)


@pytest.mark.slow
def test_documented_test_counts_match_the_suite(collected_count):
    """Every documented test count == what pytest collects.

    This is the check that was missing when three documents disagreed.
    """
    for where, claimed in _claims(COUNT_CLAIMS).items():
        assert claimed == collected_count, (
            f"{where} claims {claimed} tests; the suite collects "
            f"{collected_count}. Update the doc (or explain the difference)."
        )


def test_documented_coverage_percentages_agree():
    """All four coverage claims must state the same number.

    Cheap, no subprocess: catches the common case where one doc is updated
    after a coverage run and the other three are not.
    """
    values = _claims(COVERAGE_CLAIMS)
    assert len(set(values.values())) == 1, (
        f"coverage percentage disagrees across documents: {values}"
    )


def test_documented_coverage_arithmetic_is_consistent():
    """`TOTAL n statements m missed p%` must actually satisfy p = (n-m)/n."""
    m = re.search(r"TOTAL\s+(\d+) statements\s+(\d+) missed\s+(\d+)%",
                  _read(TESTS_README))
    assert m, "the TOTAL coverage line is missing from tests/README.md"
    stmts, missed, pct = (int(g) for g in m.groups())
    actual = round(100 * (stmts - missed) / stmts)
    assert actual == pct, (
        f"tests/README.md says {stmts} statements, {missed} missed, {pct}% -- "
        f"but ({stmts}-{missed})/{stmts} rounds to {actual}%. One of the three "
        "numbers was updated without the others."
    )
    # and the headline percentage must be the same one
    headline = set(_claims(COVERAGE_CLAIMS).values())
    assert headline == {pct}, (
        f"the TOTAL line says {pct}% but the headlines say {headline}"
    )


def test_every_test_file_is_documented():
    """No test file may exist without being named in tests/README.md.

    This is the structural half. Project 08 shipped two test files that pinned
    two real bugs -- the `shed mode`/`shed_mode` node split and a null triple
    field -- and neither file appeared in the README's table, so the repo's most
    recent bugs had tests with no documented provenance. A count check would
    never have caught that; this does.
    """
    doc = _read(TESTS_README)
    on_disk = {p.name for p in TESTS.glob("test_*.py")}
    undocumented = sorted(n for n in on_disk if n not in doc)
    assert not undocumented, (
        f"test files not mentioned anywhere in tests/README.md: {undocumented}. "
        "Add a row to the table (or a line to the Layer 2 section) saying what "
        "the file pins."
    )


def test_no_documented_test_file_has_been_deleted():
    """The mirror check: the README must not advertise files that are gone."""
    doc = _read(TESTS_README)
    on_disk = {p.name for p in TESTS.glob("test_*.py")}
    mentioned = set(re.findall(r"\b(test_[a-z0-9_]+\.py)\b", doc))
    missing = sorted(mentioned - on_disk)
    assert not missing, (
        f"tests/README.md documents files that no longer exist: {missing}"
    )
