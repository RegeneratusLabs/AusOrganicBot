"""Tests for the human-adjudicated benchmark.

The most important property here is non-circularity: a draft item must never
contribute to a reported accuracy number, and verification must require a named
human. Everything else (Wilson intervals, metric aggregation) supports that.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

# Loaded by explicit file path: the evals directory is not a package and the
# name `benchmark` collides with a pytest/plugin module.
EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "evals"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EVALS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# `score_benchmark` imports `benchmark` and `app.*` by name, so evals must be on
# the path and the backend must be importable before loading it.
sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

benchmark = _load("benchmark", "benchmark.py")

from benchmark import (  # noqa: E402
    BenchItem,
    partition,
    stats,
    verify_item,
)
from score_benchmark import Metric, aggregate, score_answer, wilson_interval  # noqa: E402


def make_item(**kwargs) -> BenchItem:
    base = dict(
        question="What are the buffer zone requirements?",
        category="buffer_zone",
        answerable=True,
        expected_clauses=["3.3.2"],
    )
    base.update(kwargs)
    return BenchItem(**base)


class TestVerification:
    def test_draft_is_not_ground_truth(self):
        assert make_item().is_verified is False

    def test_verify_requires_a_named_human(self):
        with pytest.raises(ValueError, match="reviewer name"):
            verify_item(make_item(), verified_by="   ")

    def test_verify_stamps_reviewer_and_date(self):
        item = verify_item(
            make_item(), verified_by="Jane Smith", reference_answer="At least 5 metres (3.3.2)."
        )
        assert item.is_verified is True
        assert item.verified_by == "Jane Smith"
        assert item.verified_at  # ISO date stamped

    def test_verified_item_requires_reference_answer(self):
        with pytest.raises(ValueError, match="reference_answer"):
            verify_item(make_item(), verified_by="Jane Smith")

    def test_answerable_item_requires_expected_clauses(self):
        problems = make_item(expected_clauses=[]).validate()
        assert any("expected_clauses" in p for p in problems)

    def test_unanswerable_item_must_not_declare_clauses(self):
        problems = make_item(answerable=False, expected_clauses=["1.1"]).validate()
        assert any("must not declare" in p for p in problems)

    def test_unknown_category_is_rejected(self):
        assert any("category" in p for p in make_item(category="nonsense").validate())


class TestPartitionNonCircularity:
    def test_drafts_are_excluded_from_the_verified_set(self):
        verified = verify_item(
            make_item(), verified_by="Jane", reference_answer="5 metres"
        )
        draft = make_item(question="Unverified question?")
        parts = partition([verified, draft])
        assert parts["verified"] == [verified]
        assert parts["draft"] == [draft]

    def test_stats_report_zero_verified_for_drafts_only(self):
        summary = stats([make_item(), make_item(question="Another?")])
        assert summary["verified"] == 0
        assert summary["draft"] == 2

    def test_stats_count_verified_by_category(self):
        items = [
            verify_item(make_item(), verified_by="J", reference_answer="x"),
            verify_item(
                make_item(question="Inputs?", category="input_approval", expected_clauses=["5.1"]),
                verified_by="J",
                reference_answer="y",
            ),
        ]
        summary = stats(items)
        assert summary["verified"] == 2
        assert summary["by_category"] == {"buffer_zone": 1, "input_approval": 1}


class TestWilsonInterval:
    def test_zero_total_is_safe(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_interval_contains_the_point_estimate(self):
        lo, hi = wilson_interval(8, 10)
        assert lo < 0.8 < hi

    def test_small_sample_gives_a_wide_interval(self):
        lo, hi = wilson_interval(2, 3)
        assert hi - lo > 0.5, "3 items cannot support a precise claim"

    def test_bounds_stay_within_zero_and_one(self):
        lo, hi = wilson_interval(0, 5)
        assert lo >= 0.0 and hi <= 1.0
        lo, hi = wilson_interval(5, 5)
        assert lo >= 0.0 and hi <= 1.0


def result(answer: str, sources=None, **kwargs) -> dict:
    sources = [] if sources is None else sources
    payload = {
        "answer": answer,
        "sources": [type("S", (), {"content": c})() for c in sources],
        "verified": True,
        "unsupported_citations": [],
    }
    payload.update(kwargs)
    return payload


class TestScoreAnswer:
    def test_correct_citation_is_attributed(self):
        item = make_item(expected_clauses=["3.3.2"])
        row = score_answer(item, result("Buffer zones are 5 metres (clause 3.3.2).", ["3.3.2 rule text"]))
        assert row["cited"] == ["3.3.2"]
        assert row["misattributed"] == []
        assert row["declined"] is False

    def test_wrong_but_expected_absent_clause_is_misattributed(self):
        """The attribution check the presence-only verifier cannot do."""
        item = make_item(expected_clauses=["3.3.2"])
        row = score_answer(item, result("See clause 3.2.9.", ["3.2.9 is the GMO clause"]))
        assert row["cited"] == ["3.2.9"]
        assert row["misattributed"] == ["3.2.9"]

    def test_fabrication_flag_when_answer_was_refused_for_verification(self):
        item = make_item()
        row = score_answer(
            item,
            result("Section 9.9.9 requires 4 hectares.", ["3.3.2 rule text"],
                   verified=False, refused=True, unsupported_citations=["9.9.9"]),
        )
        assert row["fabrication"] is True

    def test_no_sources_counts_as_declined(self):
        row = score_answer(make_item(), result("hello", []))
        assert row["declined"] is True


class TestAggregate:
    def _rows(self, specs):
        rows = []
        for answerable, declined, fabrication, cited, misattributed in specs:
            rows.append({
                "answerable": answerable,
                "declined": declined,
                "fabrication": fabrication,
                "cited": cited,
                "expected_clauses": ["3.3.2"] if answerable else [],
                "misattributed": misattributed,
            })
        return rows

    def test_perfect_run(self):
        metrics = aggregate(self._rows([
            (True, False, False, ["3.3.2"], []),
            (False, True, False, [], []),
        ]))
        assert metrics["verdict_accuracy"].rate == 1.0
        assert metrics["fabrication_rate"].rate == 1.0
        assert metrics["citation_accuracy"].rate == 1.0
        assert metrics["answered_when_answerable"].rate == 1.0
        assert metrics["declined_when_unanswerable"].rate == 1.0

    def test_fabrication_and_wrong_verdicts_are_caught(self):
        metrics = aggregate(self._rows([
            (True, False, True, ["9.9.9"], ["9.9.9"]),   # fabricated + misattributed
            (False, False, False, ["7.1"], ["7.1"]),     # answered an unanswerable question
        ]))
        assert metrics["fabrication_rate"].rate == 0.5
        assert metrics["citation_accuracy"].rate == 0.0
        assert metrics["declined_when_unanswerable"].rate == 0.0
        # Row 1 answered an answerable question, which is the right verdict even
        # though the content was fabricated; row 2 answered an unanswerable one.
        assert metrics["verdict_accuracy"].rate == 0.5

    def test_rates_are_none_when_denominator_is_zero(self):
        metrics = aggregate(self._rows([(False, True, False, [], [])]))
        # nothing was answered, so fabrication has no denominator
        assert metrics["fabrication_rate"].rate is None
        assert metrics["answered_when_answerable"].rate is None

    def test_metric_dict_is_json_safe(self):
        metrics = aggregate(self._rows([(True, False, False, ["3.3.2"], [])]))
        payload = metrics["verdict_accuracy"].as_dict()
        assert set(payload) == {"successes", "total", "rate", "ci95_low", "ci95_high"}
        assert 0.0 <= payload["ci95_low"] <= payload["ci95_high"] <= 1.0
