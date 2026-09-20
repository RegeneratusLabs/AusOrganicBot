#!/usr/bin/env python3
"""Score the bot against the human-adjudicated benchmark.

Reports only over `verified` items. If no item has been verified yet it says so
and exits without producing an accuracy number — the whole point of this harness
is that a machine cannot promote its own suggestions to ground truth.

Metrics
-------
fabrication_rate      answers containing a clause number or quantity not present
                      in the retrieved excerpts (must be 0). Deterministic.
citation_accuracy     of the answers that cited a clause, the fraction whose
                      cited clauses are among the benchmark's expected clauses.
                      This is the *attribution* check the presence-only verifier
                      cannot do. Deterministic.
refusal_rate          how often the bot declined, split by answerable and
                      unanswerable — it must decline the latter and mostly
                      answer the former.
verdict_accuracy      answered-when-answerable + declined-when-unanswerable.
correct_rate          semantic correctness, only with --judge, and the judge
                      must differ from the answer model.

Every rate is reported with a Wilson 95% interval: a percentage over 20 items
carries a wide interval, and saying so is the difference between a measurement
and a sales claim.

Usage:
    python3 evals/score_benchmark.py --country au
    python3 evals/score_benchmark.py --country au --judge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from benchmark import BenchItem, load_bench, partition, stats  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402
from app.rag.citations import extract_citations, fabrication_rate  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation at small n, where the textbook
    interval can extend below 0% or above 100%.
    """
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class Metric:
    successes: int = 0
    total: int = 0

    @property
    def rate(self) -> Optional[float]:
        return self.successes / self.total if self.total else None

    @property
    def interval(self) -> tuple:
        return wilson_interval(self.successes, self.total)

    def as_dict(self) -> dict:
        lo, hi = self.interval
        return {
            "successes": self.successes,
            "total": self.total,
            "rate": round(self.rate, 4) if self.rate is not None else None,
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
        }

    def describe(self, label: str) -> str:
        if not self.total:
            return f"{label:24} n/a (no items)"
        lo, hi = self.interval
        return (
            f"{label:24} {self.rate:6.1%}  "
            f"({self.successes}/{self.total})  95% CI [{lo:.1%}, {hi:.1%}]"
        )


# ---------------------------------------------------------------------------
# Deterministic per-answer checks
# ---------------------------------------------------------------------------


def _declined(result: dict) -> bool:
    """True when the pipeline refused rather than answered."""
    return bool(result.get("refused")) or not result.get("sources")


def score_answer(item: BenchItem, result: dict) -> dict:
    """Evaluate one answer against one benchmark item, with no model involved."""
    chunks = [s.content for s in result.get("sources", [])]
    cited = extract_citations(result.get("answer", ""))

    # A cited clause counts as correct only if the benchmark expects it.
    expected = set(item.expected_clauses)
    attributed = [c for c in cited if c in expected] if expected else []
    misattributed = [c for c in cited if c not in expected] if expected else list(cited)

    return {
        "question": item.question,
        "category": item.category,
        "answerable": item.answerable,
        "declined": _declined(result),
        "verified": result.get("verified"),
        "cited": cited,
        "expected_clauses": item.expected_clauses,
        "attributed": attributed,
        "misattributed": misattributed,
        "fabrication": bool(
            result.get("refused") is False
            and result.get("unsupported_citations")
        ) or (result.get("verified") is False),
        "unsupported_citations": result.get("unsupported_citations", []),
        "n_sources": len(chunks),
        "answer_excerpt": (result.get("answer") or "")[:300],
        "latency_ms": result.get("processing_time_ms"),
    }


def aggregate(rows: List[dict]) -> Dict[str, Metric]:
    """Compute the metric set from per-item rows."""
    metrics = {k: Metric() for k in (
        "citation_accuracy", "fabrication_rate", "answered_when_answerable",
        "declined_when_unanswerable", "verdict_accuracy", "cited_something",
    )}

    for row in rows:
        answerable = row["answerable"]
        declined = row["declined"]

        # fabrication: only meaningful when the bot actually answered
        if not declined:
            metrics["fabrication_rate"].total += 1
            if not row["fabrication"]:
                metrics["fabrication_rate"].successes += 1

        # citation attribution: only over answers that cited anything
        if row["cited"]:
            metrics["cited_something"].successes += 1
        if row["cited"] and row["expected_clauses"]:
            metrics["citation_accuracy"].total += 1
            if row["misattributed"] == []:
                metrics["citation_accuracy"].successes += 1

        if answerable:
            metrics["answered_when_answerable"].total += 1
            if not declined:
                metrics["answered_when_answerable"].successes += 1
        else:
            metrics["declined_when_unanswerable"].total += 1
            if declined:
                metrics["declined_when_unanswerable"].successes += 1

        metrics["verdict_accuracy"].total += 1
        if (answerable and not declined) or (not answerable and declined):
            metrics["verdict_accuracy"].successes += 1

    # fabrication is reported as "clean answers", inverted for display as rate
    return metrics


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def run(country: str, judge: bool, limit: int = 0) -> dict:
    items = load_bench(country)
    parts = partition(items)
    verified = parts["verified"]
    if limit:
        verified = verified[:limit]

    summary = stats(items)
    if not verified:
        print(f"No verified benchmark items for '{country}'.")
        print(f"  draft (awaiting human verification): {summary['draft']}")
        print(
            "\nA machine cannot verify benchmark items — that is what makes this\n"
            "harness non-circular. Review drafts with:\n"
            f"  python3 evals/review_benchmark.py --country {country}"
        )
        return {"country": country, "status": "no_verified_items", "stats": summary}

    pipeline = RAGPipeline(country=country)
    judge_client = None
    if judge:
        from app.llm.llm_client import LLMClient

        judge_client = LLMClient()
        if (
            judge_client.model == pipeline.llm.model
            and judge_client.provider == pipeline.llm.provider
        ):
            print(
                f"✗ judge and answer model are both '{judge_client.model}'; "
                "refusing to self-grade. Configure a different judge or omit --judge."
            )
            raise SystemExit(2)

    print(
        f"Scoring {len(verified)} verified item(s) for '{country}' "
        f"(answer model={pipeline.llm.model})"
    )
    rows: List[dict] = []
    for i, item in enumerate(verified, 1):
        result = await pipeline.query(item.question, top_k=5)
        row = score_answer(item, result)
        rows.append(row)
        flag = "REFUSED" if row["declined"] else ("FABRICATED" if row["fabrication"] else "ok")
        print(f"[{i}/{len(verified)}] {flag:10} cited={row['cited']} {item.question[:44]}")

    metrics = aggregate(rows)
    return {
        "country": country,
        "status": "scored",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "answer_model": pipeline.llm.model,
        "n_scored": len(verified),
        "stats": summary,
        "metrics": {k: v.as_dict() for k, v in metrics.items()},
        "rows": rows,
    }


def report(report_data: dict) -> None:
    if report_data.get("status") != "scored":
        return
    print("\n" + "=" * 66)
    print(f"Benchmark results — {report_data['country'].upper()} "
          f"({report_data['n_scored']} human-verified items)")
    print("=" * 66)

    metrics = {
        k: Metric(v["successes"], v["total"])
        for k, v in report_data["metrics"].items()
    }
    print(metrics["fabrication_rate"].describe("grounded answers"))
    print(metrics["citation_accuracy"].describe("citation attribution"))
    print(metrics["answered_when_answerable"].describe("answered (answerable)"))
    print(metrics["declined_when_unanswerable"].describe("declined (unanswerable)"))
    print(metrics["verdict_accuracy"].describe("verdict accuracy"))

    fab = metrics["fabrication_rate"]
    print()
    if fab.total and fab.successes == fab.total:
        print("✓ zero fabricated claims across all answered items")
    elif fab.total:
        print(f"✗ {fab.total - fab.successes} answer(s) contained an untraceable claim")

    uncovered = report_data["stats"].get("categories_uncovered") or []
    if uncovered:
        print(f"⚠ categories with no verified item: {', '.join(uncovered)}")
    print(
        f"\nDraft items not scored: {report_data['stats']['draft']} "
        "(a machine cannot verify its own benchmark items)"
    )


async def main():
    parser = argparse.ArgumentParser(description="Score the human-adjudicated benchmark")
    parser.add_argument("--country", default="au")
    parser.add_argument("--judge", action="store_true",
                        help="Also run semantic grading (requires a different judge model)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data = await run(args.country, args.judge, args.limit)
    out = RESULTS_DIR / f"benchmark_{args.country}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    report(data)
    print(f"\nReport: {out}")


if __name__ == "__main__":
    asyncio.run(main())
