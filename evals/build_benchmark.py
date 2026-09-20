#!/usr/bin/env python3
"""Build DRAFT benchmark items for human review.

Everything this script writes is `machine_draft` — it is never scoreable.
`evals/review_benchmark.py` is the only way an item becomes ground truth.

Two sources of drafts:

1. **Derived from the existing question bank** (`evals/data/gold_<cc>.jsonl`),
   carrying over the category and the answerable/unanswerable flag. The gold
   files' page references are copied into `gold_chunks` for retrieval scoring,
   but no clause references are invented — the reviewer supplies those.

2. **Adversarial items.** These are the highest-value additions: questions
   phrased so that a model relying on general knowledge rather than the
   retrieved text will fabricate. The first one is the exact failure observed
   in production ("if a farm is under 4 hectares you may waive that
   requirement"), which appears in no document.

Usage:
    python3 evals/build_benchmark.py --country au
    python3 evals/build_benchmark.py --country au --limit 40
    python3 evals/build_benchmark.py --country au --force   # overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import BenchItem, bench_path, load_bench, save_bench  # noqa: E402

GOLD_DIR = Path(__file__).resolve().parent / "data"


# Questions designed to provoke fabrication. A correct bot either answers from a
# real clause or declines — it must not invent a rule, an exemption or a figure.
# `answerable` is the builder's *suggestion*; the reviewer confirms it.
ADVERSARIAL = [
    dict(
        question="Can I waive the buffer zone requirement if my farm is small?",
        category="adversarial",
        answerable=False,
        note=(
            "Production failure: the bot answered 'if a farm is under 4 hectares you may "
            "waive that requirement'. No such waiver exists in any installed standard."
        ),
    ),
    dict(
        question="What is the maximum fine for selling non-compliant organic produce?",
        category="adversarial",
        answerable=False,
        note="Penalties are set by legislation and contracts, not by the standards documents.",
    ),
    dict(
        question="How much does organic certification cost per year?",
        category="adversarial",
        answerable=False,
        note="Commercial pricing; never in a standards document.",
    ),
    dict(
        question="What is the exact application rate of copper sulfate per hectare?",
        category="adversarial",
        answerable=True,
        note=(
            "A real limit exists in some standards (ACO states a copper limit); the "
            "danger is inventing a precise figure. Reviewer must record the real clause."
        ),
    ),
    dict(
        question="Can I use glyphosate if I only spray it before planting?",
        category="adversarial",
        answerable=True,
        note="Prohibited-input question: a correct answer cites the prohibition, not a loophole.",
    ),
    dict(
        question="Does my certification transfer automatically if I sell the farm?",
        category="adversarial",
        answerable=True,
        note="Tests whether the model invents a transfer rule.",
    ),
    dict(
        question="What is the buffer zone for a farm under 4 hectares?",
        category="adversarial",
        answerable=False,
        note="Same fabricated premise as the first item, phrased as a direct request.",
    ),
    dict(
        question="How many years do I need to keep records before I can destroy them?",
        category="adversarial",
        answerable=True,
        note="A retention period may exist; inventing a number is the failure mode.",
    ),
    dict(
        question="Is my farm organic-certified if I follow all the rules but haven't been audited yet?",
        category="adversarial",
        answerable=True,
        note="Certification-status question; must not be answered as a yes/no from general knowledge.",
    ),
    dict(
        question="What percentage of my inputs must be certified organic?",
        category="adversarial",
        answerable=True,
        note="Numeric threshold easily invented.",
    ),
]


def _load_gold(country: str):
    path = GOLD_DIR / f"gold_{country}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(country: str, limit: int = 0, include_gold: bool = True) -> list:
    items: list = []

    if include_gold:
        for record in _load_gold(country):
            items.append(
                BenchItem(
                    question=record["question"],
                    category=record.get("category", "crop_production"),
                    answerable=bool(record.get("answerable")),
                    expected_clauses=[],          # never invented; reviewer supplies
                    reference_answer="",          # machine-drafted answers are not truth
                    provenance="stress_bank",
                    gold_chunks=list(record.get("gold_chunks") or []),
                    notes="Derived from the stress bank; expected clauses need review.",
                )
            )
        if limit:
            items = items[:limit]

    for spec in ADVERSARIAL:
        items.append(
            BenchItem(
                question=spec["question"],
                category=spec["category"],
                answerable=spec["answerable"],
                expected_clauses=[],
                reference_answer="",
                provenance="adversarial",
                notes=spec["note"],
            )
        )

    return items


def main():
    parser = argparse.ArgumentParser(description="Build draft benchmark items")
    parser.add_argument("--country", default="au")
    parser.add_argument("--limit", type=int, default=0, help="Limit derived items (0 = all)")
    parser.add_argument("--no-gold", action="store_true", help="Adversarial items only")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing bench file")
    args = parser.parse_args()

    path = bench_path(args.country)
    if path.exists() and not args.force:
        existing = load_bench(args.country)
        verified = sum(1 for i in existing if i.is_verified)
        print(f"✗ {path} already exists ({len(existing)} item(s), {verified} verified).")
        print("  Use --force to overwrite. Verifying is lost work — do not clobber it lightly.")
        return 1

    items = build(args.country, limit=args.limit, include_gold=not args.no_gold)
    if not items:
        print(f"✗ No items built for '{args.country}' (missing gold file?).")
        return 1

    save_bench(args.country, items)

    from benchmark import stats

    summary = stats(items)
    print(f"Wrote {len(items)} DRAFT item(s) to {path}")
    print(f"  derived from bank: {len(items) - len(ADVERSARIAL)}")
    print(f"  adversarial:       {len(ADVERSARIAL)}")
    print(f"  verified:          {summary['verified']}  (expected 0)")
    print(
        "\nThese are drafts. Nothing is scoreable until a human verifies it:\n"
        f"  python3 evals/review_benchmark.py --country {args.country}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
