#!/usr/bin/env python3
"""Interactively promote draft benchmark items to human-verified ground truth.

This is the only path by which an item becomes scoreable. It is deliberately a
human-in-the-loop CLI: an LLM suggesting both the question and the answer would
recreate exactly the circularity this harness exists to remove.

For each draft item the reviewer sees the question, the machine-suggested clause
references and a prompt for the reference answer, then chooses:

    y  verify as-is (uses the suggested clauses/answer if given)
    e  edit the expected clauses and reference answer, then verify
    s  skip (leave as draft for now)
    d  delete the item
    q  save and quit

Usage:
    python3 evals/review_benchmark.py --country au
    python3 evals/review_benchmark.py --country au --reviewer "Jane Smith"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import BenchItem, load_bench, save_bench, verify_item  # noqa: E402

BANNER = """\
==============================================================
 Benchmark review — {country}
==============================================================
An item becomes ground truth only when you verify it.
Your name is stamped on every item you verify.

  y = verify as shown      e = edit then verify
  s = skip                 d = delete          q = save & quit
"""


def _prompt_lines(prompt: str) -> list:
    print(prompt + " (blank line to finish):")
    lines = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line.rstrip())
    return lines


def main():
    parser = argparse.ArgumentParser(description="Review draft benchmark items")
    parser.add_argument("--country", default="au")
    parser.add_argument("--reviewer", default="", help="Name to stamp on verified items")
    args = parser.parse_args()

    items = load_bench(args.country)
    if not items:
        print(f"No benchmark items for '{args.country}'. Build drafts first:")
        print(f"  python3 evals/build_benchmark.py --country {args.country}")
        return 1

    drafts = [i for i in items if not i.is_verified]
    if not drafts:
        print(f"All {len(items)} item(s) for '{args.country}' are already verified.")
        return 0

    reviewer = args.reviewer.strip()
    if not reviewer:
        try:
            reviewer = input("Your name (stamped on verified items): ").strip()
        except EOFError:
            reviewer = ""
    if not reviewer:
        print("✗ A reviewer name is required — verification without attribution is not proof.")
        return 1

    print(BANNER.format(country=args.country.upper()))
    print(f"{len(drafts)} draft item(s) to review as {reviewer!r}.\n")

    remaining: list = [i for i in items if i.is_verified]
    verified_count = deleted_count = 0

    try:
        for index, item in enumerate(drafts, 1):
            print("-" * 62)
            print(f"[{index}/{len(drafts)}]  {item.category}  (provenance: {item.provenance})")
            print(f"Q: {item.question}")
            answerable = "ANSWERABLE" if item.answerable else "NOT answerable (should refuse)"
            print(f"   {answerable}")
            if item.expected_clauses:
                print(f"   suggested clauses: {', '.join(item.expected_clauses)}")
            if item.gold_chunks:
                print(f"   suggested pages:   {', '.join(item.gold_chunks[:3])}")
            if item.reference_answer:
                print(f"   suggested answer:  {item.reference_answer[:160]}")
            print()

            choice = input("y / e / s / d / q > ").strip().lower() or "s"

            if choice == "q":
                remaining.extend(drafts[index - 1:])
                break
            if choice == "s":
                remaining.append(item)
                continue
            if choice == "d":
                deleted_count += 1
                continue

            clauses = list(item.expected_clauses)
            reference = item.reference_answer

            if choice == "e":
                raw = input(
                    f"  clauses (comma-separated, blank keeps {clauses}): "
                ).strip()
                if raw:
                    clauses = [c.strip() for c in raw.split(",") if c.strip()]
                answerable = input("  answerable? [Y/n]: ").strip().lower() != "n"
                item.answerable = answerable
                if not answerable:
                    clauses = []
                print("  Reference answer (what a correct answer must contain):")
                lines = _prompt_lines("  reference_answer")
                if lines:
                    reference = " ".join(lines)
            elif not reference:
                print("  A verified item needs a reference answer.")
                print("  Reference answer (what a correct answer must contain):")
                lines = _prompt_lines("  reference_answer")
                reference = " ".join(lines)

            if item.answerable and not clauses:
                print("  ✗ An answerable item needs at least one expected clause. Skipping.\n")
                remaining.append(item)
                continue
            if not reference:
                print("  ✗ Reference answer required. Skipping.\n")
                remaining.append(item)
                continue

            try:
                verify_item(
                    item,
                    verified_by=reviewer,
                    reference_answer=reference,
                    expected_clauses=clauses,
                )
            except ValueError as exc:
                print(f"  ✗ {exc}. Skipping.\n")
                remaining.append(item)
                continue

            verified_count += 1
            remaining.append(item)
            print("  ✓ verified\n")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — saving what has been reviewed.")
        done = {id(i) for i in remaining}
        remaining.extend(i for i in drafts if id(i) not in done)

    path = save_bench(args.country, remaining)
    print(f"\nVerified this session: {verified_count}   deleted: {deleted_count}")
    print(f"Saved {len(remaining)} item(s) to {path}")

    from benchmark import stats

    print(f"\nBenchmark coverage: {stats(remaining)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
