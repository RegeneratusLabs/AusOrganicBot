#!/usr/bin/env python3
"""
Threshold sweep for the relevance gate.

For thresholds from --min to --max, evaluates the gate decision against the
gold set: gating when the question is answerable (recall) and not gating when
it isn't (precision of the "answerable" decision). Prints a table and the
threshold with the best F1, which can be set per-country in
data/<cc>/manifest.json ("relevance_threshold").

Usage:
    python3 evals/sweep_threshold.py --country au
"""

import argparse
from pathlib import Path

from common import (  # noqa: E402
    VectorStore,
    chunk_signature,
    gold_path,
    load_questions,
    read_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Sweep relevance threshold for a country")
    parser.add_argument("--country", default="au")
    parser.add_argument("--min", type=float, default=0.40)
    parser.add_argument("--max", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    gold = {r["question"]: r for r in read_jsonl(gold_path(args.country))}
    if not gold:
        raise SystemExit(
            f"No gold file for '{args.country}' — run: python3 evals/build_eval_set.py "
            f"--country {args.country}"
        )
    questions = [q for q in load_questions() if q["question"] in gold]

    vs = VectorStore(country=args.country)
    top_scores = {}
    for q in questions:
        results = vs.search(q["question"], top_k=args.top_k)
        top_scores[q["question"]] = results[0]["score"] if results else 0.0

    print(f"\nThreshold sweep — country '{args.country}' ({len(questions)} questions)\n")
    print(f"{'threshold':>10} {'gate_recall':>12} {'gate_precision':>15} {'f1':>7}")
    best = None
    t = args.min
    while t <= args.max + 1e-9:
        gated_answers = sum(
            1 for q in questions
            if top_scores[q["question"]] >= t and gold[q["question"]]["answerable"]
        )
        total_answerable = sum(1 for q in questions if gold[q["question"]]["answerable"])
        total_gated = sum(1 for q in questions if top_scores[q["question"]] >= t)

        recall = gated_answers / total_answerable if total_answerable else 0.0
        precision = gated_answers / total_gated if total_gated else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{t:>10.2f} {recall:>11.1%} {precision:>14.1%} {f1:>7.3f}")
        if best is None or f1 > best[1]:
            best = (t, f1)
        t = round(t + args.step, 4)

    print(f"\nBest threshold: {best[0]:.2f} (F1={best[1]:.3f})")
    print(f"Set it per-country in data/{args.country}/manifest.json: "
          f'"relevance_threshold": {best[0]:.2f}')


if __name__ == "__main__":
    main()
