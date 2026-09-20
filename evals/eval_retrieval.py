#!/usr/bin/env python3
"""
Retrieval evaluation against the gold set for a country.

Metrics per question (top_k=5, same as production):
  - recall@5: whether a gold chunk surfaced in the top-5
  - gate:     whether the top score passed the relevance threshold
  - correct_gate_decision: gated when answerable, not gated when not answerable

Prints a per-category summary and writes evals/data/report_retrieval_<cc>.json.

Usage:
    python3 evals/eval_retrieval.py --country au            # needs gold file
    python3 evals/eval_retrieval.py --country au --no-gold  # coverage stats only
"""

import argparse
import json
from pathlib import Path

from common import (  # noqa: E402
    VectorStore,
    RELEVANCE_THRESHOLD,
    chunk_signature,
    gold_path,
    load_questions,
    read_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval for a country")
    parser.add_argument("--country", default="au")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=RELEVANCE_THRESHOLD)
    parser.add_argument("--no-gold", action="store_true", help="Coverage stats only")
    args = parser.parse_args()

    gold = {r["question"]: r for r in read_jsonl(gold_path(args.country))}
    questions = load_questions()
    if gold:
        questions = [q for q in questions if q["question"] in gold]

    vs = VectorStore(country=args.country)
    categories = {}
    rows = []

    for q in questions:
        results = vs.search(q["question"], top_k=args.top_k)
        sigs = [chunk_signature(r["metadata"]) for r in results]
        top_score = results[0]["score"] if results else 0.0
        gated = top_score >= args.threshold

        g = gold.get(q["question"])
        if g:
            hit = any(s in sigs for s in g["gold_chunks"])
            answerable = g["answerable"]
            correct_gate = (gated and answerable) or (not gated and not answerable)
            recall = bool(hit)
        else:
            hit = recall = answerable = None
            correct_gate = None

        row = {
            "question": q["question"],
            "category": q["category"],
            "top_score": round(top_score, 3),
            "gated": gated,
            "answerable": answerable,
            "gold_hit": hit,
            "correct_gate_decision": correct_gate,
            "n_sources": len(results),
        }
        rows.append(row)
        c = categories.setdefault(
            q["category"], {"n": 0, "gated": 0, "recall_hits": 0, "answerable": 0, "good_gate": 0}
        )
        c["n"] += 1
        c["gated"] += int(gated)
        # recall is only defined for answerable questions; counting a hit on a
        # question the gold set marks unanswerable inflated category recall
        # above 100% (e.g. export_import 5/4 = 125%).
        if answerable:
            c["answerable"] += 1
            if recall:
                c["recall_hits"] += 1
        if correct_gate:
            c["good_gate"] += 1

    total = len(rows)
    answered = sum(1 for r in rows if r["answerable"])
    gate_accuracy = sum(1 for r in rows if r["correct_gate_decision"]) / total if total else 0.0
    recall5 = sum(1 for r in rows if r["gold_hit"]) / answered if answered else 0.0

    print(f"\nRetrieval evaluation — country '{args.country}' "
          f"(threshold={args.threshold}, top_k={args.top_k})")
    print(f"{'category':18} {'n':>3} {'answerable':>10} {'gated':>5} {'recall@5':>9} {'gate acc':>9}")
    for cat, c in sorted(categories.items()):
        rc = c["recall_hits"] / c["answerable"] if c["answerable"] else 0.0
        ga = c["good_gate"] / c["n"] if c["n"] else 0.0
        print(f"{cat:18} {c['n']:>3} {c['answerable']:>10} {c['gated']:>5} "
              f"{rc:>8.1%} {ga:>8.1%}")
    print("-" * 62)
    print(f"{'TOTAL':18} {total:>3} {answered:>10} "
          f"{sum(c['gated'] for c in categories.values()):>5} "
          f"{recall5:>8.1%} {gate_accuracy:>8.1%}")

    misses = [r for r in rows if r["answerable"] and not r["gold_hit"]]
    if misses:
        print(f"\nQuestions where retrieval missed all gold chunks ({len(misses)}):")
        for r in misses:
            print(f"  [{r['category']}] top={r['top_score']} {r['question']}")

    report = {
        "country": args.country,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "n_questions": total,
        "answerable": answered,
        "recall_at_k": round(recall5, 4),
        "gate_accuracy": round(gate_accuracy, 4),
        "categories": {
            cat: {
                "n": c["n"],
                "answerable": c["answerable"],
                "gated": c["gated"],
                "recall_hits": c["recall_hits"],
                "good_gate": c["good_gate"],
            }
            for cat, c in categories.items()
        },
        "rows": rows,
    }
    out = Path(__file__).parent / "data" / f"report_retrieval_{args.country}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
