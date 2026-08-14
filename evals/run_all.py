#!/usr/bin/env python3
"""
One-shot eval pipeline for a country:

  1. build_eval_set  — LLM-annotate gold labels (silver)
  2. sweep_threshold — pick the relevance threshold that maximizes gate F1
  3. eval_retrieval  — recall@5 + gate accuracy report
  4. llm_judge       — answer-quality grades + HTML report

Usage:
    python3 evals/run_all.py --country au
    python3 evals/run_all.py --country au --limit 30   # judge subset only
"""

import argparse
import subprocess
import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent


def run(args, step):
    print(f"\n{'=' * 64}\nStep {step}: {' '.join(args)}\n{'=' * 64}")
    subprocess.run([sys.executable] + args, cwd=EVALS, check=False)


def main():
    parser = argparse.ArgumentParser(description="Run the full eval pipeline for a country")
    parser.add_argument("--country", default="au")
    parser.add_argument("--limit", type=int, default=0, help="Judge subset (0 = all)")
    args = parser.parse_args()

    run([str(EVALS / "build_eval_set.py"), "--country", args.country], 1)
    run([str(EVALS / "sweep_threshold.py"), "--country", args.country], 2)
    run([str(EVALS / "eval_retrieval.py"), "--country", args.country], 3)
    judge = [str(EVALS / "llm_judge.py"), "--country", args.country]
    if args.limit:
        judge += ["--limit", str(args.limit)]
    run(judge, 4)
    print("\nAll eval steps complete. Human spot-check the flagged rows in "
          f"evals/data/report_judge_{args.country}.html")


if __name__ == "__main__":
    main()
