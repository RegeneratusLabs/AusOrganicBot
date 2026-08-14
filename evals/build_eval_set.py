#!/usr/bin/env python3
"""
Build the gold retrieval eval set for a country (LLM-annotated, silver labels).

For each question from the stress bank:
  1. retrieve the top-10 candidate chunks from the country's vector store
  2. ask the LLM which candidates actually answer the question (and whether
     the question is answerable at all from this country's standards)
  3. append the result to evals/data/gold_<cc>.jsonl

The output is the retrieval benchmark used by eval_retrieval.py and
sweep_threshold.py, and is structured for publishing as a Hugging Face
dataset (CC-BY-4.0 — see dataset/README.md).

Resumable: questions already present in the gold file are skipped.

Usage:
    python3 evals/build_eval_set.py --country au
    python3 evals/build_eval_set.py --country au --top-k 10 --limit 20
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    EVALS_DATA_DIR,
    VectorStore,
    append_jsonl,
    chunk_signature,
    existing_gold_questions,
    gold_path,
    load_questions,
)
from app.llm.llm_client import LLMClient  # noqa: E402

ANNOTATE_PROMPT = """You are labeling a retrieval benchmark for organic certification standards.

A user asked:
"{question}"

Below are candidate text chunks retrieved from the standards documents, numbered. For EACH candidate, decide whether it contains information that would help answer the question (relevant), or not (irrelevant). Also decide whether the question is answerable at all from these standards documents as a whole (answerable=true) or is genuinely not covered (answerable=false).

Candidates:
{candidates}

Respond as JSON only, in this exact shape:
{{"answerable": true, "relevant": [2, 5]}}

The "relevant" list must contain only the candidate numbers that are relevant to answering the question."""


def format_candidates(results):
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] ({chunk_signature(r['metadata'])}) score={r['score']:.3f}\n    {r['content'][:400]}")
    return "\n\n".join(lines)


async def annotate(llm: LLMClient, question: str, results) -> dict:
    prompt = ANNOTATE_PROMPT.format(
        question=question, candidates=format_candidates(results)
    )
    out = await llm.generate(prompt=prompt, temperature=0.0, max_tokens=300)
    content = out["content"]
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(match.group(0)) if match else json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  ! could not parse annotator JSON ({e}); treating as no gold: {content[:120]!r}")
        return None
    return data


async def main():
    parser = argparse.ArgumentParser(description="Build gold retrieval eval set")
    parser.add_argument("--country", default="au")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Max questions to annotate (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", default=True, help=argparse.SUPPRESS)
    args = parser.parse_args()

    vs = VectorStore(country=args.country)
    llm = LLMClient()
    questions = load_questions()
    done = existing_gold_questions(args.country)
    todo = [q for q in questions if q["question"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Annotating {len(todo)} questions for country '{args.country}' "
          f"(model: {llm.model}, {len(done)} already done)")
    for i, q in enumerate(todo, 1):
        results = vs.search(q["question"], top_k=args.top_k)
        if not results:
            print(f"[{i}/{len(todo)}] no retrieval results — skipping: {q['question']}")
            continue
        annotation = await annotate(llm, q["question"], results)
        if annotation is None:
            continue
        record = {
            "question": q["question"],
            "category": q["category"],
            "answerable": annotation.get("answerable", True),
            "gold_chunks": [
                chunk_signature(results[idx - 1]["metadata"])
                for idx in annotation.get("relevant", [])
                if 1 <= idx <= len(results)
            ],
        }
        append_jsonl(gold_path(args.country), record)
        print(f"[{i}/{len(todo)}] answerable={record['answerable']} "
              f"gold={len(record['gold_chunks'])} {q['question'][:50]}")
    print(f"Done. Gold file: {gold_path(args.country)}")


if __name__ == "__main__":
    asyncio.run(main())
