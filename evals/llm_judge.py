#!/usr/bin/env python3
"""
End-to-end answer quality evaluation with an LLM-as-judge rubric.

For each question (limit N): runs the production pipeline for a country, then
has the judge grade the answer on grounding, citation, correctness, and
refusal behavior. Results are written to:
  - evals/data/judge_<cc>.jsonl     (raw grades)
  - evals/data/report_judge_<cc>.html (human-reviewable report)

Human spot-checking a sample of the lowest-scoring rows is the intended
review loop (see evals/README.md).

Usage:
    python3 evals/llm_judge.py --country au
    python3 evals/llm_judge.py --country au --limit 10
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_questions,
)
from app.rag.pipeline import RAGPipeline, NOT_COVERED_ANSWER  # noqa: E402
from app.llm.llm_client import LLMClient  # noqa: E402

JUDGE_PROMPT = """You are an evaluation judge for an organic-standards Q&A bot. Grade the answer below.

Question: {question}

Retrieved context given to the bot (empty means the bot answered from general knowledge):
{context}

Bot answer: {answer}

Grade strictly on a JSON-only response in this exact shape:
{{"grounded": 0-2, "citation": 0 or 1, "correct": 0-2, "refusal_ok": true/false, "notes": "short"}}

- grounded: 2 = every factual claim in the answer is supported by the context; 1 = partly supported; 0 = unsupported or contradicting the context (hallucination).
- citation: 1 = the answer references a section/standard/annex; 0 = no reference.
- correct: 2 = the answer is factually right about the standards; 1 = partially right or over-general; 0 = wrong or misleading.
- refusal_ok: true when the bot refused ("not covered") AND that was the right call (context empty or irrelevant); false if it refused despite relevant context, or answered from knowledge with empty context.

JSON only:"""


async def grade(llm: LLMClient, question: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context[:3000], answer=answer)
    out = await llm.generate(prompt=prompt, temperature=0.0, max_tokens=300)
    try:
        match = re.search(r"\{.*\}", out["content"], re.DOTALL)
        return json.loads(match.group(0)) if match else json.loads(out["content"])
    except json.JSONDecodeError:
        return {"grounded": 1, "citation": 0, "correct": 1, "refusal_ok": False,
                "notes": f"judge JSON unparsable: {out['content'][:80]!r}"}


def html_report(country: str, rows: list) -> Path:
    agg = {
        "grounded": sum(r["grades"]["grounded"] for r in rows) / (len(rows) * 2),
        "citation": sum(r["grades"]["citation"] for r in rows) / len(rows),
        "correct": sum(r["grades"]["correct"] for r in rows) / (len(rows) * 2),
    }
    body = []
    for r in rows:
        g = r["grades"]
        row_cls = "bad" if g["grounded"] <= 0 or g["correct"] <= 0 else ""
        body.append(
            f"<tr class='{row_cls}'>"
            f"<td class='q'>{r['question']}</td>"
            f"<td>{r['category']}</td>"
            f"<td>{r['n_sources']}</td>"
            f"<td>{g['grounded']}/2</td><td>{g['citation']}</td><td>{g['correct']}/2</td>"
            f"<td>{'yes' if g['refusal_ok'] else 'no'}</td>"
            f"<td class='a'>{r['answer_excerpt']}</td>"
            f"<td class='n'>{g['notes']}</td></tr>"
        )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Judge report — {country}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem}}h1{{font-size:1.25rem}}
table{{border-collapse:collapse;width:100%;font-size:0.85rem}}
th,td{{border:1px solid #ddd;padding:0.4rem;text-align:left;vertical-align:top}}
th{{background:#f5f5f5}}tr.bad{{background:#fff0f0}}
.q{{max-width:16rem}}.a{{max-width:20rem}}.n{{max-width:14rem}}
</style></head><body>
<h1>AusOrganicBot judge report — {country.upper()} ({len(rows)} questions, generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC})</h1>
<p>Grounded {agg['grounded']:.1%} · Citation {agg['citation']:.1%} · Correct {agg['correct']:.1%}</p>
<table><tr><th>Question</th><th>Category</th><th>#src</th><th>Grounded</th><th>Cited</th><th>Correct</th><th>Refusal OK</th><th>Answer</th><th>Notes</th></tr>
{''.join(body)}</table></body></html>"""
    path = Path(__file__).parent / "data" / f"report_judge_{country}.html"
    path.write_text(html)
    return path


async def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge answer evaluation")
    parser.add_argument("--country", default="au")
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    args = parser.parse_args()

    pipeline = RAGPipeline(country=args.country)
    judge = LLMClient()
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    print(f"Judging {len(questions)} answers for '{args.country}' "
          f"(answer model={pipeline.llm.model}, judge={judge.model})")
    rows = []
    for i, q in enumerate(questions, 1):
        result = await pipeline.query(q["question"], top_k=5)
        context = "\n\n".join(s.content[:600] for s in result["sources"]) if result["sources"] else ""
        grades = await grade(judge, q["question"], context, result["answer"])
        rows.append(
            {
                "question": q["question"],
                "category": q["category"],
                "n_sources": len(result["sources"]),
                "answer_excerpt": result["answer"][:220],
                "grades": grades,
            }
        )
        print(f"[{i}/{len(questions)}] g={grades['grounded']}/2 c={grades['citation']} "
              f"ok={grades['correct']}/2 refuse_ok={grades['refusal_ok']} {q['question'][:45]}")

    raw = Path(__file__).parent / "data" / f"judge_{args.country}.jsonl"
    with open(raw, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = html_report(args.country, rows)
    bad = [r for r in rows if r["grades"]["grounded"] <= 0 or r["grades"]["correct"] <= 0]
    print(f"\nLow-quality answers to spot-check ({len(bad)}):")
    for r in bad:
        print(f"  - {r['question']}  (grounded={r['grades']['grounded']}, "
              f"correct={r['grades']['correct']})")
    print(f"\nRaw grades: {raw}")
    print(f"HTML report: {report}")


if __name__ == "__main__":
    asyncio.run(main())
