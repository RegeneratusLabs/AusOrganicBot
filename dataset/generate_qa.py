#!/usr/bin/env python3
"""
Generate the SFT dataset for AusOrganicBot from the vector store chunks.

For each chunk the LLM produces:
  - 1-2 "positive" pairs: a farmer-style question whose answer IS supported
    by the chunk, answered with paraphrased text + section citations
  - 1 "negative" pair: a compliance-looking question that is NOT answerable
    from the chunk — the expected answer is the refusal template. These teach
    the fine-tuned model to say "not covered" instead of hallucinating.

Outputs (append-only, resumable — re-run to continue where it stopped):
  - dataset/qa_<country>.jsonl           (all pairs)
  - dataset/qa_<country>_train.jsonl     (90% deterministic split)
  - dataset/qa_<country>_val.jsonl       (10%)

Usage:
    python3 dataset/generate_qa.py --country au
    python3 dataset/generate_qa.py --country au --max-chunks 200 --verify
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

from common import chunk_signature, VectorStore, append_jsonl  # noqa: E402
from app.llm.llm_client import LLMClient  # noqa: E402
from app.prompts import NOT_COVERED_ANSWER  # noqa: E402

GENERATE_PROMPT = """You are creating a training dataset for an organic certification Q&A assistant.

Below is an excerpt from an organic standards document.

Excerpt (from {signature}):
\"\"\"
{chunk}
\"\"\"

Create:
1. TWO "positive" question/answer pairs. Each question must be a realistic farmer/certifier question whose answer IS fully supported by this excerpt. The answer must be paraphrased (do not copy the excerpt verbatim), concise, and cite the section number or annex if one appears in the excerpt.
2. ONE "negative" question: a realistic compliance question that this excerpt does NOT answer (something a farmer might reasonably ask but is not covered here). No answer needed for the negative — the assistant's expected answer is the standard refusal text.

Respond as JSON only, in this exact shape:
{{"positives": [{{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}], "negative": {{"question": "..."}}}}

JSON only:"""

VERIFY_PROMPT = """Does this answer stay fully supported by the excerpt below (no invented rules, no contradictions)? Respond "yes" or "no" only.

Excerpt:
\"\"\"
{chunk}
\"\"\"

Answer: {answer}

yes/no:"""


def parse_generation(content: str):
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group(0)) if match else json.loads(content)
    except json.JSONDecodeError:
        return None


async def verify(llm: LLMClient, chunk: str, answer: str) -> bool:
    out = await llm.generate(
        prompt=VERIFY_PROMPT.format(chunk=chunk[:1500], answer=answer),
        temperature=0.0, max_tokens=10,
    )
    return out["content"].strip().lower().startswith("yes")


async def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset from chunks")
    parser.add_argument("--country", default="au")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = all chunks")
    parser.add_argument("--verify", action="store_true", help="LLM-verify each positive pair")
    args = parser.parse_args()

    out_all = REPO_ROOT / "dataset" / f"qa_{args.country}.jsonl"
    out_train = REPO_ROOT / "dataset" / f"qa_{args.country}_train.jsonl"
    out_val = REPO_ROOT / "dataset" / f"qa_{args.country}_val.jsonl"
    REPO_ROOT.joinpath("dataset").mkdir(exist_ok=True)

    # Resume: chunks already present in the output file are skipped
    done = set()
    if out_all.exists():
        for line in out_all.read_text().splitlines():
            try:
                done.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                pass

    vs = VectorStore(country=args.country)
    data = vs.collection.get(include=["documents", "metadatas"])
    docs = list(zip(data.get("documents", []), data.get("metadatas", [])))
    if args.max_chunks:
        docs = docs[: args.max_chunks]

    todo = [
        (text, meta) for text, meta in docs
        if chunk_signature(meta) + "::" + str(meta.get("chunk_index", "")) not in done
    ]
    llm = LLMClient()
    print(f"Generating pairs for {len(todo)} chunks of '{args.country}' "
          f"({len(done)} already done, model={llm.model})")

    n_pos = n_neg = n_verified = n_dropped = 0
    for i, (chunk, meta) in enumerate(todo, 1):
        sig = chunk_signature(meta)
        chunk_id = f"{args.country}:{sig}::{meta.get('chunk_index', '')}"
        prompt = GENERATE_PROMPT.format(signature=sig, chunk=chunk[:2500])
        out = await llm.generate(prompt=prompt, temperature=0.7, max_tokens=800)
        parsed = parse_generation(out["content"])
        if not parsed or "positives" not in parsed:
            print(f"[{i}/{len(todo)}] unparsable output, skipping")
            continue

        for pair in parsed.get("positives", []):
            question = (pair.get("question") or "").strip()
            answer = (pair.get("answer") or "").strip()
            if not question or not answer:
                continue
            if args.verify and not await verify(llm, chunk, answer):
                n_dropped += 1
                continue
            append_jsonl(
                out_all,
                {
                    "chunk_id": chunk_id,
                    "country": args.country,
                    "source": meta.get("source"),
                    "filename": meta.get("filename"),
                    "page_number": meta.get("page_number"),
                    "edition": meta.get("edition"),
                    "type": "positive",
                    "question": question,
                    "answer": answer,
                    "context": chunk,
                },
            )
            n_pos += 1
            n_verified += 1 if args.verify else 0

        neg = (parsed.get("negative") or {}).get("question", "").strip()
        if neg:
            append_jsonl(
                out_all,
                {
                    "chunk_id": chunk_id,
                    "country": args.country,
                    "source": meta.get("source"),
                    "filename": meta.get("filename"),
                    "page_number": meta.get("page_number"),
                    "edition": meta.get("edition"),
                    "type": "negative",
                    "question": neg,
                    "answer": NOT_COVERED_ANSWER,
                    "context": chunk,
                },
            )
            n_neg += 1

        print(f"[{i}/{len(todo)}] +{n_pos} -{n_neg} (dropped {n_dropped})")

    # deterministic train/val split (90/10 by question hash)
    if out_all.exists():
        train = open(out_train, "w")
        val = open(out_val, "w")
        for line in out_all.read_text().splitlines():
            rec = json.loads(line)
            target = train if hash(rec["question"]) % 10 != 0 else val
            target.write(line + "\n")
        train.close()
        val.close()
        n_train = sum(1 for _ in out_train.read_text().splitlines())
        n_val = sum(1 for _ in out_val.read_text().splitlines())
        print(f"\nDone: {n_pos} positives, {n_neg} negatives "
              f"→ train={n_train}, val={n_val}")
        print(f"  {out_all}\n  {out_train}\n  {out_val}")


if __name__ == "__main__":
    asyncio.run(main())
