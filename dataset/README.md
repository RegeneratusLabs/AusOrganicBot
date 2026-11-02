# OrgStandardsBot QA Dataset (SFT)

## What this is

Synthetic question/answer pairs derived from organic certification standards,
for fine-tuning a small RAG reader model (MiniCPM5-1B) to answer compliance
questions **only from retrieved context** and to refuse when the context does
not cover the question.

- **positives**: question → paraphrased answer citing section numbers,
  each fully supported by its source chunk
- **negatives**: compliance-shaped questions *not* answerable from the chunk,
  whose expected answer is the refusal template — these teach honest refusal
  and are the anti-hallucination backbone of the fine-tune

## Format

JSONL, one record per line:

```json
{
  "chunk_id": "au:aco/australian-certified-organic-standard-2023.pdf#p34::12",
  "country": "au",
  "source": "aco",
  "filename": "australian-certified-organic-standard-2023.pdf",
  "page_number": 34,
  "edition": "2023",
  "type": "positive",
  "question": "How wide should a broadacre buffer zone be?",
  "answer": "Section 4.7.19 states buffer zones for broadacre cropping should be no less than 15 metres...",
  "context": "4.7.19 Buffer zone width ..."
}
```

`context` is the retrieved chunk used to generate the pair. It is **not**
part of the published dataset card artifacts by default — see Licensing.

## Generation

```bash
python3 dataset/generate_qa.py --country au                # all chunks, resumable
python3 dataset/generate_qa.py --country au --max-chunks 200 --verify
```

- Runs against the local vector store (no server needed)
- Appends to `qa_<country>.jsonl` — re-running resumes where it stopped
- `--verify` re-checks every positive answer against its chunk and drops
  unsupported ones
- Deterministic 90/10 train/val split by question hash

## Licensing

- **Dataset license: CC-BY-4.0** — see `dataset/LICENSE.txt`.
- The underlying standards documents remain the copyright of their issuing
  bodies and are **not included** in this dataset. Answers are paraphrased;
  the `context` field (verbatim source excerpts) must be stripped before
  publishing to the Hugging Face Hub. Use `dataset/strip_context.py` to
  produce the publishable variant (`qa_*_publish.jsonl`).
- Publishing checklist: strip `context`, add the HF dataset card
  (`dataset/dataset_card.md`), include CC-BY-4.0, attribute the issuing
  bodies in the card's provenance section.

## Known limitations

- Synthetic data inherits generator-model biases; community/human review
  is expected (see the review workflow in `evals/README.md`)
- Negative pairs are chunk-level only; multi-chunk reasoning is not covered
- Quality scales with the generator LLM — regenerate with a stronger model
  when available
