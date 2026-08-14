# Evaluation harness

Answer reliability is the product. Four tools grade the system end-to-end,
using the 100-question synthetic bank from `scripts/stress_test.py` and
the same LLM configured in `backend/.env`.

## Pipeline

```bash
python3 evals/run_all.py --country au            # one-shot: all four steps
python3 evals/run_all.py --country au --limit 30 # judge subset only
```

Individual steps (all resumable / re-runnable):

| Step | Tool | What it measures |
|---|---|---|
| 1 | `build_eval_set.py` | LLM-annotated gold labels: which retrieved chunks answer each question, and whether the question is answerable at all (silver, human spot-checkable) |
| 2 | `sweep_threshold.py` | Best relevance-gate threshold by gate F1 vs the gold set |
| 3 | `eval_retrieval.py` | recall@5, gate accuracy, per-category breakdown |
| 4 | `llm_judge.py` | End-to-end answer grades: grounded (0-2), citation (0/1), correct (0-2), refusal_ok — with an HTML report for human review |

## Baseline results (all 5 countries, Aug 2026)

Retrieval (recall@5 / gate accuracy at threshold 0.50):

| Country | Answerable | recall@5 | Gate acc |
|---|---|---|---|
| AU | 93/100 | **100%** | 96% |
| US | 77/100 | 98.7% | 76% |
| NZ | 24/100 | 95.8% | 63% |
| EU | 76/100 | 96.1% | 82% |
| UK | 78/100 | 93.6% | 85% |

"Answerable" varies by country — NZ's two documents (framework Act +
Regulations) genuinely cover fewer of the AU-centric probe questions, so
refusals there are the *correct* behavior, which is why gate accuracy
differs rather than retrieval failing.

Answer quality (LLM judge, AU, 100 questions, hosted model):
**grounded 82.5% · citation 82% · correct 85.5%**. The 13 flagged rows are
dominated by false refusals where sources were retrieved but the model
judged them not-covered (waterways buffers, digital records, inspection
details, audit frequency) — the known retrieval/synthesis gaps from the
original stress report, now quantified and traceable per question in
`evals/data/report_judge_au.html`.

Threshold sweep: 0.40–0.45 marginally improves gate F1 (0.989 vs 0.978)
but risks routing casual questions into the RAG path — 0.50 stays the
default.

Known retrieval gap (reproduced by the harness): "vermicompost" —
genuinely in the standards but missed at top-5 (chunking/coverage issue).

## Review workflow

The judge flags low-scoring rows. The intended loop:

1. Run `llm_judge.py` → open `evals/data/report_judge_<cc>.html`
2. Human spot-checks flagged rows (red-highlighted)
3. Fix prompts / chunking / threshold; re-run — the same gold set keeps
   results comparable over time

## Publishing

`evals/data/gold_<cc>.jsonl` is structured for the Hugging Face Hub
(CC-BY-4.0): question + category + answerable + gold chunk references —
no copyrighted text is included. See `dataset/dataset_card.md` for the
publishing checklist.
