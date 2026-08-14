---
language:
  - en
license: apache-2.0
base_model: openbmb/MiniCPM5-1B
pipeline_tag: text-generation
tags:
  - rag
  - fine-tuned
  - organic-farming
  - certification
  - refusal-aware
library_name: transformers
---

# AusOrganicBot-Reader (MiniCPM5-1B)

Fine-tuned [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) reader
for organic certification Q&A. Answers compliance questions **only from
retrieved context**, cites section numbers, and refuses with a fixed
template when the context does not cover the question.

## Usage

The model expects the AusOrganicBot RAG prompt template as the user turn
(the same string the FastAPI app sends):

```
You are AusOrganicBot, an assistant for organic certification queries.

Applicable standards (answer only from these):
- ...

Guidelines:
- Answer from the provided context. ...

Relevant excerpts from the standards:
<retrieved chunks>

Question: <user question>

Answer:
```

Serve with Ollama (`finetune/Modelfile` in the source repo) or llama.cpp,
and point any OpenAI-compatible client at it. Recommended sampling:
`temperature=0.1`, `top_p=0.95`, `max_tokens=2048`.

## Training

- **Base:** `openbmb/MiniCPM5-1B` (Apache-2.0)
- **Data:** AusOrganicBot Standards QA dataset (CC-BY-4.0) — paraphrased
  QA pairs plus chunk-level refusal negatives; see `dataset/` in the source
  repository
- **Method:** QLoRA (4-bit, r=8–16), SFT with the production RAG prompt
  format; `train.py` in the source repository
- **Alignment:** the negative examples teach honest refusal; the system
  prompt (in `Modelfile`) pins the refusal template

## Intended use

Reference assistance for organic certification standards. **Not legal or
certification advice** — verify answers against current official texts.

## Limitations

- Trained on synthetic data; hallucination risk remains, especially for
  out-of-context questions (mitigated by retrieval gating in the app)
- English-centric; other languages only if the underlying data is
  regenerated per-language
- Standards change — regenerate the dataset and re-fine-tune when a
  country's standards are revised

## Provenance & license

- Model weights (this fine-tune): Apache-2.0
- Training data: CC-BY-4.0 (paraphrased QA pairs; no standards text is
  included)
- Base model: MiniCPM5-1B by OpenBMB (Apache-2.0)
