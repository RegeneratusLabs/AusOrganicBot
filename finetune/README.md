# Fine-tuning MiniCPM5-1B as the AusOrganicBot RAG reader

This directory turns the QA dataset (`dataset/`) into a small,
self-hosted compliance-answering model built on OpenBMB's **MiniCPM5-1B**
(1B-class SOTA, Apache-2.0, standard Llama architecture — serves on
Ollama/llama.cpp/vLLM).

## Why this model

- **MiniCPM5-1B** — OpenBMB's current small-model flagship; hybrid
  `<think>` reasoning; official GGUF + fine-tune cookbooks (TRL / Unsloth /
  LLaMA-Factory / ms-swift). Upgrade path within the family: MiniCPM3-4B.
- Training format is **aligned with production inference**: the user turn
  is the app's own RAG prompt template, so the fine-tuned model behaves
  identically when served through `backend`'s OpenAI-compatible client.

## Workflow

```bash
# 1. Dataset (see dataset/README.md)
python3 dataset/generate_qa.py --country au --verify

# 2. Isolated training venv (does not touch the app env)
python3 -m venv finetune/.venv
finetune/.venv/bin/pip install -r finetune/requirements.txt

# 3. Smoke test the pipeline (5 steps)
finetune/.venv/bin/python finetune/train.py \
    --data dataset/qa_au_train.jsonl \
    --output finetune/runs/smoke --max-steps 5 --max-seq-length 1024

# 4. Full run (~1-2h on an RTX 3060 6GB with the settings below)
finetune/.venv/bin/python finetune/train.py \
    --data dataset/qa_au_train.jsonl \
    --output finetune/runs/ausorganicbot-minicpm5 \
    --epochs 2 --lora-r 8 --batch-size 1 --grad-accum 8 --max-seq-length 1536

# 5. Export GGUF + serve via Ollama
finetune/.venv/bin/python finetune/export_gguf.py \
    --model-dir finetune/runs/ausorganicbot-minicpm5/merged \
    --output finetune/runs/ausorganicbot-minicpm5/ausorganicbot-minicpm5.Q4_K_M.gguf
ollama create ausorganicbot -f finetune/Modelfile

# 6. Point the app at the local model (backend/.env)
#    LLM_BASE_URL=http://localhost:11434/v1
#    LLM_MODEL=ausorganicbot
#    THINKING_PARAM=off

# 7. Evaluate the fine-tune vs the hosted baseline
python3 evals/llm_judge.py --country au --limit 50   # once per model config
```

## Hardware

| GPU | Settings |
|---|---|
| 6GB (RTX 3060) | `--lora-r 8 --batch-size 1 --grad-accum 8 --max-seq-length 1536` |
| 12GB+ | `--lora-r 16 --batch-size 2 --grad-accum 4 --max-seq-length 2048` |

CPU-only training is not supported by the QLoRA path; use a cloud GPU or
downsize to MiniCPM4-0.5B with plain LoRA instead.

## Evaluation

The refusal negatives are the quality lever: the fine-tuned model must
*not* answer from parametric knowledge. Compare it against the hosted
baseline with the same judge:

```bash
python3 evals/llm_judge.py --country au --limit 50   # hosted baseline
# ... switch backend/.env to Ollama + re-run with the same --limit
```

Report grounded/citation/correctness deltas — target: grounded ≥ baseline,
refusal_ok strictly improved on low-retrieval questions.

## Files

- `train.py` — Unsloth QLoRA SFT (production-aligned prompt format)
- `export_gguf.py` — merged model → GGUF Q4_K_M via llama.cpp
- `Modelfile` — Ollama packaging with the bot's refusal system prompt
- `MODEL_CARD.md` — the model card to publish alongside the GGUF
