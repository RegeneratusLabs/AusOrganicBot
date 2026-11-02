# OrgStandardsBot — Setup Guide

Open-source AI Q&A for Australian organic and biodynamic certification. Uses RAG (Retrieval-Augmented Generation) to answer questions by searching actual standards documents.

## Prerequisites

- **Python 3.10+** — `python3 --version`
- **Node.js 18+** — `node --version`
- An API key for any OpenAI-compatible LLM provider

## Step 1: Configure API Key

```bash
cd backend
cp .env.example .env
```

Edit `backend/.env`:

```bash
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

Works with any OpenAI-compatible API. Examples:

| Provider | `LLM_BASE_URL` | `LLM_MODEL` |
|----------|----------------|-------------|
| DeepSeek (OpenCode Go) | `https://opencode.ai/zen/go/v1` | `deepseek-v4-flash` |
| DeepSeek (direct) | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Ollama (local, free) | `http://localhost:11434/v1` | `qwen2.5:0.5b` |

### Local model option (OpenBMB MiniCPM5-1B via Ollama) — the default

Fully offline, no API key, on OpenBMB's 1B-class SOTA model:

```bash
# Pull the official GGUF from Hugging Face through Ollama
ollama pull hf.co/openbmb/MiniCPM5-1B-GGUF

# Create the bot-configured model (template verified working)
ollama create orgstandardsbot -f finetune/Modelfile          # fine-tuned reader
ollama create orgstandardsbot-minicpm5 -f finetune/Modelfile.base  # base model
#   — see finetune/README.md for the Modelfile; the MiniCPM5 template +
#     stop tokens are required, or generation degenerates
```

Then in `backend/.env`:

```bash
LLM_API_KEY=
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=orgstandardsbot
THINKING_PARAM=off
EMBEDDING_FAMILY=minicpm   # MiniCPM-Embedding-Light (OpenBMB) — then re-ingest
```

**Verified end-to-end**: "How wide should a buffer zone be for broadacre
cropping?" → *"Section 4.7.19 (Document 1) states that buffer zones should
be no less than 15 metres for broadacre cropping activities."* (grounded,
cited, streamed through the app; the base model emits a `<think>` block —
the fine-tuned variant is SFT'd to skip it, see `finetune/`).

For the fully local OpenBMB stack, also switch embeddings
(`EMBEDDING_FAMILY=minicpm` + re-ingest) and see `ultrarag/` for the
UltraRAG pipeline.

**Expected quality**: answers grounded in the standards with sources on
straightforward compliance questions. The model is small — expect shorter,
less detailed answers than a hosted model, and 10–40 s latency per query
on CPU/edge GPU. Use it for offline/demo deployments; use the hosted
provider or the fine-tuned GGUF for production.
| Local (LM Studio) | `http://localhost:1234/v1` | `local-model` |
| Local (Ollama) | `http://localhost:11434/v1` | `llama3` |

## Step 2: Install Backend

```bash
cd backend
pip install -r requirements.txt
```

## Step 3: Download Standards Documents

```bash
python scripts/download_docs.py
```

Downloads and sha256-verifies the standards listed in each country's manifest.

Or manually place PDFs in:

| Source | Directory |
|--------|-----------|
| NASAA Organic Standards | `data/au/nasaa/` |
| ACO Certification Standards | `data/au/aco/` |
| National Standard | `data/au/national/` |

Currently indexed: **NASAA** (120 pages), **ACO** (113 pages), **National Standard editions 3.7 + 3.8** (152 pages).

## Step 4: Ingest Documents

```bash
python scripts/ingest.py               # every installed country
python scripts/ingest.py --country au  # one country
```

This will:
1. Validate the country manifest (schema + sha256 integrity — hard fail on mismatch)
2. Load all active PDFs from `data/<cc>/<source>/` (superseded editions are skipped)
3. Split into chunks (512 chars with overlap, minimum 20 chars)
4. Generate embeddings via OpenBMB MiniCPM-Embedding-Light (`EMBEDDING_FAMILY=minicpm`, 1024-dim) — runs locally; first run downloads the model
5. Store in a per-country ChromaDB collection

Expected output: `✓ Ingestion complete! Total chunks: ~2000`

To regenerate with HTML page previews:
```bash
python scripts/reindex_with_html.py --country au
```

## Adding a country

```bash
python scripts/add_country.py --country nz --name "New Zealand"
```

Then:
1. Edit `data/nz/manifest.json` — for each standards body, fill in `id`, `name`, `edition`, `url` or `local_path`, `status: "active"`
2. Per source, set `license` (the document's actual licence) and `redistribute`
   — full copies (PDF / HTML / page endpoints) are served **only** when
   `redistribute: true`. Leave `false` for "all rights reserved" documents
   (e.g. the ACO standard); retrieval and citation still work.
3. Place PDFs under `data/nz/<source-id>/` (or set `url` and run `scripts/download_docs.py --country nz`)
4. `python scripts/update_hashes.py --country nz` — pin sha256 hashes
5. `python scripts/ingest.py --country nz`

The country appears in the chat UI's selector automatically. Each country can
override `relevance_threshold` and `embedding_family` in its manifest.

## Step 5: Start the App

### One-Click Start

```bash
./start.sh
```

### Manual Start

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install   # first time only
npm run dev
```

Open **http://localhost:3000**

> Backend defaults to port 8080. Override with `PORT=8000 ./start.sh` or set `PORT=8081` in `backend/.env`.

## Streaming Responses

The frontend uses the streaming endpoint by default. Answers appear token-by-token as the model generates them. If `THINKING_ENABLED=true`, a collapsible "🧠 Show thinking" section shows the model's chain-of-thought reasoning.

## RAG Relevance Gating

The pipeline checks the top retrieval score against the relevance threshold (**0.40** default, per-country overridable):

- **≥ 0.40**: RAG path — LLM answers from document excerpts with source citations
- **< 0.40 but compliance-shaped**: RAG path with a low-confidence note — the model answers if the excerpts genuinely cover the question, otherwise declines warmly with the closest related guidance
- **< 0.40 and not compliance-shaped** (greetings, off-topic, thanks): Conversational path — the model responds naturally

Multi-turn: the last few turns of the conversation are included in the prompt, so follow-ups like "and how wide?" resolve against earlier context.

## Sample Questions

Once running, try:

1. "What are buffer zone requirements between organic and conventional crops?"
2. "Can I use copper sulfate as a fungicide?"
3. "What records do I need to keep for certification?"
4. "How do I transition from conventional to organic farming?"
5. "hello" (should get a warm greeting, not a standards refusal)

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "LLM not connected" | Check `LLM_API_KEY` in `.env`, verify `LLM_BASE_URL` is correct |
| Backend won't start | Port in use? Try `PORT=8081 ./start.sh` |
| No documents found | Run `python scripts/ingest.py` |
| "0 standards loaded" | Backend running? PDFs in `data/`? |
| Frontend won't start | `cd frontend && npm install` then try again |

## Configuration

All settings in `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | API key |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | LLM API endpoint |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `PORT` | `8080` | Backend port |
| `THINKING_ENABLED` | `false` | Show chain-of-thought reasoning |
| `DEFAULT_TEMPERATURE` | `0.1` | Lower = more factual |
| `DEFAULT_MAX_TOKENS` | `2048` | Response length |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
