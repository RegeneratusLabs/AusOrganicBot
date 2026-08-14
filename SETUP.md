# AusOrganicBot — Setup Guide

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

### Local model option (Ollama) — tested

For a fully offline, no-API-key setup:

```bash
ollama pull qwen2.5:0.5b
ollama serve   # or ensure the Ollama app is running
```

Then in `backend/.env`:

```bash
LLM_API_KEY=
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:0.5b
THINKING_PARAM=off   # Ollama rejects the DeepSeek thinking param (auto handles this)
```

**Expected quality** (tested with qwen2.5:0.5b): answers grounded in the standards with sources on straightforward compliance questions (buffer zones, input approvals, record-keeping). The model is small — expect shorter, less detailed answers than a hosted model, and 6–26 s latency per query on CPU. Use it for development/testing; consider a larger model (`qwen2.5:7b`) or a hosted provider for production.
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
4. Generate embeddings via all-MiniLM-L6-v2 (runs locally)
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
2. Place PDFs under `data/nz/<source-id>/` (or set `url` and run `scripts/download_docs.py --country nz`)
3. `python scripts/update_hashes.py --country nz` — pin sha256 hashes
4. `python scripts/ingest.py --country nz`

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

Not every query hits the vector store. The pipeline checks if the top search result scores above **0.50** relevance:
- **≥ 0.50**: RAG path — LLM answers from document excerpts with source citations
- **< 0.50**: Conversational path — LLM responds naturally (greetings, off-topic, thanks)

This prevents the "I don't have that information" cold response for non-compliance questions.

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
