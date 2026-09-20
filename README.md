# OrgStandardsBot 🌱

**Open-source AI Q&A for Australian organic and biodynamic certification.**

Ask questions about organic farming standards, input approvals, buffer zones, record-keeping, and certification requirements — and get answers grounded in the actual standards documents via RAG (Retrieval-Augmented Generation). The AI searches NASAA, ACO, and National Standard PDFs for relevant excerpts before answering, with source citations.

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- An API key for any OpenAI-compatible LLM provider

### 1. Clone and configure

```bash
git clone <repo-url>   # e.g. https://github.com/your-org/OrgStandardsBot
cd OrgStandardsBot/backend
cp .env.example .env
# Edit .env and set your LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL
```

**Supported providers** — any OpenAI-compatible API:

| Provider | Base URL | Model example |
|----------|----------|---------------|
| DeepSeek (OpenCode Go) | `https://opencode.ai/zen/go/v1` | `deepseek-v4-flash` |
| DeepSeek (direct) | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Ollama (local, free) | `http://localhost:11434/v1` | `qwen2.5:0.5b` |
| LM Studio (local) | `http://localhost:1234/v1` | `local-model` |

> **Local testing without an API key:** `ollama pull qwen2.5:0.5b`, set `LLM_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=qwen2.5:0.5b` — see [SETUP.md](SETUP.md) for tested quality expectations.

### 2. Install backend

```bash
pip install -r requirements.txt
```

### 3. Download standards documents

```bash
python scripts/download_docs.py
```

Downloads + sha256-verifies each installed country's standards (see the manifest in `data/<cc>/manifest.json`).

Or manually place PDFs in `data/au/nasaa/`, `data/au/aco/`, `data/au/national/`

### 4. Ingest documents into the vector store

```bash
python scripts/ingest.py             # every installed country
python scripts/ingest.py --country au
```

### 5. Start the app

```bash
./start.sh
```

Or start services separately:

```bash
# Terminal 1 — backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Visit **http://localhost:3000** and start asking questions.

## OpenBMB stack (the default runtime)

The project runs fully on the [OpenBMB](https://www.openbmb.cn) ecosystem —
no hosted LLM required:

- **Reader model** — [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
  (Apache-2.0, 1B-class SOTA, hybrid thinking), fine-tuned as the bot's RAG
  reader: `ollama create orgstandardsbot -f finetune/Modelfile` (see
  [`finetune/`](finetune/README.md)). Set
  `LLM_BASE_URL=http://localhost:11434/v1` + `LLM_MODEL=orgstandardsbot`.
- **Embeddings** — [MiniCPM-Embedding-Light](https://huggingface.co/openbmb/MiniCPM-Embedding-Light)
  (440M, 1024-dim, zh+en, dense+sparse): set `EMBEDDING_FAMILY=minicpm`
  and re-run `scripts/ingest.py`.
- **RAG framework** — [UltraRAG](https://github.com/OpenBMB/UltraRAG)
  (Apache-2.0, MCP-based): the same pipeline expressed as an UltraRAG
  experiment in [`ultrarag/`](ultrarag/README.md) with Milvus.

Any OpenAI-compatible API remains supported for evaluation (see
`backend/.env.example`), but the shipped default is the fully local
OpenBMB stack.

## Evaluation, dataset, fine-tuning

| Directory | What it does |
|---|---|
| [`evals/`](evals/README.md) | Gold-set retrieval eval, threshold sweep, LLM-as-judge answer grading (silver baseline: AU recall@5 95.7%, gate accuracy 95% — model-labelled, see the caveats in the evals README) |
| [`dataset/`](dataset/README.md) | Chunk → QA pair generation (with refusal negatives), CC-BY-4.0, HF-publishable |
| [`finetune/`](finetune/README.md) | QLoRA fine-tune of MiniCPM5-1B as the RAG reader + GGUF export + Ollama Modelfile |
| [`ultrarag/`](ultrarag/README.md) | UltraRAG YAML pipeline (MiniCPM-Embedding-Light + MiniCPM5-1B + Milvus) |

## Multi-country support

The bot answers from **one country's standards per query** — each country is an isolated, versioned set of documents:

```
data/
├── au/
│   ├── manifest.json        # which standards, which editions, sha256 integrity
│   ├── nasaa/*.pdf          # one dir per standards body
│   ├── aco/*.pdf
│   └── national/*.pdf
├── nz/
│   └── manifest.json        # add countries as needed
└── vectorstore/             # one ChromaDB collection per country
```

**Adding a country** (CLI only — no admin UI):

```bash
python scripts/add_country.py --country nz --name "New Zealand"
# 1. edit data/nz/manifest.json — list each standards body + edition
# 2. set "license" + "redistribute" per source (see Data licensing below)
# 3. place PDFs under data/nz/<source-id>/  (or set "url" and re-run download_docs.py)
# 4. python scripts/update_hashes.py --country nz   # pin file hashes
# 5. python scripts/ingest.py --country nz
```

The country then appears in the chat UI's country selector automatically.

Design points:

- **Edition provenance** — every source declares `edition` + `status` (`active` / `superseded`). Superseded editions stay on disk but are excluded from retrieval, and answers cite the edition (e.g. "ACO Standard 2023, Section 4.7.19").
- **Integrity** — `download_docs.py` and `ingest.py` hard-fail on sha256 mismatch, so a silently changed PDF can't poison answers.
- **Per-country tuning** — each manifest can override `relevance_threshold` and `embedding_family` (e.g. `minicpm` for OpenBMB's multilingual MiniCPM-Embedding-Light, 1024-dim zh+en) without touching other countries.
- **Data licensing** — the code is MIT, but standards documents remain the copyright of their issuing bodies. Standards PDFs and generated HTML/vector stores are **never committed** to the repo (fetched at provisioning time from each body's official site). Each manifest source declares its `license` and whether this deployment may **redistribute** full copies (`redistribute: true`). Full-copy endpoints — `GET /api/documents/{country}/{source}/pdf`, `/html`, `/page/{page}` — return `403` unless a source declares redistribution rights (e.g. CC-BY, Crown/OGL, public domain). Retrieval and citation always work regardless. See the per-source flags in `data/*/manifest.json` and [SECURITY.md](SECURITY.md).

## How it works — RAG pipeline

1. You ask a question (e.g. "What are buffer zone requirements?") and pick a country (defaults to Australia)
2. The question is converted to a vector embedding using OpenBMB MiniCPM-Embedding-Light (`EMBEDDING_FAMILY=minicpm`, 1024-dim) — fully local
3. ChromaDB finds the most semantically similar chunks from that country's standards collection
4. Those excerpts are injected into the LLM prompt with source + edition citations. If the top match scores below the relevance threshold (0.40 default, per-country overridable), a low-confidence note is added so the model answers only if the excerpts genuinely cover the question
5. The MiniCPM5-1B reader (fine-tuned) answers from the retrieved context — and declines warmly with the closest related guidance if the context doesn't cover the question. Greetings/thanks/off-topic go to a conversational path. Multi-turn history (last few turns) is included so follow-ups resolve in context
6. Answers stream in real-time with markdown formatting (bold, lists, code)

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Ask a compliance question (non-streaming). Body: `{question, top_k?, country?}` |
| `/api/query/stream` | POST | Stream a response (reasoning + answer tokens) |
| `/api/input-check` | POST | Check if a product input is approved |
| `/api/status` | GET | System status + installed countries + editions |
| `/api/sources` | GET | Loaded document sources (per country) |
| `/api/documents` | GET | Document metadata (per country), incl. `license` + `redistributable` |
| `/api/documents/{country}/{source}/html` | GET | Full HTML rendering of a standard (403 unless source is `redistribute: true`) |
| `/api/documents/{country}/{source}/page/{page}` | GET | Single page HTML (403 unless `redistribute: true`) |
| `/api/documents/{country}/{source}/pdf` | GET | Original PDF (403 unless `redistribute: true`) |

### POST /api/query (streaming)

```json
{"type": "reasoning", "content": "Thinking about the question..."}
{"type": "content", "content": "Section 4.1.2 states that buffer zones..."}
{"done": true}
```

## Configuration

Set these in `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | Your LLM provider API key |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | API endpoint |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `PORT` | `8080` | Backend server port |
| `THINKING_ENABLED` | `false` | Enable chain-of-thought reasoning |
| `THINKING_PARAM` | `off` | Send DeepSeek thinking param: `auto`/`force`/`off` (off for OpenBMB) |
| `DEFAULT_TEMPERATURE` | `0.2` | Lower = more factual |
| `DEFAULT_MAX_TOKENS` | `2048` | Max response length |
| `DEFAULT_COUNTRY` | `au` | Country queried when the client doesn't specify one |
| `EMBEDDING_FAMILY` | `minicpm` | `minicpm` (OpenBMB MiniCPM-Embedding-Light, default) or `all-minilm` |
| `RELEVANCE_THRESHOLD` | `0.40` | Relevance gate (below → low-confidence note); per-country override in each manifest |
| `API_TOKEN` | — | If set, requests require `Authorization: Bearer <token>` |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max LLM queries per IP per minute (`0` = unlimited) |

## Project Structure

```
OrgStandardsBot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI server (auth, rate limiting, country routing)
│   │   ├── config.py            # Configuration
│   │   ├── schemas.py           # Pydantic models
│   │   ├── llm/
│   │   │   └── llm_client.py    # OpenAI-compatible LLM client
│   │   ├── rag/
│   │   │   ├── pipeline.py      # RAG orchestration + relevance gating + refusal guard
│   │   │   ├── retriever.py     # ChromaDB wrapper (one collection per country)
│   │   │   └── embeddings.py    # Sentence transformers (all-MiniLM / MiniCPM)
│   │   └── data/
│   │       ├── manifest.py      # Country manifests (editions, sha256 integrity)
│   │       ├── loader.py        # PDF ingestion (country-aware)
│   │       └── html_converter.py
│   ├── tests/                   # pytest suite
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Chat interface (streaming + markdown + country selector)
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── lib/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   └── package.json
├── scripts/
│   ├── add_country.py           # Scaffold a new country manifest
│   ├── download_docs.py         # Manifest-driven, sha256-verified downloads
│   ├── ingest.py                # Per-country ingestion
│   ├── update_hashes.py         # Pin sha256 hashes in a manifest
│   ├── validate_manifest.py     # Schema + integrity checks
│   ├── reindex_with_html.py
│   ├── stress_test.py           # Synthetic question stress testing
│   └── test_stream_parser.py    # Streaming parser regression test
├── .github/
│   ├── ISSUE_TEMPLATE/          # Bug + feature request templates
│   └── pull_request_template.md
├── CONTRIBUTING.md
├── data/                        # Standards PDFs + manifests (not committed — see download_docs.py)
│   └── <country_code>/          # e.g. au/, nz/
│       ├── manifest.json
│       └── <source>/*.pdf
├── start.sh
├── LICENSE
└── README.md
```

## License

MIT for the code — see [LICENSE](LICENSE). Derived QA datasets carry
CC-BY-4.0. Standards documents remain the copyright of their issuing
bodies; each manifest source declares its `license` and whether this
deployment may redistribute full copies (`redistribute`). Sources that
don't permit redistribution (e.g. the ACO standard) are used for
retrieval and citation only — the full-copy endpoints return 403. See
[SECURITY.md](SECURITY.md), [dataset/README.md](dataset/README.md) and
the per-source flags in `data/*/manifest.json`.

---

**Stack:** FastAPI, Next.js 16, ChromaDB, sentence-transformers, any OpenAI-compatible LLM
**OpenBMB:** MiniCPM5-1B · MiniCPM-Embedding-Light · UltraRAG
**Built for:** Australian organic farmers, certifiers, and industry bodies — now multi-country
