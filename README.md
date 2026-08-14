# AusOrganicBot 🌱

**Open-source AI Q&A for Australian organic and biodynamic certification.**

Ask questions about organic farming standards, input approvals, buffer zones, record-keeping, and certification requirements — and get answers grounded in the actual standards documents via RAG (Retrieval-Augmented Generation). The AI searches NASAA, ACO, and National Standard PDFs for relevant excerpts before answering, with source citations.

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- An API key for any OpenAI-compatible LLM provider

### 1. Clone and configure

```bash
git clone <repo-url>   # e.g. https://github.com/your-org/AusOrganicBot
cd AusOrganicBot/backend
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
# 2. place PDFs under data/nz/<source-id>/  (or set "url" and re-run download_docs.py)
# 3. python scripts/update_hashes.py --country nz   # pin file hashes
# 4. python scripts/ingest.py --country nz
```

The country then appears in the chat UI's country selector automatically.

Design points:

- **Edition provenance** — every source declares `edition` + `status` (`active` / `superseded`). Superseded editions stay on disk but are excluded from retrieval, and answers cite the edition (e.g. "ACO Standard 2023, Section 4.7.19").
- **Integrity** — `download_docs.py` and `ingest.py` hard-fail on sha256 mismatch, so a silently changed PDF can't poison answers.
- **Per-country tuning** — each manifest can override `relevance_threshold` and `embedding_family` (e.g. `minicpm` for OpenBMB's multilingual MiniCPM-Embedding-Light, 1024-dim zh+en) without touching other countries.
- **Data licensing** — the code is MIT, but standards documents remain the copyright of their issuing bodies. Countries' PDFs are never committed or redistributed; manifests and scripts only.

## How it works — RAG pipeline

1. You ask a question (e.g. "What are buffer zone requirements?") and pick a country (defaults to Australia)
2. The question is converted to a vector embedding using `all-MiniLM-L6-v2` (runs locally; swap to OpenBMB MiniCPM-Embedding-Light via `EMBEDDING_FAMILY=minicpm`)
3. ChromaDB finds the most semantically similar chunks from that country's standards collection
4. If the top match scores above the relevance threshold (0.50 default, per-country overridable), those excerpts are injected into the LLM prompt with source + edition citations
5. If nothing relevant is found, the bot either responds conversationally or — for compliance-shaped questions — refuses with "consult your certifier" rather than answering from model knowledge (hallucination guard)
6. Answers stream in real-time with markdown formatting (bold, lists, code)

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Ask a compliance question (non-streaming). Body: `{question, top_k?, country?}` |
| `/api/query/stream` | POST | Stream a response (reasoning + answer tokens) |
| `/api/input-check` | POST | Check if a product input is approved |
| `/api/status` | GET | System status + installed countries + editions |
| `/api/sources` | GET | Loaded document sources (per country) |
| `/api/documents` | GET | Document metadata (per country) |
| `/api/documents/{country}/{source}/html` | GET | Full HTML rendering of a standard |
| `/api/documents/{country}/{source}/pdf` | GET | Original PDF |

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
| `THINKING_PARAM` | `auto` | Send DeepSeek thinking param: `auto`/`force`/`off` |
| `DEFAULT_TEMPERATURE` | `0.1` | Lower = more factual |
| `DEFAULT_MAX_TOKENS` | `2048` | Max response length |
| `DEFAULT_COUNTRY` | `au` | Country queried when the client doesn't specify one |
| `EMBEDDING_FAMILY` | `all-minilm` | `all-minilm` (default) or `minicpm` (OpenBMB MiniCPM-Embedding-Light) |
| `RELEVANCE_THRESHOLD` | `0.50` | Global relevance gate; per-country override in each manifest |
| `API_TOKEN` | — | If set, requests require `Authorization: Bearer <token>` |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max LLM queries per IP per minute (`0` = unlimited) |

## Project Structure

```
AusOrganicBot/
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

MIT — see [LICENSE](LICENSE)

---

**Stack:** FastAPI, Next.js 16, ChromaDB, sentence-transformers, any OpenAI-compatible LLM
**Built for:** Australian organic farmers, certifiers, and industry bodies
