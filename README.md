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
git clone https://github.com/your-org/AusOrganicBot
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

Or manually place PDFs in `data/nasaa/`, `data/aco/`, `data/sxo/`, `data/national/`

### 4. Ingest documents into the vector store

```bash
python scripts/ingest.py
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

## How it works — RAG pipeline

1. You ask a question (e.g. "What are buffer zone requirements?")
2. The question is converted to a vector embedding using `all-MiniLM-L6-v2` (runs locally)
3. ChromaDB finds the most semantically similar chunks from the standards PDFs
4. If the top match scores above 0.50 relevance, those excerpts are injected into the LLM prompt with source citations
5. If nothing relevant is found (greetings, off-topic), the LLM responds conversationally instead
6. Answers stream in real-time with markdown formatting (bold, lists, code)

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Ask a compliance question (non-streaming) |
| `/api/query/stream` | POST | Stream a response (reasoning + answer tokens) |
| `/api/input-check` | POST | Check if a product is approved |
| `/api/status` | GET | System status |
| `/api/sources` | GET | Loaded document sources |

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

## Project Structure

```
AusOrganicBot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI server
│   │   ├── config.py            # Configuration
│   │   ├── schemas.py           # Pydantic models
│   │   ├── llm/
│   │   │   └── llm_client.py    # OpenAI-compatible LLM client
│   │   ├── rag/
│   │   │   ├── pipeline.py      # RAG orchestration + relevance gating
│   │   │   ├── retriever.py     # ChromaDB wrapper
│   │   │   └── embeddings.py    # Sentence transformers
│   │   └── data/
│   │       ├── loader.py        # PDF ingestion
│   │       └── html_converter.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Chat interface (streaming + markdown)
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── lib/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   └── package.json
├── scripts/
│   ├── download_docs.py
│   ├── ingest.py
│   ├── reindex_with_html.py
│   ├── stress_test.py           # Synthetic question stress testing
│   └── test_stream_parser.py    # Streaming parser regression test
├── .github/
│   ├── ISSUE_TEMPLATE/          # Bug + feature request templates
│   └── pull_request_template.md
├── CONTRIBUTING.md
├── data/                        # Standards PDFs (not committed — see download_docs.py)
│   └── vectorstore/             # ChromaDB (auto-generated)
├── start.sh
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE)

---

**Stack:** FastAPI, Next.js 16, ChromaDB, sentence-transformers, any OpenAI-compatible LLM
**Built for:** Australian organic farmers, certifiers, and industry bodies
