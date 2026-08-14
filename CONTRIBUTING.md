# Contributing to AusOrganicBot

Thanks for your interest in AusOrganicBot! This project aims to make Australian organic certification knowledge accessible to farmers, producers, and certifiers. Here's how to help.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/AusOrganicBot.git
   cd AusOrganicBot
   ```
3. Follow [SETUP.md](SETUP.md) to get a local instance running
4. Create a feature branch: `git checkout -b feat/your-feature`

## Development Workflow

- **Backend** (`backend/`): FastAPI + RAG pipeline. Run the test suite before submitting: `python3 -m pytest backend/tests`
- **Frontend** (`frontend/`): Next.js chat UI. Verify with `npm run lint && npm run build` before submitting.
- **Documentation**: Keep `README.md` and `SETUP.md` in sync with any behavior changes.

## Code Style

- **Python**: Follow PEP 8, keep functions small and focused
- **TypeScript/React**: Follow existing patterns in `app/page.tsx`
- **No secrets**: Never commit `.env`, API keys, or tokens. The `.env` file is gitignored for a reason.
- **No standards PDFs**: Do not commit files under `data/` — they're fetched by `scripts/download_docs.py`

## Testing Changes

Before submitting a PR, at minimum:

1. Backend imports cleanly: `cd backend && python3 -c "from app.main import app"`
2. Frontend builds: `cd frontend && npm run build`
3. A test query returns a grounded answer: `curl -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' -d '{"question":"What are buffer zone requirements?"}'`

## Submitting Changes

1. Commit with a clear message (e.g. `feat: add fertiliser input category`)
2. Push and open a pull request
3. Fill out the pull request template — describe the change and how you tested it

## Reporting Issues

Use the issue templates: **Bug Report** for things that break, **Feature Request** for improvements. Include the error message, the query you asked, and what you expected.

## Code of Conduct

Be respectful. This project serves a community of farmers and certifiers — keep discussions constructive and professional.
