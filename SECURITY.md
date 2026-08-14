# Security Policy

## Reporting a Vulnerability

AusOrganicBot answers compliance questions from regulatory documents. If you
find a vulnerability — especially anything that could produce **incorrect or
fabricated compliance answers**, or expose API keys/user data — please report
it privately rather than opening a public issue.

- **Where:** open a GitHub Security Advisory ("Report a vulnerability" on the
  Security tab), or email the maintainers directly.
- **What to include:** a description, steps to reproduce, affected
  version(s)/commit, and any suggested fix.
- **Response:** maintainers will acknowledge within 7 days and aim to
  confirm or decline within 30 days, coordinating a fix and disclosure
  timeline with you.

We ask that you not exploit the issue beyond what is necessary to demonstrate
it, and that you not disclose it publicly until a fix is released.

## Security design notes

- **No user secrets in the repo.** API keys live in `backend/.env`
  (gitignored). Never commit `.env` or any standards PDF under `data/`.
- **API hardening.** Set `API_TOKEN` in `.env` to require a bearer token on
  LLM-backed endpoints, and tune `RATE_LIMIT_PER_MINUTE` for your
  deployment. The default rate limit is 30 queries/IP/minute.
- **The bot is not legal advice.** Answers are generated from retrieved
  excerpts and can be wrong. The refusal guard prevents answering
  compliance-shaped questions from model knowledge when retrieval finds
  nothing, but no guard is perfect. Production deployments should present
  the bot as reference assistance, not certification advice.
- **Document integrity.** Country manifests pin sha256 hashes of standards
  PDFs; `scripts/download_docs.py` and `scripts/ingest.py` refuse to
  proceed on mismatch.

## Supported versions

The latest commit on `main`. Security fixes are released as commits/PRs;
there is no LTS track yet.
