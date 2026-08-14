"""Shared helpers for the eval harness.

All scripts import the backend from the repo root and reuse its .env LLM
configuration, so no separate setup is required:

    python3 evals/build_eval_set.py --country au
    python3 evals/sweep_threshold.py --country au
    python3 evals/eval_retrieval.py --country au
    python3 evals/llm_judge.py --country au --limit 30
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVALS_DIR = Path(__file__).resolve().parent
EVALS_DATA_DIR = EVALS_DIR / "data"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from app.config import DATA_DIR, RELEVANCE_THRESHOLD  # noqa: E402
from app.rag.retriever import VectorStore  # noqa: E402

EVALS_DATA_DIR.mkdir(exist_ok=True)


def load_questions():
    """The 100-question synthetic compliance bank from scripts/stress_test.py."""
    from stress_test import QUESTIONS

    return QUESTIONS


def gold_path(country: str) -> Path:
    return EVALS_DATA_DIR / f"gold_{country}.jsonl"


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def existing_gold_questions(country: str):
    return {r["question"] for r in read_jsonl(gold_path(country))}


def chunk_signature(metadata: dict) -> str:
    """Compact identity for a chunk: source/filename/page."""
    return (
        f"{metadata.get('source', '?')}/{metadata.get('filename', '?')}"
        f"#p{metadata.get('page_number', '?')}"
    )


def load_chunk_sigs(vs: VectorStore) -> dict:
    """Map chunk signature -> text, for gold-label interpretation."""
    data = vs.collection.get(include=["documents", "metadatas"])
    out = {}
    for text, meta in zip(data.get("documents", []), data.get("metadatas", [])):
        out[chunk_signature(meta)] = text
    return out
