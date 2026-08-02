#!/usr/bin/env python3
"""
Synthetic stress test for AusOrganicBot RAG pipeline.

Runs a bank of questions across compliance categories and records:
- retrieval scores (are relevant chunks found?)
- whether sources were returned (did it pass the relevance gate?)
- answer text (is it grounded or a refusal?)
- latency per query

Usage:
    python3 scripts/stress_test.py [--provider opencode|ollama] [--output report.json]

Requires a running backend on PORT (default 8080) configured for the chosen provider,
OR set BACKEND_URL=http://localhost:8080 to target a running instance.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import httpx

DEFAULT_BACKEND = "http://localhost:8080"

# Question bank: (category, question)
QUESTIONS: List[Dict[str, str]] = [
    # --- Input approvals ---
    {"category": "input_approval", "question": "Can I use copper sulfate as a fungicide?"},
    {"category": "input_approval", "question": "Is potassium permanganate permitted for use in organic production?"},
    {"category": "input_approval", "question": "Can I use fish emulsion as a fertiliser?"},
    # --- Buffer zones ---
    {"category": "buffer_zone", "question": "What are the buffer zone requirements between organic and conventional crops?"},
    {"category": "buffer_zone", "question": "How wide should a buffer zone be for broadacre cropping?"},
    # --- Record keeping ---
    {"category": "record_keeping", "question": "What records do I need to keep for organic certification?"},
    {"category": "record_keeping", "question": "How long must I retain certification records?"},
    # --- Transition ---
    {"category": "transition", "question": "How do I transition from conventional to organic farming?"},
    {"category": "transition", "question": "How long is the transition period for land?"},
    # --- Compost (the reported failure) ---
    {"category": "compost", "question": "What are the requirements for compost production on an organic farm?"},
    {"category": "compost", "question": "Can I use compost made from animal manure in organic production?"},
    {"category": "compost", "question": "Is compost allowed as a soil amendment under organic standards?"},
    # --- Labelling ---
    {"category": "labelling", "question": "What are the labelling requirements for certified organic products?"},
    {"category": "labelling", "question": "When can I use the certified organic logo?"},
    # --- Export / import ---
    {"category": "export_import", "question": "What do I need to export certified organic produce overseas?"},
    # --- Nonsense / greeting probes ---
    {"category": "casual", "question": "hello"},
    {"category": "casual", "question": "what is the meaning of life?"},
    {"category": "casual", "question": "thanks for your help"},
]


def run_question(client: httpx.Client, backend: str, q: Dict[str, str]) -> Dict:
    """Run one query against the backend /api/query endpoint."""
    start = time.time()
    try:
        resp = client.post(
            f"{backend}/api/query",
            json={"question": q["question"], "top_k": 5},
            timeout=120,
        )
        latency = (time.time() - start) * 1000
        if resp.status_code != 200:
            return {**q, "ok": False, "error": f"HTTP {resp.status_code}", "latency_ms": latency}
        data = resp.json()
        sources = data.get("sources", [])
        return {
            **q,
            "ok": True,
            "confidence": round(data.get("confidence", 0), 3),
            "n_sources": len(sources),
            "top_scores": [round(s.get("score", 0), 3) for s in sources[:3]],
            "sources_retrieved": len(sources) > 0,
            "answer_excerpt": data.get("answer", "")[:200],
            "latency_ms": round(latency),
        }
    except Exception as e:  # noqa: BLE001 - report any failure
        return {**q, "ok": False, "error": str(e), "latency_ms": round((time.time() - start) * 1000)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test AusOrganicBot RAG pipeline")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Backend base URL")
    parser.add_argument("--output", default="stress_report.json", help="Output report file")
    args = parser.parse_args()

    print(f"Testing backend at {args.backend}")
    print(f"Running {len(QUESTIONS)} synthetic questions...\n")

    results: List[Dict] = []
    with httpx.Client() as client:
        for q in QUESTIONS:
            r = run_question(client, args.backend, q)
            results.append(r)
            status = "✓" if r.get("ok") and r.get("sources_retrieved") else "⚠" if r.get("ok") else "✗"
            print(
                f"{status} [{r.get('category','?')}] {r.get('question','?')[:60]:60} "
                f"conf={r.get('confidence','-'):>6} src={r.get('n_sources',0)} "
                f"top={r.get('top_scores',[])} {r.get('latency_ms','-')}ms"
            )

    report_path = Path(args.output)
    report_path.write_text(json.dumps(results, indent=2))

    # Summary
    ok = sum(1 for r in results if r.get("ok"))
    grounded = sum(1 for r in results if r.get("sources_retrieved"))
    print(f"\nSummary: {ok}/{len(results)} responded, {grounded} returned sources")
    print(f"Report written to {report_path}")

    # Flag low-relevance categories (potential retrieval failures)
    print("\nCategories with low retrieval (investigate):")
    low = [r for r in results if r.get("ok") and not r.get("sources_retrieved")]
    for r in low:
        print(f"  - [{r.get('category')}] {r.get('question')} (conf={r.get('confidence')})")
    if not low:
        print("  none — all queries retrieved sources")


if __name__ == "__main__":
    main()
