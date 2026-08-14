#!/usr/bin/env python3
"""
Synthetic stress test for AusOrganicBot RAG pipeline.

Runs a bank of questions across compliance categories and records:
- retrieval scores (are relevant chunks found?)
- whether sources were returned (did it pass the relevance gate?)
- answer text (is it grounded or a refusal?)
- latency per query

Usage:
    python3 scripts/stress_test.py [--backend URL] [--output report.json]
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

# ---------------------------------------------------------------------------
# Question bank — 100 synthetic questions across compliance categories
# ---------------------------------------------------------------------------

QUESTIONS: List[Dict[str, str]] = [
    # --- Input approvals (10) ---
    {"category": "input_approval", "question": "Can I use copper sulfate as a fungicide?"},
    {"category": "input_approval", "question": "Is potassium permanganate permitted for use in organic production?"},
    {"category": "input_approval", "question": "Can I use fish emulsion as a fertiliser?"},
    {"category": "input_approval", "question": "Is pyrethrin allowed for pest control in organic farming?"},
    {"category": "input_approval", "question": "Can I apply neem oil to my organic crops?"},
    {"category": "input_approval", "question": "Is sodium bicarbonate permitted for fungal disease control?"},
    {"category": "input_approval", "question": "Can sulphur be used as a fungicide in organic production?"},
    {"category": "input_approval", "question": "Is hydrogen peroxide allowed as a disinfectant on organic farms?"},
    {"category": "input_approval", "question": "Can I use diatomaceous earth for pest control?"},
    {"category": "input_approval", "question": "Is rotenone permitted as an insecticide under organic standards?"},
    # --- Buffer zones (8) ---
    {"category": "buffer_zone", "question": "What are the buffer zone requirements between organic and conventional crops?"},
    {"category": "buffer_zone", "question": "How wide should a buffer zone be for broadacre cropping?"},
    {"category": "buffer_zone", "question": "What buffer zone is required for intensive cropping?"},
    {"category": "buffer_zone", "question": "Do I need a buffer zone around my orchard?"},
    {"category": "buffer_zone", "question": "What buffer zone is needed when adjacent to a conventional farm?"},
    {"category": "buffer_zone", "question": "How do I manage buffer zones to prevent spray drift?"},
    {"category": "buffer_zone", "question": "What buffer zone is required next to waterways?"},
    {"category": "buffer_zone", "question": "How is buffer zone width determined and documented?"},
    # --- Record keeping (8) ---
    {"category": "record_keeping", "question": "What records do I need to keep for organic certification?"},
    {"category": "record_keeping", "question": "How long must I retain certification records?"},
    {"category": "record_keeping", "question": "What records are needed to prove an audit trail?"},
    {"category": "record_keeping", "question": "Do I need to record all inputs applied to my crops?"},
    {"category": "record_keeping", "question": "What sales records must an organic producer keep?"},
    {"category": "record_keeping", "question": "What production records are required for certification?"},
    {"category": "record_keeping", "question": "What should I prepare before an organic inspection?"},
    {"category": "record_keeping", "question": "Can certification records be kept digitally?"},
    # --- Transition (7) ---
    {"category": "transition", "question": "How do I transition from conventional to organic farming?"},
    {"category": "transition", "question": "How long is the transition period for land?"},
    {"category": "transition", "question": "What is the transition period for crops?"},
    {"category": "transition", "question": "How long does livestock need to be under organic management?"},
    {"category": "transition", "question": "Can I transition part of my farm to organic while the rest stays conventional?"},
    {"category": "transition", "question": "Do I need a transition plan for certification?"},
    {"category": "transition", "question": "Can I run organic and non-organic production in parallel during transition?"},
    # --- Compost (8) ---
    {"category": "compost", "question": "What are the requirements for compost production on an organic farm?"},
    {"category": "compost", "question": "Can I use compost made from animal manure in organic production?"},
    {"category": "compost", "question": "Is compost allowed as a soil amendment under organic standards?"},
    {"category": "compost", "question": "Is compost tea permitted in organic farming?"},
    {"category": "compost", "question": "Can I use vermicompost produced from worm castings?"},
    {"category": "compost", "question": "Is green waste compost acceptable for organic production?"},
    {"category": "compost", "question": "What temperature must compost reach during the composting process?"},
    {"category": "compost", "question": "Are there restrictions on composting livestock manures?"},
    # --- Labelling (8) ---
    {"category": "labelling", "question": "What are the labelling requirements for certified organic products?"},
    {"category": "labelling", "question": "When can I use the certified organic logo?"},
    {"category": "labelling", "question": "What percentage of a product must be organic to be labelled organic?"},
    {"category": "labelling", "question": "Can I label a product as made with organic ingredients?"},
    {"category": "labelling", "question": "How are imported organic products labelled in Australia?"},
    {"category": "labelling", "question": "Can I use my brand name on certified organic products?"},
    {"category": "labelling", "question": "Do I need to display my certification number on product labels?"},
    {"category": "labelling", "question": "What are the labelling rules for organic retail versus wholesale products?"},
    # --- Export / import (5) ---
    {"category": "export_import", "question": "What do I need to export certified organic produce overseas?"},
    {"category": "export_import", "question": "What are the requirements for importing organic products into Australia?"},
    {"category": "export_import", "question": "How does organic equivalence work between countries?"},
    {"category": "export_import", "question": "What is an organic export certificate and how do I get one?"},
    {"category": "export_import", "question": "Which markets accept Australian organic certification for exports?"},
    # --- Crop production (10) ---
    {"category": "crop_production", "question": "Is crop rotation required under organic standards?"},
    {"category": "crop_production", "question": "Can I use cover crops in organic farming?"},
    {"category": "crop_production", "question": "What are the irrigation requirements for organic crops?"},
    {"category": "crop_production", "question": "Where must I source organic seed from?"},
    {"category": "crop_production", "question": "When can non-organic seed be used in organic production?"},
    {"category": "crop_production", "question": "Is genetically modified seed prohibited in organic production?"},
    {"category": "crop_production", "question": "What are the rules for sourcing planting material?"},
    {"category": "crop_production", "question": "What are the requirements for post-harvest handling of organic crops?"},
    {"category": "crop_production", "question": "Are there specific requirements for greenhouse organic production?"},
    {"category": "crop_production", "question": "What practices maintain soil fertility in organic crop production?"},
    # --- Livestock (8) ---
    {"category": "livestock", "question": "What are the feed requirements for organic livestock?"},
    {"category": "livestock", "question": "What pasture requirements apply to organic livestock?"},
    {"category": "livestock", "question": "Which animal health treatments are permitted in organic livestock production?"},
    {"category": "livestock", "question": "Where must organic livestock be sourced from?"},
    {"category": "livestock", "question": "What are the transport requirements for organic livestock?"},
    {"category": "livestock", "question": "What are the slaughter requirements for certified organic livestock?"},
    {"category": "livestock", "question": "What are the requirements for organic wool production?"},
    {"category": "livestock", "question": "What are the standards for organic poultry production?"},
    # --- Pest & disease (6) ---
    {"category": "pest_disease", "question": "How must pests be managed on an organic farm?"},
    {"category": "pest_disease", "question": "What are the requirements for disease control in organic production?"},
    {"category": "pest_disease", "question": "Are biological controls permitted for pest management?"},
    {"category": "pest_disease", "question": "What cultural practices help control pests organically?"},
    {"category": "pest_disease", "question": "What physical controls are allowed for pest management?"},
    {"category": "pest_disease", "question": "Which pest treatments are allowed under organic standards?"},
    # --- Soil health (6) ---
    {"category": "soil_health", "question": "How do I maintain soil fertility in organic systems?"},
    {"category": "soil_health", "question": "What nutrient management practices are required organically?"},
    {"category": "soil_health", "question": "What are the soil conservation requirements for organic farms?"},
    {"category": "soil_health", "question": "How do I prevent soil erosion on an organic farm?"},
    {"category": "soil_health", "question": "Are green manures required in organic farming systems?"},
    {"category": "soil_health", "question": "What are the land preparation requirements before organic planting?"},
    # --- Processing & handling (6) ---
    {"category": "processing", "question": "What are the processing requirements for organic food products?"},
    {"category": "processing", "question": "Where must organic ingredients be sourced from for processing?"},
    {"category": "processing", "question": "How do I prevent contamination during organic processing?"},
    {"category": "processing", "question": "What cleaning requirements apply to organic processing facilities?"},
    {"category": "processing", "question": "How is pest control handled in organic processing facilities?"},
    {"category": "processing", "question": "What traceability is required for organic processed products?"},
    # --- Audit & inspection (5) ---
    {"category": "audit", "question": "What happens during an organic inspection?"},
    {"category": "audit", "question": "What do organic inspectors check on farm?"},
    {"category": "audit", "question": "How often are organic farms audited?"},
    {"category": "audit", "question": "What happens if I am found non-compliant during inspection?"},
    {"category": "audit", "question": "What is a corrective action in organic certification?"},
    # --- Casual / greeting probes (5) ---
    {"category": "casual", "question": "hello"},
    {"category": "casual", "question": "thanks for your help"},
    {"category": "casual", "question": "what can you do?"},
    {"category": "casual", "question": "what is the meaning of life?"},
    {"category": "casual", "question": "how are you going?"},
]


def run_question(client: httpx.Client, backend: str, q: Dict[str, str], country: str = "au") -> Dict:
    """Run one query against the backend /api/query endpoint."""
    start = time.time()
    try:
        resp = client.post(
            f"{backend}/api/query",
            json={"question": q["question"], "top_k": 5, "country": country},
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
    parser.add_argument("--country", default="au", help="Country code to query (default: au)")
    parser.add_argument("--output", default="stress_report.json", help="Output report file")
    args = parser.parse_args()

    print(f"Testing backend at {args.backend} (country={args.country})")
    print(f"Running {len(QUESTIONS)} synthetic questions...\n")

    results: List[Dict] = []
    with httpx.Client() as client:
        for i, q in enumerate(QUESTIONS, 1):
            r = run_question(client, args.backend, q, country=args.country)
            results.append(r)
            status = "✓" if r.get("ok") and r.get("sources_retrieved") else "⚠" if r.get("ok") else "✗"
            print(
                f"[{i:3}/{len(QUESTIONS)}] {status} [{r.get('category','?'):15}] "
                f"{r.get('question','?')[:45]:45} conf={r.get('confidence','-'):>6} "
                f"src={r.get('n_sources',0)} {r.get('latency_ms','-')}ms"
            )

    report_path = Path(args.output)
    report_path.write_text(json.dumps(results, indent=2))

    # Summary
    ok = sum(1 for r in results if r.get("ok"))
    grounded = sum(1 for r in results if r.get("sources_retrieved"))
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / len(results)
    print(f"\nSummary: {ok}/{len(results)} responded, {grounded} returned sources")
    print(f"Average latency: {round(avg_latency)}ms")
    print(f"Report written to {report_path}")

    # Per-category breakdown
    print("\nBy category:")
    cats = {}
    for r in results:
        cats.setdefault(r.get("category", "?"), {"n": 0, "grounded": 0, "avg_conf": 0})
        cats[r.get("category", "?")]["n"] += 1
        if r.get("sources_retrieved"):
            cats[r.get("category", "?")]["grounded"] += 1
        cats[r.get("category", "?")]["avg_conf"] += r.get("confidence", 0)
    for cat, stats in cats.items():
        avg_conf = stats["avg_conf"] / stats["n"]
        print(f"  {cat:15} {stats['grounded']:2}/{stats['n']:2} grounded  avg_conf={round(avg_conf,3)}")

    # Flag low-relevance categories (potential retrieval failures)
    low = [r for r in results if r.get("ok") and not r.get("sources_retrieved")]
    print(f"\nQueries without sources ({len(low)}):")
    for r in low:
        print(f"  - [{r.get('category')}] {r.get('question')} (conf={r.get('confidence')})")


if __name__ == "__main__":
    main()
