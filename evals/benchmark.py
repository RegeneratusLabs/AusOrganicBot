"""Human-adjudicated accuracy benchmark.

Why this exists
---------------
Every accuracy figure the project produced before this was circular: the
retrieval gold labels were written by the same local model the bot runs on
(`build_eval_set.py`), and the judge graded that same model's answers
(`llm_judge.py`). Both measured self-agreement.

This module defines the benchmark that replaces them. The rule that makes it
non-circular is structural, not procedural:

    **An item is ground truth only when a human has verified it.**

Machine-suggested items live alongside human-verified ones but are counted as
`draft`. Scoring refuses to report accuracy over drafts, so an LLM can never
launder its own suggestions into a headline number. A human promotes an item
with `evals/review_benchmark.py`.

Item fields
-----------
question          The question as a user would ask it.
category          One of CATEGORIES.
answerable        Can the installed standards actually answer this?
expected_clauses  Clause numbers a correct answer must cite. Empty when the
                  question is not answerable. This is what citation
                  *attribution* is scored against — the check the presence-only
                  verifier cannot do.
reference_answer  A human-written answer, for semantic grading.
provenance        Where the question came from (real enquiry, stress bank, ...).
verified_by       Name of the human who verified it; empty means draft.
verified_at       ISO date of verification.
gold_chunks       Optional retrieval expectations, for retrieval scoring.
notes             Free text from the reviewer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

BENCH_DIR = Path(__file__).resolve().parent / "bench"

# Categories are aligned with the existing stress bank so retrieval and answer
# metrics can be compared per topic.
CATEGORIES = (
    "input_approval",
    "buffer_zone",
    "record_keeping",
    "transition",
    "labelling",
    "livestock",
    "crop_production",
    "processing",
    "export_import",
    "audit",
    "soil_health",
    "compost",
    "pest_disease",
    "casual",
    "adversarial",
)

PROVENANCE_VALUES = (
    "real_enquiry",      # a question a farmer/certifier actually asked
    "expert_authored",   # written by someone who knows the standard
    "stress_bank",       # derived from scripts/stress_test.py
    "adversarial",       # designed to provoke fabrication
    "machine_draft",     # LLM-suggested, pending human verification
)


@dataclass
class BenchItem:
    question: str
    category: str
    answerable: bool
    expected_clauses: List[str] = field(default_factory=list)
    reference_answer: str = ""
    provenance: str = "machine_draft"
    verified_by: str = ""
    verified_at: str = ""
    gold_chunks: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_verified(self) -> bool:
        """Ground truth only when a named human has signed it off."""
        return bool(self.verified_by.strip()) and bool(self.verified_at.strip())

    def validate(self) -> List[str]:
        """Structural problems with this item (empty list = valid)."""
        problems: List[str] = []
        if not self.question.strip():
            problems.append("empty question")
        if self.category not in CATEGORIES:
            problems.append(f"unknown category {self.category!r}")
        if self.provenance not in PROVENANCE_VALUES:
            problems.append(f"unknown provenance {self.provenance!r}")
        if self.answerable and not self.expected_clauses:
            problems.append(
                "answerable item needs expected_clauses — without it, citation "
                "attribution cannot be scored"
            )
        if not self.answerable and self.expected_clauses:
            problems.append("unanswerable item must not declare expected_clauses")
        if self.is_verified and not self.reference_answer.strip():
            problems.append("a verified item needs a reference_answer")
        if self.is_verified and not self.verified_by.strip():
            problems.append("a verified item needs verified_by")
        return problems

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchItem":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def bench_path(country: str) -> Path:
    return BENCH_DIR / f"bench_{country}.jsonl"


def load_bench(country: str) -> List[BenchItem]:
    path = bench_path(country)
    if not path.exists():
        return []
    items: List[BenchItem] = []
    for line in path.read_text().splitlines():
        if line.strip():
            items.append(BenchItem.from_dict(json.loads(line)))
    return items


def save_bench(country: str, items: Iterable[BenchItem]) -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = bench_path(country)
    with open(path, "w") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return path


def verify_item(item: BenchItem, verified_by: str, reference_answer: str = None,
                expected_clauses: List[str] = None, notes: str = None) -> BenchItem:
    """Promote a draft item to ground truth, stamping the reviewer."""
    if not verified_by.strip():
        raise ValueError("verification requires a reviewer name")
    if expected_clauses is not None:
        item.expected_clauses = list(expected_clauses)
    if reference_answer is not None:
        item.reference_answer = reference_answer
    if notes is not None:
        item.notes = notes
    item.verified_by = verified_by.strip()
    item.verified_at = date.today().isoformat()
    problems = item.validate()
    if problems:
        raise ValueError("cannot verify item: " + "; ".join(problems))
    return item


def partition(items: Iterable[BenchItem]) -> Dict[str, List[BenchItem]]:
    """Split into verified (scoreable) and draft (not scoreable) sets."""
    verified: List[BenchItem] = []
    draft: List[BenchItem] = []
    for item in items:
        (verified if item.is_verified else draft).append(item)
    return {"verified": verified, "draft": draft}


def stats(items: Iterable[BenchItem]) -> Dict[str, object]:
    """Coverage summary — used to decide whether the benchmark is large enough."""
    items = list(items)
    parts = partition(items)
    by_category: Dict[str, int] = {}
    for item in parts["verified"]:
        by_category[item.category] = by_category.get(item.category, 0) + 1
    return {
        "total": len(items),
        "verified": len(parts["verified"]),
        "draft": len(parts["draft"]),
        "verified_answerable": sum(1 for i in parts["verified"] if i.answerable),
        "verified_unanswerable": sum(1 for i in parts["verified"] if not i.answerable),
        "by_category": dict(sorted(by_category.items())),
        "categories_uncovered": sorted(set(CATEGORIES) - set(by_category)),
    }
