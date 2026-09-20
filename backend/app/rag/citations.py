"""Deterministic citation verification.

The reader model is asked to cite clause numbers from the retrieved context.
A 1B-class model cannot be trusted to do that reliably on its own: in testing
it cited "Section 3.2.9" for a buffer-zone rule that actually lives in clause
3.3.2, and invented a "farms under 4 hectares" waiver that appears in no
document. Both answers were fluent and confident.

This module makes citation correctness a property of the *system* rather than
of the model. It verifies, without any model involvement, that:

  1. every clause/section number the answer cites actually occurs in the
     retrieved context that was shown to the model; and
  2. every quantity-with-unit the answer asserts (e.g. "4 hectares",
     "15 metres") is traceable to that same context.

Anything that fails can then be turned into an honest refusal instead of a
fabrication. This check is deliberately conservative: it can only *reject*
claims, never invent them, so a false positive costs one refusal while a false
negative costs a wrong answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

# A citation is a number the answer explicitly attributes to the document.
# Requiring an attributing word ("section", "clause", ...) keeps ordinary
# numbers ("5 metres", "12 months") out of the citation set.
_CITE_PREFIXES = (
    r"section|sections|clause|clauses|annex|annexes|article|articles|"
    r"part|parts|schedule|schedules|standard|para|paragraph|§|sec\.?"
)
_CITATION_RE = re.compile(
    rf"(?:{_CITE_PREFIXES})\s*[:.]?\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Numbers written with a unit, used for "traceable quantity" checks. We only
# treat these as *checkable claims* when they are requirements (metres, hectares,
# years, months, kg, %, days) rather than incidental counts.
_UNIT_WORDS = (
    "metre|metres|meter|meters|m|hectare|hectares|ha|year|years|month|months|"
    "day|days|kg|kilogram|kilograms|gram|grams|g|litre|litres|liter|liters|l|"
    "percent|%|degree|degrees|celsius"
)
_QUANTITY_RE = re.compile(
    rf"(\d+(?:[.,]\d+)?)\s*({_UNIT_WORDS})\b",
    re.IGNORECASE,
)

# Units that are too generic to be worth verifying on their own (e.g. "3 years"
# is a meaningful claim, but a bare "1" near "m" is usually noise).
_STOP_UNITS = {"m", "l", "g", "s"}


def normalise_number(value: str) -> str:
    """Canonicalise a numeric literal for comparison ('4.0' -> '4')."""
    value = value.replace(",", "").strip()
    try:
        as_float = float(value)
    except ValueError:
        return value
    if as_float.is_integer():
        return str(int(as_float))
    return str(as_float)


def _number_variants(number: str) -> set:
    """Forms of a number that should count as 'present' in text.

    Handles the two ways standards documents write quantities: digits and
    spelled-out small numbers. Only digits are matched here; spelled-out
    handling is intentionally omitted because false negatives produce a safe
    refusal, whereas guessing wrong would produce a fabricated pass.
    """
    normalised = normalise_number(number)
    variants = {normalised}
    if "." in normalised:
        # "4.0" / "4.5" style — also accept the raw literal.
        variants.add(normalised.rstrip("0").rstrip("."))
    return {v for v in variants if v}


def extract_citations(text: str) -> List[str]:
    """Return clause/section numbers attributed by the answer, in order.

    >>> extract_citations("Buffer zones are covered in Section 3.3.2.")
    ['3.3.2']
    >>> extract_citations("At least 5 metres, per clause 3.3.2 and 3.3.3.")
    ['3.3.2', '3.3.3']
    """
    out: List[str] = []
    for match in _CITATION_RE.finditer(text or ""):
        number = match.group(1)
        if number not in out:
            out.append(number)
    return out


def extract_quantities(text: str) -> List[tuple]:
    """Return (number, unit) requirement-like quantities asserted by the answer.

    Bare single-letter units are dropped as too noisy to verify.
    """
    out: List[tuple] = []
    for match in _QUANTITY_RE.finditer(text or ""):
        number, unit = match.group(1), match.group(2).lower()
        if unit in _STOP_UNITS:
            continue
        pair = (normalise_number(number), unit)
        if pair not in out:
            out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _clause_is_present(number: str, context: str) -> bool:
    """True when `number` appears in `context` as a standalone clause marker.

    A plain substring test is not enough: citing "3.3" would match inside
    "3.3.2" and inside unrelated text like "13.35". We require token
    boundaries and reject the case where the number is merely the prefix of a
    longer dotted number.
    """
    escaped = re.escape(number)
    pattern = re.compile(rf"(?<![\d.]){escaped}(?![\d.])")
    return bool(pattern.search(context))


def _quantity_is_present(number: str, unit: str, context: str) -> bool:
    """True when `number`+`unit` (in any common form) occurs in `context`.

    The number and unit need not be adjacent: standards frequently write
    "no less than 5 metres", but also "the minimum of 5 shall apply ... metres".
    We therefore require the number to appear somewhere in the context and the
    unit to appear somewhere, which is weaker than adjacency but still catches
    wholly invented quantities like the fabricated "4 hectare" waiver.
    """
    lowered = context.lower()
    if not any(v in lowered for v in _number_variants(number)):
        return False
    return unit in lowered


@dataclass
class CitationCheck:
    """Outcome of verifying one answer against its retrieved context."""

    citations: List[str] = field(default_factory=list)
    unsupported_citations: List[str] = field(default_factory=list)
    quantities: List[tuple] = field(default_factory=list)
    unsupported_quantities: List[tuple] = field(default_factory=list)
    has_context: bool = False

    @property
    def ok(self) -> bool:
        """True when every checkable claim is traceable to the context.

        An answer with no citations at all is not a verification failure here —
        `no_citation` reports that separately so it can be tracked as a metric
        rather than silently treated as grounded.
        """
        if not self.has_context:
            return False
        return not self.unsupported_citations and not self.unsupported_quantities

    @property
    def no_citation(self) -> bool:
        return self.has_context and not self.citations

    def reason(self) -> str:
        """Short human-readable explanation of the first failure, if any."""
        if not self.has_context:
            return "no retrieved context to ground the answer"
        if self.unsupported_citations:
            return "cited " + ", ".join(self.unsupported_citations) + " which is not in the retrieved context"
        if self.unsupported_quantities:
            pairs = ", ".join(f"{n} {u}" for n, u in self.unsupported_quantities)
            return f"asserted {pairs} which is not in the retrieved context"
        return "ok"


def verify_answer(answer: str, chunks: Sequence[str]) -> CitationCheck:
    """Verify an answer against the retrieved chunk texts.

    `chunks` is the list of chunk contents that were placed in the prompt.
    Each citation and quantity is checked against the individual chunks, and
    kept only if at least one *single* chunk supports it. Checking against the
    concatenation instead would let a clause number from one excerpt be paired
    with a quantity from a different one, which is not grounding.

    Scope and honest limits: this verifies that a cited clause number or
    quantity is *present* in the excerpts the model was shown. It cannot detect
    a citation that exists but is attributed to the wrong rule — deciding that
    requires semantic judgement, and is handled by the judge/human review loop
    rather than claimed here.
    """
    if isinstance(chunks, str):
        # Guard: a bare string would otherwise be iterated character by
        # character, silently making every check meaningless.
        raise TypeError(
            "verify_answer() expects a sequence of chunk strings, not a single string"
        )
    usable = [c for c in (chunks or []) if c and c.strip()]
    check = CitationCheck(has_context=bool(usable))
    check.citations = extract_citations(answer)
    check.quantities = extract_quantities(answer)

    if not check.has_context:
        # Nothing to verify against; anything asserted is unsupported.
        check.unsupported_citations = list(check.citations)
        check.unsupported_quantities = list(check.quantities)
        return check

    for number in check.citations:
        if not any(_clause_is_present(number, chunk) for chunk in usable):
            check.unsupported_citations.append(number)

    for number, unit in check.quantities:
        if not any(_quantity_is_present(number, unit, chunk) for chunk in usable):
            check.unsupported_quantities.append((number, unit))

    return check


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def citation_precision(checks: Iterable[CitationCheck]) -> float:
    """Fraction of all cited clauses that were present in context.

    This is the headline citation metric: 1.0 means the model never cited a
    clause it was not shown. It is computed over citations, not answers, so
    answers that cite nothing do not inflate it.
    """
    total = 0
    good = 0
    for check in checks:
        total += len(check.citations)
        good += len(check.citations) - len(check.unsupported_citations)
    return good / total if total else 1.0


def fabrication_rate(checks: Iterable[CitationCheck]) -> float:
    """Fraction of answers containing at least one untraceable claim.

    This is the metric the release gate is set on: it must be 0.
    """
    checks = list(checks)
    if not checks:
        return 0.0
    bad = sum(1 for c in checks if c.unsupported_citations or c.unsupported_quantities)
    return bad / len(checks)
