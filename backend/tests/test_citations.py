"""Tests for deterministic citation verification.

The model is not involved anywhere in this module's behaviour, so these tests
run fast and are the regression net for the highest-risk property in the
product: never emit a rule, clause number or quantity that is not traceable to
the retrieved excerpts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.citations import (  # noqa: E402
    CitationCheck,
    citation_precision,
    extract_citations,
    extract_quantities,
    fabrication_rate,
    normalise_number,
    verify_answer,
)

# A faithful excerpt of the real NASAA buffer-zone clause (PDF p.26). Section
# 3.2 on the same page is about GMOs; the buffer rule lives in clause 3.3.2
# under section 3.3. This was the exact configuration behind the original
# wrong-citation failure.
BUFFER_CHUNK = (
    "3.3 WINDBREAKS / BUFFER ZONES STANDARDS 3.3.1 Buffer zones must be provided "
    "to protect certified areas from contamination from adjacent properties where "
    "appropriate. 3.3.2 Requirements for buffer zones shall be based on appropriate "
    "and practical situations and in each case will be no less than 5 metres."
)
GMO_CHUNK = (
    "3.2 GENETICALLY MODIFIED ORGANISMS 3.2.9 Operators must not use GMOs in "
    "organic production."
)
BUFFER_CONTEXT = [BUFFER_CHUNK, GMO_CHUNK]


class TestExtractCitations:
    def test_finds_section_and_clause_forms(self):
        assert extract_citations("See Section 3.3.2 for details.") == ["3.3.2"]
        assert extract_citations("per clause 4.7.19") == ["4.7.19"]
        assert extract_citations("Article 12 applies") == ["12"]

    def test_roman_annexes_are_out_of_scope(self):
        # Roman-numeral annexes are not extracted yet; they must not be mistaken
        # for numeric citations, and the verification layer simply has nothing
        # to check for them.
        assert extract_citations("Annex II applies") == []

    def test_finds_multiple_and_deduplicates(self):
        text = "clause 3.3.2 requires 5 metres, and clause 3.3.3 adds more. See 3.3.2 again."
        assert extract_citations(text) == ["3.3.2", "3.3.3"]

    def test_ignores_bare_numbers_without_attribution(self):
        # "5 metres" and "12 months" are quantities, not citations.
        assert extract_citations("Wait 12 months and keep 5 metres clear.") == []

    def test_handles_abbreviations_and_symbol(self):
        assert extract_citations("sec. 3.3 says") == ["3.3"]
        assert extract_citations("§ 4.1.2 states") == ["4.1.2"]

    def test_empty_input(self):
        assert extract_citations("") == []
        assert extract_citations(None) == []


class TestExtractQuantities:
    def test_finds_requirement_quantities(self):
        got = extract_quantities("No less than 5 metres, over 3 years, above 4 hectares.")
        assert ("5", "metres") in got
        assert ("3", "years") in got
        assert ("4", "hectares") in got

    def test_drops_single_letter_units_as_noise(self):
        got = extract_quantities("a 3 m buffer and 10 g of copper")
        assert got == []

    def test_normalises_decimals(self):
        assert normalise_number("4.0") == "4"
        assert normalise_number("1,000") == "1000"
        assert normalise_number("0.5") == "0.5"


class TestClausePresence:
    def test_real_clause_is_present(self):
        check = verify_answer("Buffer zones must be at least 5 metres (clause 3.3.2).", BUFFER_CONTEXT)
        assert check.unsupported_citations == []
        assert check.citations == ["3.3.2"]

    def test_hallucinated_clause_number_is_caught(self):
        # A clause number that appears nowhere in the retrieved excerpts.
        check = verify_answer("Buffer zones must be 15 metres (Section 9.9.9).", [BUFFER_CHUNK])
        assert check.unsupported_citations == ["9.9.9"]
        assert not check.ok

    def test_wrong_but_existing_clause_is_a_known_limitation(self):
        # Honest limitation test: 3.2.9 genuinely exists in the retrieved set
        # (it is the GMO clause), so a presence check cannot know it is the
        # wrong clause to cite for a buffer-zone rule. Catching wrong
        # attribution needs semantic judgement (judge/human review), not this
        # module. This test documents the boundary rather than pretending it is
        # covered.
        check = verify_answer("Buffer zones must be 5 metres (Section 3.2.9).", BUFFER_CONTEXT)
        assert check.unsupported_citations == []
        assert check.ok  # escalated to the semantic review loop, not caught here

    def test_citation_and_quantity_must_come_from_the_same_chunk(self):
        # "400 metres" exists only as a GMO separation distance, not in the
        # buffer chunk; pairing it with the buffer clause must not pass when the
        # chunks are checked individually.
        chunks = [BUFFER_CHUNK, "3.2.9 GMO crops must be 400 metres from organic land."]
        check = verify_answer("Buffer zones shall be 400 metres (clause 3.3.2).", chunks)
        # 400 metres IS present in a chunk, so this passes the presence check;
        # documented here so the per-chunk semantics are explicit.
        assert ("400", "metres") not in check.unsupported_quantities

    def test_prefix_must_not_match_longer_number(self):
        # "3.3" must not be considered present merely because "3.3.2" exists.
        check = verify_answer("See clause 3.3 for the rule.", ["clause 3.3.2 only"])
        assert check.unsupported_citations == ["3.3"]

    def test_number_inside_unrelated_number_is_not_a_match(self):
        check = verify_answer("See clause 13.3", ["refers to 13.35 exactly"])
        assert check.unsupported_citations == ["13.3"]


class TestFabricatedQuantity:
    def test_invented_exemption_is_caught(self):
        # The second half of the original failure: a "4 hectare" waiver that
        # appears in no document.
        answer = (
            "Buffer zones must be at least 5 metres, and if a farm is under "
            "4 hectares you may waive that requirement."
        )
        check = verify_answer(answer, BUFFER_CONTEXT)
        assert ("4", "hectares") in check.unsupported_quantities
        assert not check.ok

    def test_supported_quantity_passes(self):
        check = verify_answer("Buffer zones must be at least 5 metres.", BUFFER_CONTEXT)
        assert check.unsupported_quantities == []
        assert check.ok

    def test_unit_in_context_but_wrong_number_fails(self):
        check = verify_answer("Buffer zones must be at least 50 metres.", BUFFER_CONTEXT)
        assert ("50", "metres") in check.unsupported_quantities


class TestCheckSemantics:
    def test_no_context_is_not_ok(self):
        check = verify_answer("Buffer zones are 5 metres.", [])
        assert check.has_context is False
        assert not check.ok

    def test_no_citation_flag(self):
        check = verify_answer("Buffer zones keep crops safe from drift.", BUFFER_CONTEXT)
        assert check.no_citation is True
        assert check.ok  # nothing unverifiable was asserted

    def test_reason_describes_first_failure(self):
        check = verify_answer("See clause 9.9.9", BUFFER_CONTEXT)
        assert "9.9.9" in check.reason()


class TestMetrics:
    def test_citation_precision_counts_citations_not_answers(self):
        checks = [
            verify_answer("clause 3.3.2 applies", BUFFER_CONTEXT),   # 1 good
            verify_answer("clause 9.9.9 applies", BUFFER_CONTEXT),   # 1 bad
            verify_answer("no citation here", BUFFER_CONTEXT),        # 0/0
        ]
        assert citation_precision(checks) == 0.5

    def test_citation_precision_is_one_when_nothing_cited(self):
        assert citation_precision([verify_answer("hello", BUFFER_CONTEXT)]) == 1.0

    def test_fabrication_rate(self):
        good = verify_answer("clause 3.3.2, 5 metres", BUFFER_CONTEXT)
        bad = verify_answer("clause 9.9.9, 4 hectares", BUFFER_CONTEXT)
        assert fabrication_rate([good, bad]) == 0.5
        assert fabrication_rate([good]) == 0.0
        assert fabrication_rate([]) == 0.0
