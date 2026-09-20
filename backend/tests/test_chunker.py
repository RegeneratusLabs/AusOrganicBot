"""Tests for structure-aware chunking.

The key property under test is the one that caused a verified-false citation in
production: a chunk containing a clause must also carry the section number that
clause belongs to, so the reader has a real number to cite instead of guessing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.chunker import chunk_structured  # noqa: E402

# Modelled directly on NASAA p.26, where section 3.2 (GMOs) is immediately
# followed by section 3.3 (buffer zones). A fixed-width chunker starting mid-page
# picked up 3.2 and never saw "3.3", which is how the model came to cite 3.2.9
# for a buffer-zone rule.
NASAA_PAGE = """
3.2 GENETICALLY MODIFIED ORGANISMS

GENERAL PRINCIPLES
Organic production must not use GMOs.

STANDARDS
3.2.7 Operators must conduct an assessment of risks from contamination with GMOs.
3.2.8 Planting or sowing for organic production will not take place until 5 years after the last use of prohibited inputs.
3.2.9 Operators must not use GMOs in organic production.

3.3 WINDBREAKS / BUFFER ZONES

GENERAL PRINCIPLES
Buffer zones provide protection from contamination.

STANDARDS
3.3.1 Buffer zones must be provided to protect certified areas from contamination from adjacent properties where appropriate.
3.3.2 Requirements for buffer zones shall be based on appropriate and practical situations and in each case will be no less than 5 metres.
3.3.3 Where outside rows of a crop are used as a buffer zone, produce from these rows shall be quarantined and may not be sold as certified.
"""


class TestStructureDetection:
    def test_sections_are_detected_with_titles(self):
        chunks = chunk_structured(NASAA_PAGE, max_chars=400)
        numbers = {c.section_number for c in chunks}
        assert "3.2" in numbers
        assert "3.3" in numbers

    def test_section_titles_are_captured(self):
        chunks = chunk_structured(NASAA_PAGE, max_chars=400)
        titles = {c.section_number: c.section_title for c in chunks if c.section_number}
        assert "GENETICALLY MODIFIED ORGANISMS" in titles["3.2"]
        assert "WINDBREAKS / BUFFER ZONES" in titles["3.3"]

    def test_clause_numbers_are_recorded(self):
        # At the production budget clauses may merge into one chunk, but every
        # clause number must still be recorded on the chunk that holds it.
        chunks = chunk_structured(NASAA_PAGE, max_chars=900)
        all_clauses = {c for ch in chunks for c in ch.clause_numbers}
        for expected in ("3.2.7", "3.2.8", "3.2.9", "3.3.1", "3.3.2", "3.3.3"):
            assert expected in all_clauses


class TestSectionContextIsPreserved:
    def test_every_buffer_clause_chunk_carries_section_3_3(self):
        """The exact fix: a chunk holding 3.3.2 must also contain '3.3'."""
        chunks = chunk_structured(NASAA_PAGE, max_chars=400)
        buffer_chunks = [c for c in chunks if "3.3.2" in c.text]
        assert buffer_chunks, "clause 3.3.2 should appear in a chunk"
        for chunk in buffer_chunks:
            assert "3.3" in chunk.embed_text
            assert chunk.section_number == "3.3"

    def test_buffer_chunks_never_claim_to_be_section_3_2(self):
        chunks = chunk_structured(NASAA_PAGE, max_chars=400)
        for chunk in chunks:
            if "3.3.2" in chunk.text or "buffer zone" in chunk.text.lower():
                assert chunk.section_number != "3.2"

    def test_embed_text_prefixes_the_section_header(self):
        chunks = chunk_structured(NASAA_PAGE, max_chars=400)
        prefixed = [c for c in chunks if c.section_number]
        assert prefixed
        for chunk in prefixed:
            assert chunk.embed_text.startswith(f"[{chunk.section_number}")


class TestClausesAreNotSplit:
    def test_clause_text_is_never_broken(self):
        chunks = chunk_structured(NASAA_PAGE, max_chars=400)
        joined = "\n".join(c.text for c in chunks)
        # The full clause sentence must survive intact inside one chunk.
        clause = "Requirements for buffer zones shall be based on appropriate and practical situations"
        assert clause in joined
        holders = [c for c in chunks if clause in c.text]
        assert len(holders) == 1

    def test_oversized_clause_is_split_and_keeps_its_number(self):
        """A clause longer than max_chars is split at sentence boundaries, and
        every resulting piece keeps the clause number so a retrieval hit still
        carries something citable."""
        sentences = " ".join(f"This is sentence number {i} about the standard." for i in range(60))
        long_clause = f"4.1.1 {sentences}"
        chunks = chunk_structured(f"4.1 SECTION\n\n{long_clause}", max_chars=300)
        holders = [c for c in chunks if "4.1.1" in c.clause_numbers]
        assert len(holders) > 1, "oversized clause should be split into several chunks"
        for chunk in holders:
            assert "4.1.1" in chunk.embed_text
        # No piece should be wildly over budget.
        assert all(len(c.text) < 300 * 2 for c in holders)

    def test_ordinary_clause_is_not_split(self):
        """Splitting only happens when a clause genuinely exceeds the budget."""
        chunks = chunk_structured(NASAA_PAGE, max_chars=2000)
        clause = "Requirements for buffer zones shall be based on appropriate and practical situations"
        holders = [c for c in chunks if clause in c.text]
        assert len(holders) == 1


class TestFallback:
    def test_unstructured_text_still_chunks(self):
        text = "\n\n".join(f"Paragraph {i} " + ("content " * 80) for i in range(5))
        chunks = chunk_structured(text, max_chars=500)
        assert chunks
        # No structure invented when none can be detected.
        assert all(c.section_number is None for c in chunks)

    def test_empty_input(self):
        assert chunk_structured("") == []
        assert chunk_structured("   \n  ") == []

    def test_tiny_fragments_dropped(self):
        chunks = chunk_structured("3.1 A SECTION\n\n3.1.1 ok.", max_chars=500, min_chars=200)
        assert chunks == []


class TestRunningHeaderStripping:
    """Regulation PDFs repeat a header on every page; it must not become chunk text."""

    def _loader(self):
        from app.data.loader import DocumentLoader
        from pathlib import Path
        return DocumentLoader(Path("/tmp"), country="uk", generate_html=False)

    def test_repeated_header_is_removed(self):
        pages = [
            {"page_number": i, "text": f"14/08/2026, 18:05 Commission Regulation (EC) No 889/2008\nArticle {i}\nBody text for page {i}.", "char_count": 0}
            for i in range(1, 9)
        ]
        removed = self._loader().strip_running_headers(pages)
        assert removed == 8
        for page in pages:
            assert "Commission Regulation" not in page["text"]
            assert f"Body text for page {page['page_number']}." in page["text"]

    def test_unique_lines_are_kept(self):
        pages = [
            {"page_number": i, "text": f"Distinct opening line {i}\nBody {i}.", "char_count": 0}
            for i in range(1, 9)
        ]
        removed = self._loader().strip_running_headers(pages)
        assert removed == 0
        assert "Distinct opening line 3" in pages[2]["text"]

    def test_short_documents_are_not_stripped(self):
        pages = [
            {"page_number": i, "text": "Same header\nBody.", "char_count": 0} for i in range(2)
        ]
        assert self._loader().strip_running_headers(pages) == 0
