"""Tests for per-source/per-file ingest filtering.

Large countries could not be ingested in one run, so the loader must be able to
restrict work to a single source or a single PDF while appending to an existing
collection rather than clearing it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.loader import DocumentLoader  # noqa: E402


def _make_country(tmp_path):
    """A tiny two-source, three-PDF country tree."""
    for source, files in {"alpha": ["a1.pdf", "a2.pdf"], "beta": ["b1.pdf"]}.items():
        d = tmp_path / "zz" / source
        d.mkdir(parents=True)
        for name in files:
            (d / name).write_bytes(b"%PDF-1.4 fake")
    return tmp_path


def _loader(data_dir, captured):
    loader = DocumentLoader(data_dir, country="zz", generate_html=False)

    def fake_pages(pdf_path):
        captured.append(pdf_path.name)
        return [{"page_number": 1, "text": f"1.1 Body of {pdf_path.name} " + ("x" * 200), "char_count": 220}]

    loader.load_pdf_with_pages = fake_pages
    return loader


def test_all_sources_by_default(tmp_path):
    captured = []
    loader = _loader(_make_country(tmp_path), captured)
    loader.load_all_documents()
    assert captured == ["a1.pdf", "a2.pdf", "b1.pdf"]


def test_only_source_restricts_to_one_directory(tmp_path):
    captured = []
    loader = _loader(_make_country(tmp_path), captured)
    loader.load_all_documents(only_source="beta")
    assert captured == ["b1.pdf"]


def test_only_file_restricts_to_one_pdf(tmp_path):
    captured = []
    loader = _loader(_make_country(tmp_path), captured)
    loader.load_all_documents(only_source="alpha", only_file="a2.pdf")
    assert captured == ["a2.pdf"]


def test_metadata_records_structure(tmp_path):
    captured = []
    loader = _loader(_make_country(tmp_path), captured)
    docs = loader.load_all_documents(only_source="alpha", only_file="a1.pdf")
    assert docs
    meta = docs[0]["metadata"]
    assert meta["country_code"] == "zz"
    # Structure-aware fields are always present so retrieval can filter on them.
    assert "section_number" in meta
    assert "clause_numbers" in meta
