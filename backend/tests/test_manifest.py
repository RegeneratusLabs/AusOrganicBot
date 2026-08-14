"""Tests for country manifest loading, validation, and hashing."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.manifest import (
    CountryManifest,
    ManifestSource,
    installed_countries,
    load_manifest,
    save_manifest,
    sha256_file,
    verify_manifest,
)


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


def write_manifest(data_dir: Path, manifest: CountryManifest):
    return save_manifest(data_dir, manifest)


def test_save_and_load_roundtrip(data_dir):
    m = CountryManifest(
        country="nz",
        name="New Zealand",
        sources=[ManifestSource(id="biogro", name="BioGro NZ Standard", edition="2024")],
    )
    write_manifest(data_dir, m)
    loaded = load_manifest(data_dir, "nz")
    assert loaded.name == "New Zealand"
    assert loaded.sources[0].id == "biogro"


def test_installed_countries(data_dir):
    write_manifest(data_dir, CountryManifest(country="au", name="Australia"))
    write_manifest(data_dir, CountryManifest(country="nz", name="New Zealand"))
    assert installed_countries(data_dir) == ["au", "nz"]


def test_installed_countries_empty_when_missing(data_dir):
    assert installed_countries(data_dir) == []


def test_load_missing_manifest_raises(data_dir):
    with pytest.raises(FileNotFoundError):
        load_manifest(data_dir, "zz")


def test_duplicate_source_editions_rejected():
    with pytest.raises(Exception, match="duplicate"):
        CountryManifest(
            country="au",
            name="Australia",
            sources=[
                ManifestSource(id="x", name="X", edition="2024"),
                ManifestSource(id="x", name="X", edition="2024"),
            ],
        )


def test_same_source_different_editions_allowed():
    m = CountryManifest(
        country="au",
        name="Australia",
        sources=[
            ManifestSource(id="x", name="X", edition="2024"),
            ManifestSource(id="x", name="X", edition="2023", status="superseded"),
        ],
    )
    assert len(m.active_sources) == 1
    assert m.active_sources[0].edition == "2024"


def test_verify_manifest_hash_mismatch(data_dir):
    m = CountryManifest(
        country="nz",
        name="New Zealand",
        sources=[
            ManifestSource(
                id="biogro",
                name="BioGro",
                edition="2024",
                local_path="biogro/std.pdf",
                sha256="0" * 64,
            )
        ],
    )
    (data_dir / "nz" / "biogro").mkdir(parents=True)
    pdf = data_dir / "nz" / "biogro" / "std.pdf"
    pdf.write_bytes(b"not the right content")

    problems = verify_manifest(data_dir, m)
    assert any("sha256 mismatch" in p for p in problems)


def test_verify_manifest_ok(data_dir):
    content = b"standard text"
    (data_dir / "nz" / "biogro").mkdir(parents=True)
    pdf = data_dir / "nz" / "biogro" / "std.pdf"
    pdf.write_bytes(content)

    m = CountryManifest(
        country="nz",
        name="New Zealand",
        sources=[
            ManifestSource(
                id="biogro",
                name="BioGro",
                edition="2024",
                local_path="biogro/std.pdf",
                sha256=sha256_file(pdf),
            )
        ],
    )
    assert verify_manifest(data_dir, m) == []


def test_verify_manifest_missing_file(data_dir):
    m = CountryManifest(
        country="nz",
        name="New Zealand",
        sources=[
            ManifestSource(id="biogro", name="BioGro", edition="2024", local_path="biogro/std.pdf")
        ],
    )
    problems = verify_manifest(data_dir, m)
    assert any("file missing" in p for p in problems)


def test_invalid_country_code_rejected():
    with pytest.raises(Exception):
        CountryManifest(country="australia", name="Australia")
