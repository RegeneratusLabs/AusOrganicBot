"""Country manifests: describe a country's standards sources and editions.

A country is a directory `data/<cc>/` containing:
- manifest.json  — this schema
- <source>/      — one dir per standards body, holding its PDF(s) + generated html/
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("ausorganicbot")

MANIFEST_FILENAME = "manifest.json"
VECTORSTORE_DIRNAME = "vectorstore"


class ManifestSource(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]+$", description="Slug of the standards body")
    name: str = Field(description="Human-readable name, e.g. 'BioGro NZ Standard'")
    edition: str = Field(description="Document edition, e.g. '2024' or 'Edition 3.8'")
    url: Optional[str] = None
    local_path: Optional[str] = None
    sha256: Optional[str] = None
    status: str = Field(default="active", pattern=r"^(active|superseded)$")
    effective_date: Optional[str] = None


class CountryManifest(BaseModel):
    country: str = Field(pattern=r"^[a-z]{2}$")
    name: str
    language: str = "en"
    embedding_family: str = "all-minilm"
    relevance_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Overrides the global relevance gate"
    )
    sources: List[ManifestSource] = []

    @field_validator("sources")
    @classmethod
    def unique_source_editions(cls, v: List[ManifestSource]) -> List[ManifestSource]:
        keys = [(s.id, s.edition) for s in v]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (id, edition) pairs in manifest")
        return v

    @property
    def active_sources(self) -> List[ManifestSource]:
        return [s for s in self.sources if s.status == "active"]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def manifest_path(data_dir: Path, country: str) -> Path:
    return data_dir / country / MANIFEST_FILENAME


def installed_countries(data_dir: Path) -> List[str]:
    if not data_dir.exists():
        return []
    return sorted(
        p.parent.name
        for p in data_dir.glob(f"*/{MANIFEST_FILENAME}")
        if p.parent.name != VECTORSTORE_DIRNAME
    )


def load_manifest(data_dir: Path, country: str) -> CountryManifest:
    path = manifest_path(data_dir, country)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for country '{country}' at {path}. "
            f"Run: python scripts/add_country.py --country {country}"
        )
    return CountryManifest.model_validate(json.loads(path.read_text()))


def save_manifest(data_dir: Path, manifest: CountryManifest) -> Path:
    path = manifest_path(data_dir, manifest.country)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(exclude_none=True), indent=2) + "\n"
    )
    return path


def verify_manifest(data_dir: Path, manifest: CountryManifest) -> List[str]:
    """Check local files exist and match their declared hashes.

    Returns a list of human-readable problems (empty = valid)."""
    problems = []
    for source in manifest.sources:
        if not source.local_path:
            continue
        path = data_dir / manifest.country / source.local_path
        if not path.exists():
            problems.append(
                f"[{manifest.country}/{source.id}] file missing: {path}"
            )
            continue
        if source.sha256:
            actual = sha256_file(path)
            if actual != source.sha256:
                problems.append(
                    f"[{manifest.country}/{source.id}] sha256 mismatch for {path}: "
                    f"expected {source.sha256[:12]}… got {actual[:12]}…"
                )
    return problems


def resolve_local_path(data_dir: Path, country: str, source: ManifestSource) -> Path:
    """Absolute path for a source's local_path (relative to the country dir)."""
    if not source.local_path:
        raise ValueError(f"source '{source.id}' has no local_path")
    path = data_dir / country / source.local_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it (scripts/download_docs.py) or "
            f"place it manually, then re-run ingest."
        )
    return path
