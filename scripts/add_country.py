#!/usr/bin/env python3
"""
Scaffold a new country: data/<cc>/manifest.json template.

Usage:
    python scripts/add_country.py --country nz --name "New Zealand"
    python scripts/add_country.py --country eu --name "European Union" --language en
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import DATA_DIR
from app.data.manifest import CountryManifest, ManifestSource, save_manifest


def main():
    parser = argparse.ArgumentParser(description="Scaffold a country manifest")
    parser.add_argument("--country", required=True, help="2-letter country code (e.g. nz)")
    parser.add_argument("--name", required=True, help="Country name (e.g. 'New Zealand')")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--embedding-family",
        default="all-minilm",
        help="all-minilm | minicpm (default: all-minilm)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Optional per-country relevance threshold"
    )
    args = parser.parse_args()

    country = args.country.lower()
    if len(country) != 2 or not country.isalpha():
        print(f"✗ Invalid country code '{country}' — use 2 letters (e.g. nz)")
        sys.exit(1)

    path = DATA_DIR / country / "manifest.json"
    if path.exists():
        print(f"✗ {path} already exists — nothing to do")
        sys.exit(1)

    manifest = CountryManifest(
        country=country,
        name=args.name,
        language=args.language,
        embedding_family=args.embedding_family,
        relevance_threshold=args.threshold,
        sources=[],  # placeholder comment below shows how to fill it
    )
    save_manifest(DATA_DIR, manifest)

    # Rewrite with a helpful placeholder entry so users see the source schema
    template = {
        **manifest.model_dump(exclude_none=True),
        "sources": [
            {
                "id": "example-body",
                "name": "Example Standards Body — replace with the real one",
                "edition": "2026",
                "url": "https://example.org/standard.pdf",
                "local_path": "example-body/standard.pdf",
                "sha256": "run scripts/update_hashes.py after placing the file",
                "status": "active",
                "effective_date": "2026-01-01",
            }
        ],
    }
    path.write_text(json.dumps(template, indent=2) + "\n")

    print("✓ Created", path)
    print("  Next steps:")
    print(f"  1. Edit {path} — list each standards body + edition under 'sources'")
    print(f"  2. Place PDFs under data/{country}/<source-id>/ (or set 'url' and run download_docs.py)")
    print(f"  3. Set real sha256 per file (python scripts/update_hashes.py --country {country})")
    print(f"  4. Ingest: python scripts/ingest.py --country {country}")
    print("  5. The country appears in the UI selector automatically")


if __name__ == "__main__":
    main()
