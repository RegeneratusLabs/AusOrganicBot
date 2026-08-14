#!/usr/bin/env python3
"""
Validate country manifests: schema + file presence + sha256 integrity.

Usage:
    python scripts/validate_manifest.py --country au
    python scripts/validate_manifest.py            # all installed countries
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import DATA_DIR
from app.data.manifest import installed_countries, load_manifest, verify_manifest


def main():
    parser = argparse.ArgumentParser(description="Validate country manifests")
    parser.add_argument("--country", default=None, help="Validate a single country")
    args = parser.parse_args()

    countries = [args.country.lower()] if args.country else installed_countries(DATA_DIR)
    if not countries:
        print("No countries installed. Run: python scripts/add_country.py --country <cc> --name '<name>'")
        sys.exit(1)

    exit_code = 0
    for country in countries:
        print(f"[{country}]")
        try:
            manifest = load_manifest(DATA_DIR, country)
        except Exception as e:
            print(f"  ✗ {e}")
            exit_code = 1
            continue

        print(f"  name: {manifest.name} | language: {manifest.language} | "
              f"embedding: {manifest.embedding_family} | "
              f"sources: {len(manifest.sources)}")
        problems = verify_manifest(DATA_DIR, manifest)
        if problems:
            exit_code = 1
            for p in problems:
                print(f"  ✗ {p}")
        else:
            print("  ✓ schema, files, and hashes OK")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
