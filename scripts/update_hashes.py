#!/usr/bin/env python3
"""
Compute/refresh sha256 hashes for a country's manifest sources.

Usage:
    python scripts/update_hashes.py --country au
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import DATA_DIR
from app.data.manifest import load_manifest, save_manifest, sha256_file


def main():
    parser = argparse.ArgumentParser(description="Refresh sha256 hashes in a country manifest")
    parser.add_argument("--country", required=True, help="2-letter country code")
    args = parser.parse_args()

    manifest = load_manifest(DATA_DIR, args.country.lower())

    for source in manifest.sources:
        if not source.local_path:
            print(f"  - [{source.id}] no local_path, skipped")
            continue
        path = DATA_DIR / manifest.country / source.local_path
        if not path.exists():
            print(f"  ✗ [{source.id}] file missing: {path}")
            continue
        source.sha256 = sha256_file(path)
        print(f"  ✓ [{source.id}] {source.sha256[:16]}… {path.name}")

    save_manifest(DATA_DIR, manifest)
    print(f"Manifest updated: {DATA_DIR / manifest.country / 'manifest.json'}")


if __name__ == "__main__":
    main()
