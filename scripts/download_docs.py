#!/usr/bin/env python3
"""
Download standards PDFs for installed countries, driven by their manifests.

For every manifest source with a `url` and `local_path`, downloads the file,
then verifies the declared sha256 (hard failure on mismatch).

Usage:
    python scripts/download_docs.py                  # all installed countries
    python scripts/download_docs.py --country au     # a single country
    python scripts/download_docs.py --force          # overwrite existing files
"""

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import DATA_DIR
from app.data.manifest import (
    installed_countries,
    load_manifest,
    sha256_file,
    verify_manifest,
)


def download_source(country: str, source) -> bool:
    path = DATA_DIR / country / source.local_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not DOWNLOAD_FORCE:
        print(f"  - [{source.id}] already present: {path.name}")
        return True

    print(f"  ↓ [{source.id}] downloading {source.url}")
    try:
        with httpx.stream("GET", source.url, timeout=60, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    except Exception as e:
        print(f"    ✗ download failed: {e}")
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
        return False

    if source.sha256:
        actual = sha256_file(path)
        if actual != source.sha256:
            print(f"    ✗ sha256 mismatch: expected {source.sha256[:12]}… got {actual[:12]}…")
            print("      The remote document has changed. Update the manifest (edition + hash)")
            print("      via scripts/update_hashes.py after confirming the new edition is correct.")
            return False
        print(f"    ✓ saved + hash verified: {path.name}")
    else:
        print(f"    ✓ saved (no hash declared — run scripts/update_hashes.py): {path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download standards PDFs per country manifest")
    parser.add_argument("--country", default=None, help="Download for a single country")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    global DOWNLOAD_FORCE
    DOWNLOAD_FORCE = args.force

    countries = [args.country.lower()] if args.country else installed_countries(DATA_DIR)
    if not countries:
        print("No countries installed. Run scripts/add_country.py first.")
        sys.exit(1)

    ok = True
    for country in countries:
        print(f"\n[{country}]")
        manifest = load_manifest(DATA_DIR, country)
        for source in manifest.sources:
            if not source.url or not source.local_path:
                continue
            if source.status == "superseded":
                print(f"  - [{source.id}] superseded — skipped")
                continue
            ok = download_source(country, source) and ok

    print("\n" + "=" * 60)
    print("Verifying manifests after download...")
    for country in countries:
        manifest = load_manifest(DATA_DIR, country)
        problems = verify_manifest(DATA_DIR, manifest)
        for p in problems:
            print(f"  ✗ {p}")
            ok = False
    print("✓ Download + verification complete" if ok else "✗ Some problems — see above")
    print("Next: python scripts/ingest.py --country <cc>")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
