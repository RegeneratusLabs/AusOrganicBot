#!/usr/bin/env python3
"""
Document ingestion script for OrgStandardsBot.

Loads PDF documents from data/<country>/<source>/ into per-country vector stores.

Usage:
    python scripts/ingest.py                     # ingest every installed country
    python scripts/ingest.py --country au        # ingest a single country
    python scripts/ingest.py --country au --no-html
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.data.loader import DocumentLoader
from app.data.manifest import installed_countries, verify_manifest, load_manifest
from app.rag.retriever import VectorStore
from app.config import DATA_DIR


def ingest_country(
    country: str,
    generate_html: bool = True,
    only_source: str = None,
    only_file: str = None,
    append: bool = False,
) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Ingesting country: {country.upper()}")
    print(f"{'=' * 60}")

    try:
        manifest = load_manifest(DATA_DIR, country)
        problems = verify_manifest(DATA_DIR, manifest)
        if problems:
            print(f"✗ Manifest problems for {country}:")
            for p in problems:
                print(f"  {p}")
            print("  Aborting ingest for this country. Fix the manifest or re-download files.")
            return False
        print(f"  {manifest.name} — {len(manifest.active_sources)} active source(s)")
    except FileNotFoundError:
        print(f"  No manifest for '{country}' — loading PDFs from data/{country}/<source>/ directly")

    loader = DocumentLoader(DATA_DIR, country=country, generate_html=generate_html)
    vector_store = VectorStore(country=country)

    if append:
        # Continue a previous partial ingest instead of restarting it. Useful
        # for large documents where a single run would exceed the time budget.
        print(f"  Appending to existing '{country}' collection (not clearing)")
    else:
        print(f"  Clearing existing '{country}' vector store collection...")
        vector_store.clear()

    if only_source:
        print(f"  Restricting to source '{only_source}'")
    if only_file:
        print(f"  Restricting to file '{only_file}'")

    documents = loader.load_all_documents(only_source=only_source, only_file=only_file)
    if append:
        # Re-adding identical ids is idempotent in Chroma, so a resumed run
        # cannot double-count chunks.
        existing = vector_store.get_document_count()
        print(f"  Collection already holds {existing} chunk(s)")

    if not documents:
        print(f"✗ No documents found for {country}!")
        print(f"  Expected structure: data/{country}/<source>/*.pdf")
        return False

    print(f"✓ Loaded {len(documents)} chunks")

    all_chunks = [d["text"] for d in documents]
    all_metadatas = [DocumentLoader.sanitize_metadata(d["metadata"]) for d in documents]
    all_ids = [DocumentLoader.generate_id(d["text"], d["metadata"]) for d in documents]

    print(f"Adding {len(all_chunks)} chunks to '{country}' collection...")
    vector_store.add_documents(
        documents=all_chunks, metadatas=all_metadatas, ids=all_ids, progress=True
    )

    print(f"✓ {country.upper()} ingest complete — {vector_store.get_document_count()} chunks")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest standards PDFs into the vector store")
    parser.add_argument("--country", default=None, help="Ingest a single country code")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    parser.add_argument(
        "--source",
        default=None,
        help="Ingest only this source directory (e.g. national); implies --append",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Ingest only this PDF filename (implies --append)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing collection instead of clearing it first",
    )
    args = parser.parse_args()

    # A single-source run is part of a larger country ingest, so it must not
    # wipe the chunks already stored for the other sources.
    append = args.append or bool(args.source) or bool(args.file)

    if args.country:
        countries = [args.country.lower()]
    else:
        countries = installed_countries(DATA_DIR)
        if not countries:
            print("✗ No countries installed.")
            print("  Run: python scripts/add_country.py --country <cc> --name '<name>'")
            sys.exit(1)
        print(f"Installed countries: {', '.join(countries)}")

    results = {}
    for country in countries:
        results[country] = ingest_country(
            country,
            generate_html=not args.no_html,
            only_source=args.source,
            only_file=args.file,
            append=append,
        )

    print(f"\n{'=' * 60}")
    print("Ingestion summary:")
    for country, ok in results.items():
        print(f"  {country.upper():6} {'✓' if ok else '✗'}")
    print(f"{'=' * 60}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
