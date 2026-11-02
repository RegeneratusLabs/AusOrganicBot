#!/usr/bin/env python3
"""
Document re-indexing script with HTML generation for OrgStandardsBot.

Re-indexes PDFs for a country with page tracking and HTML versions.

Usage:
    python scripts/reindex_with_html.py --country au
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.data.loader import DocumentLoader
from app.rag.retriever import VectorStore
from app.config import DATA_DIR


def reindex_country(country: str, generate_html: bool = True) -> bool:
    print("=" * 60)
    print(f"OrgStandardsBot - Document Re-Indexing ({country.upper()})")
    print("=" * 60)
    print()

    loader = DocumentLoader(DATA_DIR, country=country, generate_html=generate_html)
    vector_store = VectorStore(country=country)

    print(f"Clearing existing '{country}' vector store...")
    vector_store.clear()
    print()

    print(f"Loading documents from: {DATA_DIR / country}")
    print(f"HTML generation: {'Enabled' if generate_html else 'Disabled'}")
    print()

    chunks = loader.load_all_documents()

    if not chunks:
        print("No documents found!")
        print(f"Please add PDF files to: {DATA_DIR / country}/<source>/")
        return False

    print(f"Total chunks: {len(chunks)}")

    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [loader.generate_id(c["text"], c["metadata"]) for c in chunks]

    vector_store.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Re-indexing complete! {vector_store.get_document_count()} chunks indexed.")
    return True


def list_documents(country: str = "au"):
    """List all indexed documents"""
    vector_store = VectorStore(country=country)
    all_data = vector_store.collection.get(include=["metadatas"])

    sources = {}
    for metadata in all_data.get("metadatas", []):
        source = metadata.get("source", "unknown")
        if source not in sources:
            sources[source] = {"files": set(), "chunks": 0, "editions": set()}
        sources[source]["files"].add(metadata.get("filename", "unknown"))
        sources[source]["chunks"] += 1
        sources[source]["editions"].add(metadata.get("edition", "unknown"))

    print()
    print(f"Indexed Documents ({country.upper()}):")
    print("-" * 40)
    for source, info in sources.items():
        print(f"  {source}:")
        print(f"    Editions: {', '.join(sorted(info['editions']))}")
        print(f"    Files: {len(info['files'])}")
        print(f"    Chunks: {info['chunks']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index documents for OrgStandardsBot")
    parser.add_argument("--country", default="au", help="Country code (default: au)")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    parser.add_argument("--list", action="store_true", help="List indexed documents")
    args = parser.parse_args()

    if args.list:
        list_documents(args.country)
    else:
        success = reindex_country(args.country, generate_html=not args.no_html)
        sys.exit(0 if success else 1)
