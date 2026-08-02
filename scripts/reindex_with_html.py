#!/usr/bin/env python3
"""
Document re-indexing script with HTML generation for AusOrganicBot

Re-indexes all PDFs with page tracking and generates HTML versions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.data.loader import DocumentLoader
from app.rag.retriever import VectorStore
from app.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def reindex_documents(generate_html: bool = True):
    """Re-index all documents with page tracking and HTML generation"""
    print("=" * 60)
    print("AusOrganicBot - Document Re-Indexing")
    print("=" * 60)
    print()

    loader = DocumentLoader(DATA_DIR, generate_html=generate_html)
    vector_store = VectorStore()

    print("Clearing existing vector store...")
    vector_store.clear()
    print()

    print(f"Loading documents from: {DATA_DIR}")
    print(f"HTML generation: {'Enabled' if generate_html else 'Disabled'}")
    print()

    chunks = loader.load_all_documents()

    if not chunks:
        print()
        print("No documents found!")
        print(f"Please add PDF files to subdirectories in: {DATA_DIR}")
        print()
        print("Expected structure:")
        print(f"  {DATA_DIR}/national/*.pdf")
        print(f"  {DATA_DIR}/aco/*.pdf")
        print(f"  {DATA_DIR}/nasaa/*.pdf")
        print(f"  {DATA_DIR}/sxo/*.pdf")
        print(f"  {DATA_DIR}/export/*.pdf")
        return False

    print()
    print(f"Total chunks: {len(chunks)}")
    print()

    print("Adding chunks to vector store...")

    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [loader.generate_id(c["text"], c["metadata"]) for c in chunks]

    vector_store.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    doc_count = vector_store.get_document_count()

    print()
    print("=" * 60)
    print("Re-indexing complete!")
    print(f"  Total chunks indexed: {doc_count}")
    print(f"  Vector store: {DATA_DIR}/vectorstore/")
    print("=" * 60)
    print()

    if generate_html:
        print("HTML documents generated in:")
        for source_dir in DATA_DIR.iterdir():
            if source_dir.is_dir() and source_dir.name != "vectorstore":
                html_dir = source_dir / "html"
                if html_dir.exists():
                    print(f"  {source_dir.name}/html/full.html")

    return True


def list_documents():
    """List all indexed documents"""
    vector_store = VectorStore()
    collection = vector_store.collection

    all_data = collection.get(include=["metadatas"])

    sources = {}
    for metadata in all_data.get("metadatas", []):
        source = metadata.get("source", "unknown")
        filename = metadata.get("filename", "unknown")
        if source not in sources:
            sources[source] = {"files": set(), "chunks": 0, "pages": 0}
        sources[source]["files"].add(filename)
        sources[source]["chunks"] += 1
        sources[source]["pages"] = metadata.get("total_pages", 0)

    print()
    print("Indexed Documents:")
    print("-" * 40)
    for source, info in sources.items():
        print(f"  {source}:")
        print(f"    Files: {len(info['files'])}")
        print(f"    Chunks: {info['chunks']}")
        print(f"    Pages: {info['pages']}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-index documents for AusOrganicBot")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    parser.add_argument("--list", action="store_true", help="List indexed documents")

    args = parser.parse_args()

    if args.list:
        list_documents()
    else:
        success = reindex_documents(generate_html=not args.no_html)
        sys.exit(0 if success else 1)
