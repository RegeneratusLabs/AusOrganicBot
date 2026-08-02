#!/usr/bin/env python3
"""
Document ingestion script for AusOrganicBot

Loads PDF documents from data/ directory into the vector store.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.data.loader import DocumentLoader
from app.rag.retriever import VectorStore
from app.config import DATA_DIR


def ingest_documents():
    """Ingest all documents into the vector store"""
    print("=" * 60)
    print("AusOrganicBot - Document Ingestion")
    print("=" * 60)

    # Initialize components
    loader = DocumentLoader(DATA_DIR)
    vector_store = VectorStore()

    # Clear existing documents
    print("\nClearing existing vector store...")
    vector_store.clear()

    # Load all documents (with page tracking)
    print(f"\nLoading documents from: {DATA_DIR}")
    documents = loader.load_all_documents()

    if not documents:
        print("\n❌ No documents found!")
        print(f"Please add PDF files to: {DATA_DIR}")
        print("\nExpected structure:")
        print(f"  {DATA_DIR}/nasaa/*.pdf")
        print(f"  {DATA_DIR}/aco/*.pdf")
        print(f"  {DATA_DIR}/sxo/*.pdf")
        print(f"  {DATA_DIR}/national/*.pdf")
        return

    print(f"\n✓ Loaded {len(documents)} chunks from documents")

    # Process chunks (already chunked by load_all_documents with page tracking)
    print("\nProcessing chunks...")
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc in documents:
        all_chunks.append(doc["text"])
        all_metadatas.append(doc["metadata"])
        chunk_id = DocumentLoader.generate_id(doc["text"], doc["metadata"])
        all_ids.append(chunk_id)

    # Add to vector store
    print(f"\nAdding {len(all_chunks)} chunks to vector store...")
    vector_store.add_documents(
        documents=all_chunks, metadatas=all_metadatas, ids=all_ids
    )

    # Report
    doc_count = vector_store.get_document_count()
    print(f"\n{'=' * 60}")
    print(f"✓ Ingestion complete!")
    print(f"  Total chunks: {doc_count}")
    print(f"  Vector store: {vector_store.client.get_settings().persist_directory}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    ingest_documents()
