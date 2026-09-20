import logging
import chromadb
from typing import List, Dict, Any
from app.config import (
    CHROMA_PERSIST_DIR,
    TOP_K_RETRIEVAL,
    HYBRID_RETRIEVAL,
    HYBRID_POOL,
)
from app.rag.embeddings import EmbeddingService

logger = logging.getLogger("orgstandardsbot")

COLLECTION_PREFIX = "organic_standards"
EMBEDDING_DIM_METADATA = "embedding_dim"


class VectorStore:
    """ChromaDB vector store — one collection per country."""

    def __init__(self, country: str = "au", collection_name: str = None):
        self.country = country.lower()
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

        # Cached lexical index for hybrid search; built lazily on first use.
        self._hybrid = None

        self.embedding_service = EmbeddingService()

        if collection_name is None:
            collection_name = f"{COLLECTION_PREFIX}_{self.country}"

        existing = None
        try:
            existing = self.client.get_collection(name=collection_name)
        except Exception:
            pass

        # Embedding dimension guard: a collection built with a different
        # embedding model (or vector dim) must not receive mismatched vectors.
        if existing is not None:
            stored_dim = existing.metadata.get(EMBEDDING_DIM_METADATA)
            if stored_dim is not None and int(stored_dim) != self.embedding_service.dimension:
                logger.warning(
                    "Collection '%s' was built with embedding dim %s but the active "
                    "embedding model produces dim %s — rebuilding the collection. "
                    "Re-run scripts/ingest.py to repopulate it.",
                    collection_name, stored_dim, self.embedding_service.dimension,
                )
                self.client.delete_collection(name=collection_name)
                existing = None

        self.collection = existing or self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                EMBEDDING_DIM_METADATA: self.embedding_service.dimension,
            },
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 128,
        progress: bool = False,
    ):
        """Add documents to the vector store.

        Embedding and writing are done in batches so that a long ingest shows
        progress and, if interrupted, a partially-populated collection reflects
        a clear high-water mark rather than an opaque stall.
        """
        total = len(documents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            embeddings = self.embedding_service.embed_batch(documents[start:end])
            self.collection.add(
                embeddings=embeddings,
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
            if progress:
                print(f"    embedded+stored {end}/{total}", flush=True)

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Dense-only by default for the raw store; `search_hybrid` adds lexical
        fusion. The pipeline uses `search_hybrid` when HYBRID_RETRIEVAL is on.
        """
        query_embedding = self.embedding_service.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append(
                    {
                        "content": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "score": 1 - results["distances"][0][i]
                        if results["distances"]
                        else 0,  # Convert distance to similarity
                    }
                )

        return documents

    def search_hybrid(
        self, query: str, top_k: int = TOP_K_RETRIEVAL, pool: int = HYBRID_POOL
    ) -> List[Dict[str, Any]]:
        """Dense search fused with BM25 lexical search (see app/rag/hybrid.py).

        Falls back to dense-only when hybrid retrieval is disabled, so callers
        can use this unconditionally.
        """
        if not HYBRID_RETRIEVAL:
            return self.search(query, top_k=top_k)

        if getattr(self, "_hybrid", None) is None:
            from app.rag.hybrid import HybridRetriever

            self._hybrid = HybridRetriever(self, enable_lexical=True)
        return self._hybrid.search(query, top_k=top_k, pool=pool)

    def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        return self.collection.count()

    def clear(self):
        """Clear all documents from the store"""
        self._hybrid = None  # drop any cached lexical index
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={
                "hnsw:space": "cosine",
                EMBEDDING_DIM_METADATA: self.embedding_service.dimension,
            },
        )
