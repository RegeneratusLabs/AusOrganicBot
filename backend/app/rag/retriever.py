import chromadb
from typing import List, Dict, Any
from app.config import CHROMA_PERSIST_DIR, TOP_K_RETRIEVAL
from app.rag.embeddings import EmbeddingService


class VectorStore:
    """ChromaDB vector store for document retrieval"""

    def __init__(self, collection_name: str = "organic_standards"):
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

        self.embedding_service = EmbeddingService()

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]
    ):
        """Add documents to the vector store"""
        embeddings = self.embedding_service.embed_batch(documents)

        self.collection.add(
            embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
        )

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Search for similar documents"""
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

    def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        return self.collection.count()

    def clear(self):
        """Clear all documents from the store"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"}
        )
