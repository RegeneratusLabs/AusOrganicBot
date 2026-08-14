"""Tests for the ChromaDB retriever (no real embedding model downloads)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import retriever as retriever_mod
from app.rag.retriever import VectorStore


class FakeEmbeddings:
    def __init__(self, dim=8):
        self.dim = dim
        self.dimension = dim

    def embed(self, text):
        # Deterministic pseudo-vector derived from text
        h = hash(text)
        return [float((h >> (i * 4)) % 100) / 100.0 for i in range(self.dim)]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever_mod, "CHROMA_PERSIST_DIR", tmp_path / "chroma")
    monkeypatch.setattr(retriever_mod, "EmbeddingService", FakeEmbeddings)
    vs = VectorStore(collection_name="test_collection")
    yield vs
    vs.client.delete_collection("test_collection")


def test_country_collections_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever_mod, "CHROMA_PERSIST_DIR", tmp_path / "chroma")
    monkeypatch.setattr(retriever_mod, "EmbeddingService", FakeEmbeddings)

    au = VectorStore(country="au")
    nz = VectorStore(country="nz")

    au.add_documents(
        documents=["Australian buffer rules."],
        metadatas=[{"source": "national", "country_code": "au"}],
        ids=["au-1"],
    )
    nz.add_documents(
        documents=["NZ compost rules."],
        metadatas=[{"source": "biogro", "country_code": "nz"}],
        ids=["nz-1"],
    )

    assert au.get_document_count() == 1
    assert nz.get_document_count() == 1
    assert au.search("compost rules")[0]["metadata"]["country_code"] == "au"
    assert nz.search("compost rules")[0]["metadata"]["country_code"] == "nz"

    # Clearing one country must not touch the other
    au.clear()
    assert au.get_document_count() == 0
    assert nz.get_document_count() == 1

    au.client.delete_collection(au.collection.name)
    nz.client.delete_collection(nz.collection.name)


def test_dimension_mismatch_rebuilds_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever_mod, "CHROMA_PERSIST_DIR", tmp_path / "chroma")

    class DimA(FakeEmbeddings):
        def __init__(self):
            super().__init__(dim=8)

    monkeypatch.setattr(retriever_mod, "EmbeddingService", DimA)
    vs = VectorStore(country="au")
    vs.add_documents(["hello world text"], [{"source": "x"}], ["1"])
    assert vs.get_document_count() == 1
    vs.client.delete_collection(vs.collection.name)

    class DimB(FakeEmbeddings):
        def __init__(self):
            super().__init__(dim=4)

    monkeypatch.setattr(retriever_mod, "EmbeddingService", DimB)
    vs2 = VectorStore(country="au")
    assert vs2.get_document_count() == 0  # rebuilt because dim changed
    vs2.client.delete_collection(vs2.collection.name)


def test_add_and_count(store):
    store.add_documents(
        documents=["Buffer zones must be at least 15 metres.", "Records kept for 5 years."],
        metadatas=[
            {"source": "national", "filename": "std.pdf"},
            {"source": "aco", "filename": "aco.pdf"},
        ],
        ids=["a", "b"],
    )
    assert store.get_document_count() == 2


def test_search_returns_results_with_metadata(store):
    store.add_documents(
        documents=["Buffer zone rules text.", "Compost temperature rules."],
        metadatas=[{"source": "national"}, {"source": "aco"}],
        ids=["buf", "comp"],
    )
    results = store.search("buffer zone requirements", top_k=2)
    assert len(results) == 2
    assert {r["metadata"]["source"] for r in results} == {"national", "aco"}
    assert all(r["score"] <= 1.0 for r in results)
    assert all("content" in r for r in results)


def test_search_empty(store):
    assert store.search("anything") == []


def test_clear(store):
    store.add_documents(["hello"], [{"source": "x"}], ["1"])
    store.clear()
    assert store.get_document_count() == 0
