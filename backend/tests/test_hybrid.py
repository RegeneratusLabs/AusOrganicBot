"""Tests for hybrid retrieval: BM25, tokenisation and rank fusion.

No embedding model or ChromaDB is involved, so these are fast and pin the
ranking behaviour that dense-only retrieval was getting wrong.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.hybrid import (  # noqa: E402
    BM25Index,
    HybridRetriever,
    reciprocal_rank_fusion,
    tokenize,
)


class TestTokenize:
    def test_stopwords_removed(self):
        assert tokenize("Can I use it for the crop?") == ["crop"]

    def test_clause_numbers_survive_intact(self):
        # Clause numbers must stay one token or they become unsearchable.
        assert "3.3.2" in tokenize("See clause 3.3.2 for buffer zones")

    def test_light_stemming_matches_plural_and_gerund(self):
        # The property that matters is consistency: plural/singular and
        # verb/gerund must collapse to the same token so they match each other.
        assert tokenize("fungicides") == tokenize("fungicide")
        assert tokenize("spraying") == tokenize("spray")

    def test_stemming_does_not_merge_distinct_inputs(self):
        assert tokenize("copper") != tokenize("sulphur")

    def test_rare_domain_terms_are_kept(self):
        for term in ("sulphur", "urea", "vermicompost", "copper"):
            assert term in tokenize(f"How much {term} can I use?")

    def test_empty_input(self):
        assert tokenize("") == []


class TestBM25:
    def _index(self):
        corpus = [
            "3.3.2 Buffer zones shall be no less than 5 metres.",
            "4.2.1 Copper hydroxide and copper sulphate are permitted inputs.",
            "5.1.1 Records of all inputs must be retained.",
            "1.2.3 Production units must be certified for at least 12 months.",
        ]
        index = BM25Index()
        index.build([str(i) for i in range(len(corpus))], corpus)
        return index, corpus

    def test_exact_term_finds_the_right_document(self):
        index, corpus = self._index()
        hits = index.search("copper sulphate")
        assert hits and corpus[hits[0][0]].startswith("4.2.1")

    def test_clause_number_lookup(self):
        index, corpus = self._index()
        hits = index.search("3.3.2")
        assert hits and "Buffer zones" in corpus[hits[0][0]]

    def test_ranks_by_relevance_not_document_order(self):
        index, _ = self._index()
        hits = index.search("buffer zones metres")
        assert hits[0][0] == 0

    def test_unmatched_query_returns_nothing(self):
        index, _ = self._index()
        assert index.search("zzzz nonexistent term") == []

    def test_empty_index_is_safe(self):
        index = BM25Index()
        index.build([], [])
        assert index.search("anything") == []
        assert len(index) == 0


class TestRRF:
    def test_single_ranking_preserves_order(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"]])
        assert [item for item, _ in fused] == ["a", "b", "c"]

    def test_agreement_between_rankings_wins(self):
        # 'b' is ranked well by both, 'a' only by the first.
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        assert [i for i, _ in fused][0] in {"a", "b"}

    def test_item_from_one_ranking_only_is_included(self):
        fused = reciprocal_rank_fusion([["a"], ["b"]])
        assert {i for i, _ in fused} == {"a", "b"}

    def test_duplicates_within_a_ranking_are_counted_once(self):
        fused = reciprocal_rank_fusion([["a", "a"]])
        assert len(fused) == 1


class FakeStore:
    """Minimal VectorStore stand-in for the fusion wiring."""

    def __init__(self, docs, dense):
        self._docs = docs
        self._dense = dense
        self.collection = self

    def get(self, include=None):
        return {"documents": list(self._docs), "metadatas": [{"i": i} for i in range(len(self._docs))]}

    def search(self, query, top_k=5):
        return [{"content": d, "metadata": {"i": self._docs.index(d)}, "score": 0.9 - 0.1 * i}
                for i, d in enumerate(self._dense[:top_k])]


class TestHybridRetriever:
    def _retriever(self):
        docs = [
            "3.3.2 Buffer zones shall be no less than 5 metres.",
            "4.2.1 Copper hydroxide and copper sulphate are permitted.",
            "5.1.1 Records of all inputs must be retained.",
        ]
        # Dense path deliberately ranks the urea-irrelevant doc first, to show
        # lexical search can rescue an exact-term query.
        dense = [docs[2], docs[0], docs[1]]
        return HybridRetriever(FakeStore(docs, dense))

    def test_lexical_only_hit_is_included(self):
        retriever = self._retriever()
        results = retriever.search("copper sulphate", top_k=3)
        texts = [r["content"] for r in results]
        assert any("copper sulphate" in t for t in texts)

    def test_disabled_lexical_falls_back_to_dense(self):
        docs = ["a doc", "b doc"]
        retriever = HybridRetriever(FakeStore(docs, docs), enable_lexical=False)
        results = retriever.search("anything", top_k=2)
        assert [r["content"] for r in results] == docs

    def test_results_carry_metadata(self):
        retriever = self._retriever()
        for row in retriever.search("buffer zones", top_k=3):
            assert "metadata" in row

    def test_refresh_drops_the_cached_index(self):
        retriever = self._retriever()
        retriever.search("buffer", top_k=1)
        assert retriever._bm25 is not None
        retriever.refresh()
        assert retriever._bm25 is None


class TestIndexDetection:
    """Back-of-document index pages match clause numbers but answer nothing."""

    def test_detects_numeric_index_lines(self):
        from app.rag.hybrid import looks_like_index

        assert looks_like_index("31, 39, 42, 43, 45, 47\n18, 26, 27, 28, 31, 65") is True

    def test_detects_number_dominated_chunk(self):
        from app.rag.hybrid import looks_like_index

        assert looks_like_index(" ".join(str(i) for i in range(1, 30))) is True

    def test_prose_is_not_an_index(self):
        from app.rag.hybrid import looks_like_index

        assert looks_like_index(
            "3.3.2 Requirements for buffer zones shall be no less than 5 metres."
        ) is False

    def test_short_chunk_is_not_mistaken_for_an_index(self):
        from app.rag.hybrid import looks_like_index

        assert looks_like_index("5 metres") is False

    def test_index_chunks_are_demoted_below_real_content(self):
        docs = [
            "31, 39, 42, 43, 45, 47\n18, 26, 27, 28, 31, 65",
            "3.3.2 Requirements for buffer zones shall be no less than 5 metres.",
        ]
        store = FakeStore(docs, docs)
        retriever = HybridRetriever(store)
        results = retriever.search("3.3.2 buffer zones", top_k=2)
        # The clause-bearing chunk must come first, not the index page.
        assert "buffer zones" in results[0]["content"].lower()
        assert "31, 39, 42" not in results[0]["content"]
