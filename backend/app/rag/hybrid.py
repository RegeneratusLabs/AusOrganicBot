"""Hybrid retrieval: dense vectors fused with BM25 lexical search.

Dense embeddings average a question into one vector, which loses the specific
tokens that make a compliance question answerable. In testing, "Can sulphur be
used as a fungicide in organic production?" missed the gold chunk at rank 5+ even
though "sulphur" and "fungicide" both appear in the standards — the dense
representation had smoothed them into the general topic of organic production.

Lexical search has the opposite profile: excellent on rare, exact terms
("sulphur", "3.3.2", "copper sulfate") and poor on paraphrase. Fusing the two
with Reciprocal Rank Fusion gives a ranker that is strong on both, with no model
download and negligible CPU cost.

RRF is used rather than score normalisation because dense cosine similarity and
BM25 scores are not comparable scales; RRF only needs the *rankings*, so it
cannot be skewed by either signal's score distribution.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Tokeniser: keep alphanumerics and internal dots so clause numbers like
# "3.3.2" survive as a single searchable token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")

# Words carrying no discriminative signal in this corpus. Deliberately short:
# over-aggressive stopwording hurts the rare-term matches BM25 exists to catch.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "in", "is", "it", "its", "may", "me",
    "must", "my", "of", "on", "or", "should", "that", "the", "their", "there",
    "these", "this", "to", "under", "use", "used", "was", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}

# A simple suffix stemmer. Crude on purpose: it exists so that "fungicides"
# matches "fungicide" and "spraying" matches "spray", which is the failure mode
# observed. Anything more aggressive starts merging distinct legal terms.
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "es", "s")

# Back-of-document index pages are dense with clause numbers and page refs
# ("31, 39, 42, 43, 45 ..."), which makes them win any query containing a clause
# number while answering nothing. They are detected so they can be demoted in
# the lexical ranking rather than removed from the corpus.
_INDEX_LINE_RE = re.compile(r"^\s*[\d,\s.\-–]+\s*$")
_INDEX_TOKEN_RE = re.compile(r"^[a-z]{0,3}\d+[a-z]*$|^\d+$")


def looks_like_index(text: str) -> bool:
    """True when a chunk is dominated by page numbers / index entries."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    numeric_lines = sum(1 for ln in lines if _INDEX_LINE_RE.match(ln))
    if numeric_lines >= max(2, len(lines) // 2):
        return True
    tokens = (text or "").split()
    if len(tokens) < 12:
        return False
    numeric = sum(1 for t in tokens if _INDEX_TOKEN_RE.match(t.strip(",.;:")))
    return numeric / len(tokens) > 0.5


def _stem(word: str) -> str:
    """Reduce a word to a retrieval token.

    Order matters. Plurals are handled before the trailing-'e' rule so that
    "fungicides" -> "fungicid" and "fungicide" -> "fungicid" agree, rather than
    the plural and singular diverging (which silently defeats matching).
    """
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    # Only after plural stripping: otherwise "fungicide" keeps its 'e' and no
    # longer matches the plural form.
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """Lowercase tokens, dropping stopwords and applying light stemming."""
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        if not raw[0].isdigit():
            raw = _stem(raw)
        tokens.append(raw)
    return tokens


class BM25Index:
    """In-memory BM25 (Okapi) over a fixed corpus of chunks.

    Parameters are the standard defaults (k1=1.5, b=0.75). The corpus is built
    once per collection and held in memory; for the installed countries
    (100-800 chunks each) this is well under a megabyte.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_len: List[int] = []
        self.avg_len: float = 0.0
        self._df: Counter = Counter()

    def __len__(self) -> int:
        return len(self.doc_ids)

    def build(self, ids: Sequence[str], documents: Sequence[str]) -> None:
        self.doc_ids = list(ids)
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        self._df = df

    def _idf(self, term: str) -> float:
        """BM25 idf with the standard +0.5 smoothing."""
        n = len(self.doc_ids)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Return [(index_into_corpus, score)] ordered by descending score."""
        if not self.doc_ids or self.avg_len == 0:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: List[tuple] = []
        for i, freqs in enumerate(self.doc_freqs):
            length = self.doc_len[i] or 1
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term)
                if not tf:
                    continue
                idf = self._idf(term)
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += idf * (tf * (self.k1 + 1)) / denominator
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[Any]],
    k: int = 60,
) -> List[tuple]:
    """Fuse ranked lists by Reciprocal Rank Fusion.

    Returns [(item, fused_score)] ordered by descending fused score. `k=60` is
    the value from the original RRF paper and is not sensitive.
    """
    fused: Dict[Any, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    """Dense retrieval fused with BM25, over one country's collection.

    Wraps an existing VectorStore so the dense path, embedding model and
    metadata handling stay identical to the non-hybrid path.
    """

    def __init__(self, vector_store, enable_lexical: bool = True):
        self.vector_store = vector_store
        self.enable_lexical = enable_lexical
        self._bm25: Optional[BM25Index] = None
        self._documents: List[str] = []
        self._text_meta: Dict[str, Dict[str, Any]] = {}

    # -- corpus ------------------------------------------------------------
    def _ensure_index(self) -> None:
        """Build the lexical index once from the collection's stored chunks."""
        if self._bm25 is not None:
            return
        data = self.vector_store.collection.get(include=["documents", "metadatas"])
        self._documents = list(data.get("documents") or [])
        metadatas = list(data.get("metadatas") or [])
        self._text_meta = {}
        for text, meta in zip(self._documents, metadatas):
            self._text_meta.setdefault(text, meta or {})

        index = BM25Index()
        index.build([str(i) for i in range(len(self._documents))], self._documents)
        self._bm25 = index

    def refresh(self) -> None:
        """Drop the cached index (call after re-ingesting)."""
        self._bm25 = None
        self._documents = []
        self._text_meta = {}

    # -- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, pool: int = 20) -> List[Dict[str, Any]]:
        """Return top_k results, fusing the dense and lexical rankings.

        `pool` is how many candidates each signal contributes before fusion; it
        must exceed `top_k` so fusion has something to reorder.
        """
        pool = max(pool, top_k)
        dense = self.vector_store.search(query, top_k=pool)
        if not self.enable_lexical:
            return dense[:top_k]

        self._ensure_index()
        assert self._bm25 is not None
        if len(self._bm25) == 0:
            return dense[:top_k]

        dense_by_text: Dict[str, Dict[str, Any]] = {row["content"]: row for row in dense}
        dense_order = [row["content"] for row in dense]

        lexical_order = [
            self._documents[row_index]
            for row_index, _score in self._bm25.search(query, top_k=pool)
            if 0 <= row_index < len(self._documents)
        ]
        # Move index-page noise to the back of the lexical ranking: it matches
        # any clause number but answers nothing, and would otherwise crowd out
        # the real clause in the fused result.
        lexical_order.sort(key=lambda text: looks_like_index(text))
        if not lexical_order:
            return dense[:top_k]

        fused = reciprocal_rank_fusion([dense_order, lexical_order])

        # Final demotion pass: an index page can also rank well densely, so
        # demoting it in one ranking is not enough. Retrieval must never return
        # a back-of-document index as the context for an answer, so index-like
        # chunks are pushed to the end of the fused order.
        fused.sort(key=lambda pair: looks_like_index(pair[0]))

        results: List[Dict[str, Any]] = []
        for text, fused_score in fused[:top_k]:
            if text in dense_by_text:
                row = dict(dense_by_text[text])
                row["retrieval"] = "hybrid"
            else:
                # Lexical-only hit: no dense score, so carry a score of 0.0. The
                # relevance gate will treat it as below-threshold, which routes
                # it down the cautious path rather than silently trusting it.
                row = {
                    "content": text,
                    "metadata": self._text_meta.get(text, {}),
                    "score": 0.0,
                    "retrieval": "lexical",
                }
            row["fused_score"] = fused_score
            results.append(row)

        return results
