import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.rag.retriever import VectorStore
from app.llm.llm_client import LLMClient
from app.schemas import SourceDocument
from app.prompts import (
    RAG_PROMPT,
    RAG_RETRY_SUFFIX,
    CONVERSATIONAL_PROMPT,
    COMPLIANCE_KEYWORDS,
    NOT_COVERED_ANSWER,
)
from app.rag.citations import verify_answer, CitationCheck
from app.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    RELEVANCE_THRESHOLD,
    DEFAULT_COUNTRY,
    DATA_DIR,
)
from app.data.manifest import CountryManifest, load_manifest

# ---------------------------------------------------------------------------
# Multi-turn conversation bounds (protect the model's 4096-token context).
# ---------------------------------------------------------------------------
HISTORY_MAX_TURNS = 6
HISTORY_TURN_CHARS = 600

# Shown when the reader produced an answer whose citations or quantities could
# not be traced back to the retrieved excerpts. Better an honest refusal than a
# confident wrong rule — see app/rag/citations.py.
UNVERIFIED_REFUSAL = (
    "I found material that looks relevant, but I couldn't verify that the specific "
    "rule, clause or figure I'd need to quote is actually in the standards documents "
    "I have available — so I won't guess at it. Please check with your certifier, or "
    "ask me something more specific and I'll search again."
)

# Streaming is cosmetic once the answer is already generated and verified; we
# re-emit it in small pieces so the UI still renders progressively.
_STREAM_PIECE_CHARS = 160


def _split_for_stream(text: str, size: int = _STREAM_PIECE_CHARS) -> List[str]:
    """Split verified answer text into whitespace-friendly pieces for streaming."""
    if not text:
        return []
    pieces: List[str] = []
    remaining = text
    while len(remaining) > size:
        cut = remaining.rfind(" ", 0, size)
        if cut <= 0:
            cut = size
        pieces.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        pieces.append(remaining)
    return pieces

# ---------------------------------------------------------------------------
# Prompts & constants live in app/prompts.py (dependency-free: those templates
# are the single source of truth for what the app sends).
# ---------------------------------------------------------------------------


class RAGPipeline:
    """RAG pipeline for compliance Q&A with relevance-based conversational fallback."""

    def __init__(self, country: str = DEFAULT_COUNTRY, data_dir: Path = DATA_DIR):
        self.country = country.lower()
        self.data_dir = data_dir
        self.manifest: Optional[CountryManifest] = None
        try:
            self.manifest = load_manifest(data_dir, self.country)
        except FileNotFoundError:
            pass

        self.vector_store = VectorStore(country=self.country)
        self.llm = LLMClient()
        self.temperature = DEFAULT_TEMPERATURE
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.relevance_threshold = (
            self.manifest.relevance_threshold
            if self.manifest and self.manifest.relevance_threshold is not None
            else RELEVANCE_THRESHOLD
        )

    @property
    def country_label(self) -> str:
        if self.manifest:
            return f"{self.manifest.name} ({self.country.upper()})"
        return self.country.upper()

    def _standards_block(self) -> str:
        if not self.manifest:
            return f"Standards documents for country code {self.country}."
        lines = [
            f"- {s.name}, Edition {s.edition} ({self.country.upper()})"
            for s in self.manifest.active_sources
        ]
        return "\n".join(lines) if lines else "(no active sources declared in manifest)"

    def _language_instruction(self) -> str:
        if self.manifest and self.manifest.language and self.manifest.language != "en":
            return f"Answer in {self.manifest.language}.\n"
        return ""

    def _rag_prompt(self, context: str, question: str, history: Optional[List[dict]] = None, low_confidence: bool = False) -> str:
        history_block = self._history_block(history)
        return RAG_PROMPT.format(
            country_label=self.country_label,
            standards_block=self._standards_block(),
            language_instruction=self._language_instruction(),
            history_block=history_block,
            low_confidence_note=(
                "Note: the retrieved excerpts below are a weaker match for this question, "
                "so if they do not actually answer it, keep your reply short and ask one "
                "clarifying question or point them to a certifier — do not guess.\n"
                if low_confidence else ""
            ),
            context=context,
            question=question,
        )

    def _conversational_prompt(self, question: str) -> str:
        return CONVERSATIONAL_PROMPT.format(
            country_label=self.country_label, question=question
        )

    @staticmethod
    def _looks_like_compliance_question(question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in COMPLIANCE_KEYWORDS)

    @staticmethod
    def _history_block(history: Optional[List[dict]] = None) -> str:
        """Render the prior conversation (bounded) as a compact 'Conversation so far' block."""
        if not history:
            return ""
        lines = ["Conversation so far:"]
        for turn in history[-HISTORY_MAX_TURNS:]:
            role = turn.get("role", "user")
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            label = "User" if role != "assistant" else "Assistant"
            # truncate long turns to protect the 4096-token context window
            if len(content) > HISTORY_TURN_CHARS:
                content = content[:HISTORY_TURN_CHARS] + "…"
            lines.append(f"- {label}: {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Answer verification (deterministic — see app/rag/citations.py)
    # ------------------------------------------------------------------

    async def _generate_verified(
        self,
        prompt: str,
        chunks: List[str],
        temperature: float,
        history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Generate once, verify citations/quantities, retry strictly, then refuse.

        Returns a dict with the final answer plus verification metadata. A
        rejection that survives the retry becomes an explicit refusal rather
        than a plausible-sounding but untraceable rule.
        """
        llm_out = await self.llm.generate(
            prompt=prompt, temperature=temperature, max_tokens=self.max_tokens, history=history,
        )
        check = verify_answer(llm_out["content"], chunks)
        if check.ok:
            return {"answer": llm_out["content"], "reasoning": llm_out.get("reasoning", ""),
                    "verified": True, "verification": check, "retried": False}

        retry_out = await self.llm.generate(
            prompt=prompt + RAG_RETRY_SUFFIX, temperature=0.0,
            max_tokens=self.max_tokens, history=history,
        )
        retry_check = verify_answer(retry_out["content"], chunks)
        if retry_check.ok:
            return {"answer": retry_out["content"], "reasoning": retry_out.get("reasoning", ""),
                    "verified": True, "verification": retry_check, "retried": True}

        # Both attempts contain claims we cannot trace to the excerpts. Refuse.
        return {"answer": UNVERIFIED_REFUSAL, "reasoning": retry_out.get("reasoning", ""),
                "verified": False, "verification": retry_check, "retried": True,
                "refused": True, "refusal_reason": retry_check.reason()}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(
        self, question: str, top_k: int = 5, history: Optional[List[dict]] = None
    ) -> Dict[str, Any]:
        """Process a query — compliance or conversational fallback."""
        start_time = time.time()

        results = self.vector_store.search_hybrid(query=question, top_k=top_k)

        # --- Routing ---
        # No retrieval at all -> conversational (greetings, thanks, off-topic).
        # Below-threshold + non-compliance -> conversational too.
        # Below-threshold + compliance-shaped -> RAG with a low-confidence note:
        #   the reader answers from context if it actually covers the
        #   question, or declines warmly (no canned refusal, no hallucination).
        if not results or (
            results[0]["score"] < self.relevance_threshold
            and not self._looks_like_compliance_question(question)
        ):
            prompt = self._conversational_prompt(question)
            llm_out = await self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=self.max_tokens,
                history=history,
            )
            return {
                "answer": llm_out["content"],
                "reasoning": llm_out.get("reasoning", ""),
                "sources": [],
                "confidence": results[0]["score"] if results else 0.0,
                "model": self.llm.model,
                "provider": self.llm.provider,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        # --- RAG path (the reader is prompted to answer from retrieved
        # context, or refuse honestly when it doesn't cover the question) ---
        context_parts = []
        sources = []

        for i, doc in enumerate(results):
            context_parts.append(f"[Document {i + 1}]\n{doc['content']}")
            sources.append(
                SourceDocument(
                    content=doc["content"], metadata=doc["metadata"], score=doc["score"]
                )
            )

        context = "\n\n".join(context_parts)
        top_score = results[0]["score"]
        low_confidence = top_score < self.relevance_threshold
        prompt = self._rag_prompt(
            context=context, question=question, history=history, low_confidence=low_confidence
        )

        llm_out = await self._generate_verified(
            prompt=prompt,
            chunks=[doc["content"] for doc in results],
            temperature=self.temperature,
            history=history,
        )

        avg_confidence = sum(doc["score"] for doc in results) / len(results)
        processing_time = (time.time() - start_time) * 1000

        return {
            "answer": llm_out["answer"],
            "reasoning": llm_out.get("reasoning", ""),
            "sources": sources,
            "confidence": avg_confidence,
            "model": self.llm.model,
            "provider": self.llm.provider,
            "processing_time_ms": processing_time,
            "verified": llm_out["verified"],
            "unsupported_citations": llm_out["verification"].unsupported_citations,
            "refused": llm_out.get("refused"),
            "refusal_reason": llm_out.get("refusal_reason"),
        }

    async def query_stream(
        self, question: str, top_k: int = 5, history: Optional[List[dict]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response — compliance or conversational fallback.

        Compliance answers are generated and citation-verified *before* the
        first token is emitted, then streamed from the verified text. A
        1B-class model cannot be trusted to police its own citations mid-stream
        and a correction event after the fact would be both confusing and
        unreliable, so we accept added latency in exchange for never streaming
        an unverifiable rule. Conversational turns retrieve nothing and stream
        directly.
        """
        results = self.vector_store.search_hybrid(query=question, top_k=top_k)

        if not results or (
            results[0]["score"] < self.relevance_threshold
            and not self._looks_like_compliance_question(question)
        ):
            prompt = self._conversational_prompt(question)
            async for evt in self.llm.generate_stream(
                prompt=prompt, temperature=0.7, max_tokens=self.max_tokens, history=history,
            ):
                yield json.dumps({"type": evt["type"], "content": evt["chunk"]}) + "\n"
            yield json.dumps({"done": True})
            return

        # --- RAG path: verify first, then stream the verified answer ---
        context_parts = []
        for i, doc in enumerate(results):
            context_parts.append(f"[Document {i + 1}]\n{doc['content']}")

        context = "\n\n".join(context_parts)
        low_confidence = results[0]["score"] < self.relevance_threshold
        prompt = self._rag_prompt(
            context=context, question=question, history=history, low_confidence=low_confidence
        )

        verified = await self._generate_verified(
            prompt=prompt,
            chunks=[doc["content"] for doc in results],
            temperature=self.temperature,
            history=history,
        )

        # Tell the client how the answer was verified before its text arrives.
        yield json.dumps({
            "type": "verification",
            "verified": verified["verified"],
            "unsupported_citations": verified["verification"].unsupported_citations,
            "refused": bool(verified.get("refused")),
        }) + "\n"

        for piece in _split_for_stream(verified["answer"]):
            yield json.dumps({"type": "content", "content": piece}) + "\n"
        yield json.dumps({"done": True})

    # ------------------------------------------------------------------
    # Input approval checker
    # ------------------------------------------------------------------

    async def check_input_approval(
        self, product_name: str, ingredients: List[str], intended_use: str
    ) -> Dict[str, Any]:
        """Check if an input product is approved for organic use"""
        ingredient_str = ", ".join(ingredients)
        query = f"Is {product_name} with ingredients {ingredient_str} approved for use in {intended_use} under organic certification standards?"
        result = await self.query(query, top_k=5)

        answer_lower = result["answer"].lower()
        approved_keywords = ["approved", "permitted", "allowed", "acceptable"]
        rejected_keywords = [
            "prohibited", "not allowed", "not permitted", "forbidden", "banned",
        ]

        is_approved = any(kw in answer_lower for kw in approved_keywords)
        is_rejected = any(kw in answer_lower for kw in rejected_keywords)

        if is_rejected and not is_approved:
            status = "Not Approved"
        elif is_approved:
            status = "Approved"
        else:
            status = "Uncertain - Review Required"

        return {
            "product_name": product_name,
            "approved": is_approved and not is_rejected,
            "status": status,
            "details": [result["answer"]],
            "sources": result["sources"],
        }
