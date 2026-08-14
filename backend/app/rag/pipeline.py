import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.rag.retriever import VectorStore
from app.llm.llm_client import LLMClient
from app.schemas import SourceDocument
from app.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    RELEVANCE_THRESHOLD,
    DEFAULT_COUNTRY,
    DATA_DIR,
)
from app.data.manifest import CountryManifest, load_manifest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOT_COVERED_ANSWER = (
    "I don't see that covered in the standards documents. Please consult your certifier."
)

# Below-threshold queries that still look like compliance questions must NOT be
# answered from model knowledge (hallucination risk). They get a refusal.
COMPLIANCE_KEYWORDS = [
    "organic", "biodynamic", "certif", "standard", "complian", "permit", "approv",
    "allowed", "prohibit", "buffer", "input", "label", "record", "audit", "inspect",
    "transition", "conversion", "compost", "manure", "seed", "fertiliser",
    "fertilizer", "pesticide", "fungicide", "insecticide", "feed", "livestock",
    "poultry", "slaughter", "import", "export", "acreage", "paddock",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RAG_PROMPT = """You are AusOrganicBot, an assistant for {country_label} organic certification queries.

Applicable standards (answer only from these):
{standards_block}

Guidelines:
- Answer from the provided context. When the context contains the answer, be precise and cite section numbers (e.g. "Section 4.1.2 states…"). Mention which standard you are citing when multiple are shown.
- If the context does NOT contain the answer, say so honestly: "I don't see that covered in the standards documents. Please consult your certifier."
- Be concise and practical. No preamble or filler.
- Do not fabricate rules — stick to what's in the context.
{language_instruction}
Relevant excerpts from the standards:
{context}

Question: {question}

Answer:"""

CONVERSATIONAL_PROMPT = """You are AusOrganicBot, a friendly assistant for {country_label} organic certification. You help farmers, producers, and certifiers with compliance questions.

The user's message doesn't match a specific compliance question in the standards documents. Reply naturally based on what they said:

- If they greeted you: greet them back warmly in 1-2 sentences and invite them to ask about organic certification (buffer zones, input approvals, record-keeping, transition).
- If they asked what you can do: briefly list your capabilities.
- If they said thanks: acknowledge and invite further questions.
- If their question is off-topic or unclear: politely redirect to organic certification topics.

User: {question}

Response:"""


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

    def _rag_prompt(self, context: str, question: str) -> str:
        return RAG_PROMPT.format(
            country_label=self.country_label,
            standards_block=self._standards_block(),
            language_instruction=self._language_instruction(),
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

    def _conversational_fallback(self, question: str) -> Dict[str, Any]:
        """Below-threshold path: refuse compliance-shaped questions instead of
        answering from model knowledge (hallucination guard)."""
        if self._looks_like_compliance_question(question):
            return {
                "answer": NOT_COVERED_ANSWER,
                "reasoning": "",
                "sources": [],
                "confidence": 0.0,
                "refused": True,
            }
        return None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query — compliance or conversational fallback."""
        start_time = time.time()

        results = self.vector_store.search(query=question, top_k=top_k)

        # --- Relevance gate ---
        # ChromaDB always returns n_results; check if the top hit is actually relevant
        if not results or results[0]["score"] < self.relevance_threshold:
            top_score = results[0]["score"] if results else 0.0
            refused = self._conversational_fallback(question)
            if refused is not None:
                refused["processing_time_ms"] = (time.time() - start_time) * 1000
                refused["model"] = self.llm.model
                refused["provider"] = self.llm.provider
                return refused

            prompt = self._conversational_prompt(question)
            llm_out = await self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=self.max_tokens,
            )
            return {
                "answer": llm_out["content"],
                "reasoning": llm_out.get("reasoning", ""),
                "sources": [],
                "confidence": top_score,
                "model": self.llm.model,
                "provider": self.llm.provider,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        # --- Compliance / RAG path ---
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
        prompt = self._rag_prompt(context=context, question=question)

        llm_out = await self.llm.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        avg_confidence = sum(doc["score"] for doc in results) / len(results)
        processing_time = (time.time() - start_time) * 1000

        return {
            "answer": llm_out["content"],
            "reasoning": llm_out.get("reasoning", ""),
            "sources": sources,
            "confidence": avg_confidence,
            "model": self.llm.model,
            "provider": self.llm.provider,
            "processing_time_ms": processing_time,
        }

    async def query_stream(self, question: str, top_k: int = 5) -> AsyncGenerator[str, None]:
        """Stream a response — compliance or conversational fallback."""
        results = self.vector_store.search(query=question, top_k=top_k)

        # --- Relevance gate ---
        if not results or results[0]["score"] < self.relevance_threshold:
            refused = self._conversational_fallback(question)
            if refused is not None:
                yield json.dumps({"type": "content", "content": refused["answer"]}) + "\n"
                yield json.dumps({"done": True})
                return

            prompt = self._conversational_prompt(question)
            async for evt in self.llm.generate_stream(
                prompt=prompt, temperature=0.7, max_tokens=self.max_tokens,
            ):
                yield json.dumps({"type": evt["type"], "content": evt["chunk"]}) + "\n"
            yield json.dumps({"done": True})
            return

        # --- Compliance / RAG path ---
        context_parts = []
        for i, doc in enumerate(results):
            context_parts.append(f"[Document {i + 1}]\n{doc['content']}")

        context = "\n\n".join(context_parts)
        prompt = self._rag_prompt(context=context, question=question)

        async for evt in self.llm.generate_stream(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            yield json.dumps({"type": evt["type"], "content": evt["chunk"]}) + "\n"

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
