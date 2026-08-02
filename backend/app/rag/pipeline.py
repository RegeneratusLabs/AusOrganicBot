import json
import time
from typing import List, Dict, Any, AsyncGenerator
from app.rag.retriever import VectorStore
from app.llm.llm_client import LLMClient
from app.schemas import SourceDocument
from app.config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.50  # below this, treat as out-of-scope / non-standards query

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RAG_PROMPT = """You are AusOrganicBot, an assistant for Australian organic certification queries.

Guidelines:
- Answer from the provided context. When the context contains the answer, be precise and cite section numbers (e.g. "Section 4.1.2 states…").
- If the context does NOT contain the answer, say so honestly: "I don't see that covered in the standards documents. Please consult your certifier."
- Be concise and practical. No preamble or filler.
- Do not fabricate rules — stick to what's in the context.

Relevant excerpts from the standards:
{context}

Question: {question}

Answer:"""

CONVERSATIONAL_PROMPT = """You are AusOrganicBot, a friendly assistant for Australian organic certification. You help farmers, producers, and certifiers with compliance questions.

The user's message doesn't match a specific compliance question in the standards documents. Reply naturally based on what they said:

- If they greeted you: greet them back warmly in 1-2 sentences and invite them to ask about organic certification (buffer zones, input approvals, record-keeping, transition).
- If they asked what you can do: briefly list your capabilities.
- If they said thanks: acknowledge and invite further questions.
- If their question is off-topic or unclear: politely redirect to organic certification topics.

User: {question}

Response:"""


class RAGPipeline:
    """RAG pipeline for compliance Q&A with relevance-based conversational fallback."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = LLMClient()
        self.temperature = DEFAULT_TEMPERATURE
        self.max_tokens = DEFAULT_MAX_TOKENS

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query — compliance or conversational fallback."""
        start_time = time.time()

        results = self.vector_store.search(query=question, top_k=top_k)

        # --- Relevance gate ---
        # ChromaDB always returns n_results; check if the top hit is actually relevant
        if not results or results[0]["score"] < RELEVANCE_THRESHOLD:
            top_score = results[0]["score"] if results else 0.0
            prompt = CONVERSATIONAL_PROMPT.format(question=question)
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
        prompt = RAG_PROMPT.format(context=context, question=question)

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
        if not results or results[0]["score"] < RELEVANCE_THRESHOLD:
            prompt = CONVERSATIONAL_PROMPT.format(question=question)
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
        prompt = RAG_PROMPT.format(context=context, question=question)

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
