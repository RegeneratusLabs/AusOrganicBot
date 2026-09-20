"""Tests for the RAG pipeline: routing, low-confidence note, streaming, input check."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import pipeline as pipeline_mod
from app.rag.pipeline import RAGPipeline


class FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k=5):
        return list(self._results)

    # The pipeline calls search_hybrid; this fake returns fixed results either
    # way so routing/verification logic can be tested without retrieval.
    def search_hybrid(self, query, top_k=5, pool=None):
        return list(self._results)


class FakeLLM:
    """Stand-in reader.

    By default it extracts a grounded answer from the excerpts it was actually
    given, so that deterministic citation verification (app/rag/citations.py)
    passes — a fabricated fixture would now be refused by the pipeline, which
    is the behaviour under test elsewhere. Pass `rag_answer` to force a
    specific (possibly unverifiable) answer.
    """

    def __init__(self, rag_answer=None, count_fabricated=False):
        self.model = "fake-model"
        self.provider = "fake"
        self.calls = []
        self.rag_answer = rag_answer
        self.count_fabricated = count_fabricated
        self.generate_calls = 0

    @staticmethod
    def _grounded_answer(prompt: str) -> str:
        """Answer using clause numbers copied verbatim from the excerpts."""
        marker = "Relevant excerpts from the standards:"
        excerpts = prompt.split(marker, 1)[1] if marker in prompt else ""
        import re
        clause = re.search(r"\b\d+\.\d+(?:\.\d+)*\b", excerpts)
        if clause:
            return f"Clause {clause.group(0)} states the requirement shown in the excerpts."
        return "The excerpts do not cover that; please check with your certifier."

    async def generate(self, prompt, temperature=0.1, max_tokens=2048, history=None):
        self.calls.append({
            "type": "generate", "prompt": prompt, "temperature": temperature, "history": history,
        })
        self.generate_calls += 1
        if "Relevant excerpts from the standards:" not in prompt:
            return {"content": "G'day! How can I help?", "reasoning": ""}
        if self.rag_answer is not None:
            return {"content": self.rag_answer, "reasoning": ""}
        if self.count_fabricated:
            # Cite a clause number that is not in the excerpts.
            return {"content": "Section 9.9.9 requires 4 hectares.", "reasoning": ""}
        return {"content": self._grounded_answer(prompt), "reasoning": ""}

    async def generate_stream(self, prompt, temperature=0.1, max_tokens=2048, history=None):
        self.calls.append({"type": "stream", "prompt": prompt, "history": history})
        yield {"type": "content", "chunk": "G'day! How can I help?"}

    async def check_health(self):
        return True


def make_rag(results, rag_answer=None, country="au", manifest=None, threshold=None,
             count_fabricated=False):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.vector_store = FakeVectorStore(results)
    pipeline.llm = FakeLLM(rag_answer, count_fabricated=count_fabricated)
    pipeline.temperature = 0.1
    pipeline.max_tokens = 2048
    pipeline.country = country
    pipeline.data_dir = Path(__file__).parent
    pipeline.manifest = manifest
    pipeline.relevance_threshold = (
        threshold
        if threshold is not None
        else (manifest.relevance_threshold if manifest and manifest.relevance_threshold else pipeline_mod.RELEVANCE_THRESHOLD)
    )
    return pipeline


def doc(content, score, source="national", filename="std.pdf"):
    return {
        "content": content,
        "metadata": {"source": source, "filename": filename, "page_number": 3},
        "score": score,
    }


def test_rag_path_uses_sources(monkeypatch):
    results = [doc("Buffer zones shall be no less than 15 metres.", 0.64)]
    pipeline = make_rag(results)
    result = asyncio.run(pipeline.query("How wide should a buffer zone be?"))
    assert result["sources"] and result["sources"][0].score == 0.64
    assert pipeline.llm.calls[0]["temperature"] == 0.1
    assert "Relevant excerpts" in pipeline.llm.calls[0]["prompt"]
    assert result["confidence"] == pytest.approx(0.64)


def test_below_threshold_non_compliance_conversational(monkeypatch):
    """Greeting with weak retrieval still routes to the conversational path."""
    results = [doc("Some chunk text.", 0.3)]
    pipeline = make_rag(results)
    result = asyncio.run(pipeline.query("thanks for your help"))
    assert result["sources"] == []
    assert "G'day" in result["answer"]
    assert pipeline.llm.calls[0]["temperature"] == 0.7


def test_no_results_conversational():
    pipeline = make_rag([])
    result = asyncio.run(pipeline.query("hello"))
    assert result["sources"] == []
    assert "G'day" in result["answer"]


def test_compliance_below_threshold_reaches_model_with_note():
    """Hallucination guard is now prompt-based: below-threshold compliance
    questions go to the model with the low-confidence note instead of a canned
    refusal, so the model can answer if context covers it or decline warmly."""
    results = [doc("Unrelated chunk.", 0.35)]
    pipeline = make_rag(results, rag_answer="I couldn't find an exact rule covering that…")
    result = asyncio.run(pipeline.query("Do I need a transition plan for certification?"))
    assert len(result["sources"]) == 1
    assert result["sources"][0].score == 0.35
    assert pipeline.llm.calls[0]["temperature"] == 0.1
    prompt = pipeline.llm.calls[0]["prompt"]
    assert "weaker match" in prompt  # low-confidence note present
    assert "Relevant excerpts" in prompt


def test_refusal_keyword_variants():
    assert RAGPipeline._looks_like_compliance_question("Can I use vermicompost?")
    assert RAGPipeline._looks_like_compliance_question("is it allowed to spray neem oil?")
    assert RAGPipeline._looks_like_compliance_question("What are the buffer zone rules?")
    assert not RAGPipeline._looks_like_compliance_question("hello")
    assert not RAGPipeline._looks_like_compliance_question("thanks for your help")


def test_history_included_in_prompt():
    results = [doc("Buffer zones shall be no less than 15 metres.", 0.7)]
    pipeline = make_rag(results)
    history = [
        {"role": "user", "content": "What about buffer zones?"},
        {"role": "assistant", "content": "Section 4.7.19 covers that."},
    ]
    asyncio.run(pipeline.query("and how wide for broadacre?", history=history))
    prompt = pipeline.llm.calls[0]["prompt"]
    assert "Conversation so far:" in prompt
    assert "What about buffer zones?" in prompt
    assert "Section 4.7.19 covers that." in prompt
    # history also forwarded to the LLM client for multi-turn messages
    assert pipeline.llm.calls[0]["history"] == history


def test_history_bounded_and_truncated():
    results = [doc("Chunk.", 0.7)]
    pipeline = make_rag(results)
    long_turn = {"role": "user", "content": "x" * 5000}
    history = [long_turn] + [{"role": "assistant", "content": f"turn {i}"} for i in range(10)]
    asyncio.run(pipeline.query("question?", history=history))
    prompt = pipeline.llm.calls[0]["prompt"]
    # only the last HISTORY_MAX_TURNS turns appear, long turns truncated
    assert "turn 9" in prompt
    assert "turn 0" not in prompt
    assert "…" in prompt


def test_stream_rag_path():
    results = [doc("4.7.19 Buffer zones shall be no less than 15 metres.", 0.7)]
    pipeline = make_rag(results)

    async def collect():
        return [json.loads(line) for line in [x async for x in pipeline.query_stream("buffer?")]]

    events = asyncio.run(collect())
    # A verification event precedes the streamed (already verified) answer text.
    assert events[0]["type"] == "verification"
    assert events[0]["verified"] is True
    text = "".join(e.get("content", "") for e in events if e.get("type") == "content")
    assert "4.7.19" in text
    assert events[-1]["done"] is True


def test_stream_compliance_below_threshold_streams_model():
    results = [doc("4.1.2 Unrelated chunk.", 0.35)]
    pipeline = make_rag(results)

    async def collect():
        return [json.loads(line) for line in [x async for x in pipeline.query_stream("is vermicompost allowed?")]]

    events = asyncio.run(collect())
    assert "weaker match" in pipeline.llm.calls[0]["prompt"]
    # compliance-shaped, below threshold -> RAG path with verification event
    assert events[0]["type"] == "verification"


def test_fabricated_citation_is_refused_not_returned():
    """The core safety property: an unverifiable citation never reaches the user."""
    results = [doc("4.1.2 Buffer zones shall be no less than 15 metres.", 0.7)]
    pipeline = make_rag(results, count_fabricated=True)
    result = asyncio.run(pipeline.query("How wide should a buffer zone be?"))
    assert result["verified"] is False
    assert result["refused"] is True
    assert "9.9.9" not in result["answer"]
    assert "4 hectares" not in result["answer"]
    assert result["unsupported_citations"] == ["9.9.9"]
    # It tried a strict retry before giving up.
    assert pipeline.llm.generate_calls == 2


def test_grounded_answer_is_returned_and_marked_verified():
    results = [doc("4.7.19 Buffer zones shall be no less than 15 metres.", 0.7)]
    pipeline = make_rag(results)
    result = asyncio.run(pipeline.query("How wide should a buffer zone be?"))
    assert result["verified"] is True
    assert result["refused"] is None
    assert result["unsupported_citations"] == []
    assert "4.7.19" in result["answer"]


def test_conversational_path_is_not_citation_checked():
    pipeline = make_rag([])
    result = asyncio.run(pipeline.query("hello"))
    assert result.get("verified") is None
    assert "G'day" in result["answer"]


def test_input_check_approved():
    results = [doc("Potassium permanganate is permitted as a sanitation treatment.", 0.62)]
    pipeline = make_rag(results, rag_answer="Potassium permanganate is permitted as a sanitation treatment.")
    result = asyncio.run(
        pipeline.check_input_approval("Potassium permanganate", ["KMnO4"], "sanitation")
    )
    assert result["approved"] is True
    assert result["status"] == "Approved"


def test_input_check_rejected():
    results = [doc("Rotenone is prohibited under organic standards.", 0.6)]
    pipeline = make_rag(results, rag_answer="Rotenone is prohibited under organic standards.")
    result = asyncio.run(
        pipeline.check_input_approval("Rotenone", ["rotenone"], "pest control")
    )
    assert result["approved"] is False
    assert result["status"] == "Not Approved"


def test_input_check_uncertain():
    results = [doc("The standards mention general pest control principles.", 0.55)]
    pipeline = make_rag(results)
    result = asyncio.run(
        pipeline.check_input_approval("Mystery product", ["x"], "crop_production")
    )
    assert result["status"] == "Uncertain - Review Required"


# ---------------------------------------------------------------------------
# Country routing
# ---------------------------------------------------------------------------

from app.data.manifest import CountryManifest, ManifestSource


def nz_manifest():
    return CountryManifest(
        country="nz",
        name="New Zealand",
        language="en",
        sources=[
            ManifestSource(id="biogro", name="BioGro NZ Standard", edition="2024"),
            ManifestSource(id="biogro", name="BioGro NZ Standard", edition="2020", status="superseded"),
        ],
    )


def test_rag_prompt_includes_country_and_active_editions():
    results = [doc("Buffer zones shall be no less than 15 metres.", 0.7)]
    pipeline = make_rag(results, country="nz", manifest=nz_manifest())
    asyncio.run(pipeline.query("How wide should a buffer zone be?"))
    prompt = pipeline.llm.calls[0]["prompt"]
    assert "New Zealand (NZ)" in prompt
    assert "BioGro NZ Standard, Edition 2024" in prompt
    assert "Edition 2020" not in prompt  # superseded edition excluded


def test_country_label_without_manifest():
    pipeline = make_rag([], country="zz")
    assert pipeline.country_label == "ZZ"
    assert "country code zz" in pipeline._standards_block()


def test_per_country_threshold_override():
    results = [doc("Marginally relevant chunk.", 0.55)]
    manifest = CountryManifest(
        country="nz", name="New Zealand", relevance_threshold=0.60, sources=[]
    )
    pipeline = make_rag(results, country="nz", manifest=manifest)
    assert pipeline.relevance_threshold == 0.60
    # 0.55 < 0.60 → below-threshold compliance still reaches the model, but
    # flagged as a weaker match (low-confidence note)
    result = asyncio.run(pipeline.query("What are the buffer zone rules?"))
    assert len(result["sources"]) == 1
    assert "weaker match" in pipeline.llm.calls[0]["prompt"]


def test_language_instruction_for_non_english():
    manifest = CountryManifest(country="de", name="Germany", language="de", sources=[])
    pipeline = make_rag([], country="de", manifest=manifest)
    assert "Answer in de." in pipeline._language_instruction()
    assert "Answer in en." not in pipeline._language_instruction()


def test_default_threshold_used_without_manifest():
    pipeline = make_rag([])
    assert pipeline.relevance_threshold == pipeline_mod.RELEVANCE_THRESHOLD
