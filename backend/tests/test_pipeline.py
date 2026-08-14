"""Tests for the RAG pipeline: relevance gate, refusal guard, streaming, input check."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import pipeline as pipeline_mod
from app.rag.pipeline import RAGPipeline, NOT_COVERED_ANSWER


class FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k=5):
        return list(self._results)


class FakeLLM:
    def __init__(self, rag_answer="Section 4.7.19 states the buffer must be no less than 15 metres."):
        self.model = "fake-model"
        self.provider = "fake"
        self.calls = []
        self.rag_answer = rag_answer

    async def generate(self, prompt, temperature=0.1, max_tokens=2048):
        self.calls.append({"type": "generate", "prompt": prompt, "temperature": temperature})
        if "Relevant excerpts" in prompt:
            return {"content": self.rag_answer, "reasoning": ""}
        return {"content": "G'day! How can I help?", "reasoning": ""}

    async def generate_stream(self, prompt, temperature=0.1, max_tokens=2048):
        self.calls.append({"type": "stream", "prompt": prompt})
        yield {"type": "content", "chunk": "streamed answer"}

    async def check_health(self):
        return True


def make_rag(results, rag_answer=None, country="au", manifest=None, threshold=None):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.vector_store = FakeVectorStore(results)
    pipeline.llm = FakeLLM(rag_answer) if rag_answer else FakeLLM()
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


def test_below_threshold_conversational(monkeypatch):
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


def test_compliance_below_threshold_is_refused():
    """Hallucination guard: compliance-shaped question under threshold must refuse."""
    results = [doc("Unrelated chunk.", 0.4)]
    pipeline = make_rag(results)
    result = asyncio.run(pipeline.query("Do I need a transition plan for certification?"))
    assert result["answer"] == NOT_COVERED_ANSWER
    assert result["refused"] is True
    assert result["sources"] == []
    assert pipeline.llm.calls == []  # no LLM call, no hallucination


def test_refusal_keyword_variants():
    assert RAGPipeline._looks_like_compliance_question("Can I use vermicompost?")
    assert RAGPipeline._looks_like_compliance_question("is it allowed to spray neem oil?")
    assert RAGPipeline._looks_like_compliance_question("What are the buffer zone rules?")
    assert not RAGPipeline._looks_like_compliance_question("hello")
    assert not RAGPipeline._looks_like_compliance_question("thanks for your help")


def test_stream_rag_path():
    results = [doc("Buffer zones shall be no less than 15 metres.", 0.7)]
    pipeline = make_rag(results)

    async def collect():
        return [json.loads(line) for line in [x async for x in pipeline.query_stream("buffer?")]]

    events = asyncio.run(collect())
    assert any(e["type"] == "content" and e["content"] == "streamed answer" for e in events)


def test_stream_refused_no_llm():
    results = [doc("Unrelated.", 0.4)]
    pipeline = make_rag(results)

    async def collect():
        return [json.loads(line) for line in [x async for x in pipeline.query_stream("is vermicompost allowed?")]]

    events = asyncio.run(collect())
    assert events[0]["content"] == NOT_COVERED_ANSWER
    assert events[-1] == {"done": True}
    assert pipeline.llm.calls == []


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
    # 0.55 < 0.60 → refusal path (question contains compliance keyword)
    result = asyncio.run(pipeline.query("What are the buffer zone rules?"))
    assert result["answer"] == NOT_COVERED_ANSWER


def test_language_instruction_for_non_english():
    manifest = CountryManifest(country="de", name="Germany", language="de", sources=[])
    pipeline = make_rag([], country="de", manifest=manifest)
    assert "Answer in de." in pipeline._language_instruction()
    assert "Answer in en." not in pipeline._language_instruction()


def test_default_threshold_used_without_manifest():
    pipeline = make_rag([])
    assert pipeline.relevance_threshold == pipeline_mod.RELEVANCE_THRESHOLD
