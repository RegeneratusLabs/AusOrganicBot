"""Tests for the OpenAI-compatible LLM client using a mocked transport."""
import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import llm_client as llm_mod
from app.llm.llm_client import LLMClient


def make_client(handler, base_url="https://api.openai.com/v1"):
    client = LLMClient.__new__(LLMClient)
    client.api_key = "test-key"
    client.base_url = base_url.rstrip("/")
    client.model = "test-model"
    client.provider = llm_mod._infer_provider(base_url)
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=10.0)
    return client


def test_generate_returns_content_and_reasoning():
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        body = __import__("json").loads(request.read())
        assert body["model"] == "test-model"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "the answer", "reasoning_content": "the thinking"}}]},
        )

    client = make_client(handler)
    result = asyncio.run(client.generate("q"))
    assert result == {"content": "the answer", "reasoning": "the thinking"}


def test_generate_raises_on_http_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    client = make_client(handler)
    with pytest.raises(RuntimeError, match="LLM API error"):
        asyncio.run(client.generate("q"))


def test_generate_stream_parses_sse():
    lines = [
        'data: {"choices": [{"delta": {"reasoning_content": "think"}}]}',
        'data: {"choices": [{"delta": {"content": "hello"}}]}',
        'data: {"choices": [{"delta": {"content": " world"}}]}',
        "data: [DONE]",
    ]

    def handler(request):
        return httpx.Response(200, text="\n".join(lines) + "\n")

    client = make_client(handler)
    events = asyncio.run(_collect_stream(client))
    assert events == [
        {"type": "reasoning", "chunk": "think"},
        {"type": "content", "chunk": "hello"},
        {"type": "content", "chunk": " world"},
    ]


async def _collect_stream(client):
    return [evt async for evt in client.generate_stream("q")]


def test_thinking_param_gated_for_non_deepseek(monkeypatch):
    monkeypatch.setattr(llm_mod, "THINKING_PARAM_MODE", "auto")
    client = make_client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}))
    payload = client._get_thinking_param()
    assert payload == {}  # openai provider: no thinking param


def test_thinking_param_sent_for_deepseek(monkeypatch):
    monkeypatch.setattr(llm_mod, "THINKING_PARAM_MODE", "auto")
    monkeypatch.setattr(llm_mod, "THINKING_ENABLED", False)
    client = make_client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}),
                         base_url="https://api.deepseek.com/v1")
    assert client._get_thinking_param() == {"thinking": {"type": "disabled"}}


def test_health_true(monkeypatch):
    client = make_client(lambda r: httpx.Response(200, json={"data": []}))
    assert asyncio.run(client.check_health()) is True


def test_health_false_without_key():
    client = make_client(lambda r: httpx.Response(200, json={}))
    client.api_key = ""
    assert asyncio.run(client.check_health()) is False
