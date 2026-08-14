"""API-level tests: health, auth guard, rate limiting, input validation."""
import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as main_mod
from app.config import RATE_LIMIT_PER_MINUTE


class FakeRAG:
    def __init__(self):
        self.vector_store = None
        self.llm = FakeLLM()

    async def query(self, question, top_k=5):
        return {
            "answer": "answer",
            "reasoning": "",
            "sources": [],
            "confidence": 0.9,
            "model": "fake",
            "provider": "fake",
            "processing_time_ms": 1.0,
        }


class FakeLLM:
    model = "fake"
    provider = "fake"

    async def check_health(self):
        return True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_mod, "get_rag", lambda country="au": FakeRAG())
    monkeypatch.setattr(main_mod, "installed_countries", lambda data_dir: ["au"])
    monkeypatch.setattr(
        main_mod,
        "load_manifest",
        lambda data_dir, code: __import__("app.data.manifest", fromlist=["CountryManifest"])
        .CountryManifest(country=code, name="Australia"),
    )
    with TestClient(main_mod.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_status(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_connected"] is True
    assert body["countries"][0]["code"] == "au"


def test_query_endpoint(client):
    resp = client.post("/api/query", json={"question": "buffer zone?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "answer"
    assert resp.json()["country"] == "au"


def test_query_unknown_country_returns_404(client, monkeypatch):
    monkeypatch.setattr(main_mod, "installed_countries", lambda data_dir: ["au"])
    resp = client.post("/api/query", json={"question": "buffer?", "country": "zz"})
    assert resp.status_code == 404


def test_query_invalid_country_code_rejected(client):
    resp = client.post("/api/query", json={"question": "buffer?", "country": "AUS"})
    assert resp.status_code == 422


def test_question_too_long_rejected(client):
    resp = client.post("/api/query", json={"question": "x" * 600})
    assert resp.status_code == 422


def test_top_k_capped(client):
    resp = client.post("/api/query", json={"question": "buffer?", "top_k": 999})
    assert resp.status_code == 422


def test_auth_required_when_token_set(monkeypatch):
    monkeypatch.setattr(main_mod, "API_TOKEN", "sekret")
    monkeypatch.setattr(main_mod, "get_rag", lambda country="au": FakeRAG())
    with TestClient(main_mod.app) as c:
        assert c.post("/api/query", json={"question": "q"}).status_code == 401
        ok = c.post(
            "/api/query",
            json={"question": "q"},
            headers={"Authorization": "Bearer sekret"},
        )
        assert ok.status_code == 200
        assert c.post(
            "/api/query", json={"question": "q"}, headers={"Authorization": "Bearer wrong"}
        ).status_code == 401


def test_rate_limit_enforced(monkeypatch):
    monkeypatch.setattr(main_mod, "RATE_LIMIT_PER_MINUTE", 3)
    monkeypatch.setattr(main_mod, "get_rag", lambda country="au": FakeRAG())
    monkeypatch.setattr(main_mod, "_rate_window", __import__("collections").defaultdict(
        lambda: __import__("collections").deque(maxlen=256)
    ))
    with TestClient(main_mod.app) as c:
        codes = [c.post("/api/query", json={"question": "q"}).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
