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

    async def query(self, question, top_k=5, history=None):
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


# ---------------------------------------------------------------------------
# Document redistribution gating (licence flags in country manifests)
# ---------------------------------------------------------------------------

from app.data.manifest import CountryManifest, ManifestSource  # noqa: E402


def _make_manifest(redistribute_map: dict):
    """Build a CountryManifest with per-id redistribute flags."""
    sources = []
    for sid, (licence, flag) in redistribute_map.items():
        sources.append(
            ManifestSource(
                id=sid,
                name=f"{sid} Standard",
                edition="2023",
                license=licence,
                redistribute=flag,
                local_path=f"{sid}/std.pdf",
            )
        )
    return CountryManifest(country="au", name="Australia", sources=sources)


def _patch_doc_env(monkeypatch, tmp_path, manifest):
    """Patch the app so document endpoints don't boot the real RAG pipeline
    (which would load the embedding model from disk/network)."""
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main_mod, "installed_countries", lambda data_dir: ["au"])
    monkeypatch.setattr(main_mod, "load_manifest", lambda data_dir, code: manifest)
    monkeypatch.setattr(main_mod, "get_rag", lambda country="au": FakeRAG())


def test_document_endpoints_403_for_non_redistributable(monkeypatch, tmp_path):
    """ACO-style (all rights reserved) source: PDF/HTML/page all return 403."""
    manifest = _make_manifest({"aco": ("All rights reserved", False)})
    base = tmp_path / "au"
    (base / "aco" / "html").mkdir(parents=True)
    (base / "aco" / "html" / "full.html").write_text("<html>ACO</html>")
    (base / "aco" / "html" / "page-1.html").write_text("<html>p1</html>")
    (base / "aco" / "std.pdf").write_bytes(b"%PDF-1.4 aco")

    _patch_doc_env(monkeypatch, tmp_path, manifest)

    with TestClient(main_mod.app) as c:
        assert c.get("/api/documents/au/aco/pdf").status_code == 403
        assert c.get("/api/documents/au/aco/html").status_code == 403
        assert c.get("/api/documents/au/aco/page/1").status_code == 403


def test_document_endpoints_200_for_redistributable(monkeypatch, tmp_path):
    """National-Standard-style (CC-BY) source: full copies are served."""
    manifest = _make_manifest({"national": ("CC-BY-4.0", True)})
    base = tmp_path / "au"
    (base / "national" / "html").mkdir(parents=True)
    (base / "national" / "html" / "full.html").write_text("<html>National</html>")
    (base / "national" / "html" / "page-1.html").write_text("<html>p1</html>")
    (base / "national" / "std.pdf").write_bytes(b"%PDF-1.4 national")

    _patch_doc_env(monkeypatch, tmp_path, manifest)

    with TestClient(main_mod.app) as c:
        assert c.get("/api/documents/au/national/pdf").status_code == 200
        assert c.get("/api/documents/au/national/html").status_code == 200
        assert c.get("/api/documents/au/national/page/1").status_code == 200
        pdf = c.get("/api/documents/au/national/pdf")
        assert pdf.content == b"%PDF-1.4 national"


def test_document_unknown_source_404(monkeypatch, tmp_path):
    manifest = _make_manifest({"aco": ("All rights reserved", False)})
    _patch_doc_env(monkeypatch, tmp_path, manifest)

    with TestClient(main_mod.app) as c:
        assert c.get("/api/documents/au/ghost/pdf").status_code == 404


def test_status_exposes_license_and_redistributable(monkeypatch):
    manifest = _make_manifest(
        {"national": ("CC-BY-4.0", True), "aco": ("All rights reserved", False)}
    )
    monkeypatch.setattr(main_mod, "DATA_DIR", Path("/nonexistent"))
    monkeypatch.setattr(main_mod, "installed_countries", lambda data_dir: ["au"])
    monkeypatch.setattr(main_mod, "load_manifest", lambda data_dir, code: manifest)
    monkeypatch.setattr(main_mod, "get_rag", lambda country="au": FakeRAG())

    with TestClient(main_mod.app) as c:
        body = c.get("/api/status").json()
        standards = {s["id"]: s for s in body["countries"][0]["standards"]}
        assert standards["national"]["license"] == "CC-BY-4.0"
        assert standards["national"]["redistributable"] is True
        assert standards["aco"]["redistributable"] is False
