from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import json
import os
import time
import logging
import collections

from app.schemas import (
    QueryRequest,
    QueryResponse,
    InputCheckRequest,
    InputCheckResponse,
)
from app.rag.pipeline import RAGPipeline
from app.config import DATA_DIR, CORS_ORIGINS, LOG_LEVEL, API_TOKEN, RATE_LIMIT_PER_MINUTE, DEFAULT_COUNTRY
from app.data.manifest import installed_countries, load_manifest

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("orgstandardsbot")

_rag: Optional[RAGPipeline] = None
_rags: dict = {}


def get_rag(country: str = DEFAULT_COUNTRY) -> RAGPipeline:
    global _rag
    country = country.lower()
    if country == DEFAULT_COUNTRY and _rag is not None:
        return _rag
    pipeline = _rags.get(country)
    if pipeline is None:
        logger.info("Initializing RAG pipeline for country '%s'...", country)
        pipeline = RAGPipeline(country=country)
        _rags[country] = pipeline
        if country == DEFAULT_COUNTRY:
            _rag = pipeline
        logger.info("RAG pipeline ready (%s)!", country)
    return pipeline


def _known_country(country: str) -> str:
    """Validate + normalize a country code against installed manifests."""
    country = country.lower()
    if country not in installed_countries(DATA_DIR):
        raise HTTPException(
            status_code=404,
            detail=f"No standards installed for country '{country}'. "
                   "Run scripts/add_country.py to add one.",
        )
    return country


def _source_redistribution(country: str, source: str) -> tuple[bool, str]:
    """Return (redistributable, licence_label) for a source id.

    A source is redistributable only when EVERY manifest entry with that id
    declares ``redistribute: true`` (missing/False → not redistributable).
    Unknown source ids raise 404.
    """
    manifest = load_manifest(DATA_DIR, country)
    entries = [s for s in manifest.sources if s.id == source]
    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"Source '{source}' not found for country '{country}'.",
        )
    allowed = all(s.redistribute for s in entries)
    licenses = sorted({s.license or "not declared" for s in entries})
    return allowed, ", ".join(licenses)


def _require_redistributable(country: str, source: str) -> None:
    """403 unless this source may be served in full (PDF / HTML)."""
    allowed, licences = _source_redistribution(country, source)
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Full copies of '{source}' are not distributed by this deployment "
                f"(licence: {licences}). Retrieval and citation still work; contact "
                "the standards body for the full document."
            ),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_rag(DEFAULT_COUNTRY)
    yield


app = FastAPI(
    title="OrgStandardsBot",
    description="AI assistant for Australian organic certification compliance — open source",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Security: optional API token + per-IP rate limiting
# ---------------------------------------------------------------------------

# Sliding window: ip -> deque of request timestamps
_rate_window: "collections.defaultdict[str, collections.deque]" = collections.defaultdict(
    lambda: collections.deque(maxlen=256)
)
_RATE_WINDOW_SECONDS = 60.0


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    ip = _client_ip(request)
    now = time.monotonic()
    window = _rate_window[ip]
    while window and now - window[0] > _RATE_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded — try again shortly")
    window.append(now)


async def _require_auth(request: Request) -> None:
    if not API_TOKEN:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


async def _guard(request: Request) -> None:
    await _require_auth(request)
    _check_rate_limit(request)


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/status")
async def status():
    rag = get_rag()
    doc_count = rag.vector_store.get_document_count() if rag.vector_store else 0
    llm_healthy = await rag.llm.check_health()

    countries = []
    for code in installed_countries(DATA_DIR):
        try:
            manifest = load_manifest(DATA_DIR, code)
            countries.append(
                {
                    "code": code,
                    "name": manifest.name,
                    "language": manifest.language,
                    "standards": [
                        {
                            "id": s.id,
                            "name": s.name,
                            "edition": s.edition,
                            "license": s.license,
                            "redistributable": s.redistribute,
                        }
                        for s in manifest.active_sources
                    ],
                }
            )
        except Exception as e:
            logger.warning("Skipping country '%s' in status: %s", code, e)

    return {
        "documents_loaded": doc_count,
        "llm_connected": llm_healthy,
        "model": rag.llm.model,
        "countries": countries,
        "default_country": DEFAULT_COUNTRY,
    }


# ---------------------------------------------------------------------------
# Chat / Query
# ---------------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, req: Request):
    await _guard(req)
    country = _known_country(request.country)
    rag = get_rag(country)
    result = await rag.query(
        question=request.question, top_k=request.top_k, history=request.history
    )
    result["country"] = country
    return QueryResponse(**result)


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest, req: Request):
    await _guard(req)
    country = _known_country(request.country)
    rag = get_rag(country)

    async def generate():
        try:
            async for chunk in rag.query_stream(
                request.question, request.top_k, history=request.history
            ):
                yield chunk
            yield json.dumps({"done": True}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/input-check", response_model=InputCheckResponse)
async def check_input(request: InputCheckRequest, req: Request):
    await _guard(req)
    country = _known_country(request.country)
    rag = get_rag(country)
    result = await rag.check_input_approval(
        product_name=request.product_name,
        ingredients=request.ingredients,
        intended_use=request.intended_use,
    )
    result["country"] = country
    return InputCheckResponse(**result)


# ---------------------------------------------------------------------------
# Sources & Documents
# ---------------------------------------------------------------------------


@app.get("/api/sources")
async def list_sources(country: str = DEFAULT_COUNTRY):
    country = _known_country(country)
    rag = get_rag(country)
    collection = rag.vector_store.collection
    all_data = collection.get(include=["metadatas"])

    sources = {}
    for metadata in all_data.get("metadatas", []):
        source = metadata.get("source", "unknown")
        filename = metadata.get("filename", "unknown")
        if source not in sources:
            sources[source] = []
        if filename not in sources[source]:
            sources[source].append(filename)

    return {"country": country, "sources": sources}


@app.get("/api/documents")
async def list_documents(country: str = DEFAULT_COUNTRY):
    """List all available documents with metadata"""
    country = _known_country(country)
    rag = get_rag(country)
    collection = rag.vector_store.collection
    all_data = collection.get(include=["metadatas"])

    # Per-source licence + redistribution flags from the manifest
    manifest = load_manifest(DATA_DIR, country)
    licence_map = {s.id: s for s in manifest.sources}

    documents = {}
    for metadata in all_data.get("metadatas", []):
        source = metadata.get("source", "unknown")
        filename = metadata.get("filename", "unknown")
        key = f"{source}/{filename}"

        if key not in documents:
            src = licence_map.get(source)
            documents[key] = {
                "source": source,
                "filename": filename,
                "edition": metadata.get("edition", ""),
                "total_pages": metadata.get("total_pages", 0),
                "total_chunks": 0,
                "html_available": False,
                "redistributable": bool(src and src.redistribute),
                "license": (src.license if src else None),
            }

        documents[key]["total_chunks"] += 1

        if documents[key]["redistributable"]:
            html_path = metadata.get("html_path")
            if html_path and os.path.exists(html_path):
                documents[key]["html_available"] = True

    return {"country": country, "documents": list(documents.values())}


@app.get("/api/documents/{country}/{source}/html")
async def get_document_html(country: str, source: str, req: Request):
    """Serve the full HTML version of a document (redistributable sources only)"""
    await _guard(req)
    country = _known_country(country)
    _require_redistributable(country, source)
    html_path = DATA_DIR / country / source / "html" / "full.html"

    if not html_path.exists():
        raise HTTPException(
            status_code=404, detail=f"HTML document not found for source: {country}/{source}"
        )

    return FileResponse(html_path, media_type="text/html")


@app.get("/api/documents/{country}/{source}/page/{page}")
async def get_document_page(country: str, source: str, page: int, req: Request):
    """Get a specific page of a document (redistributable sources only)"""
    await _guard(req)
    country = _known_country(country)
    _require_redistributable(country, source)
    page_path = DATA_DIR / country / source / "html" / f"page-{page}.html"

    if not page_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Page {page} not found for source: {country}/{source}"
        )

    return FileResponse(page_path, media_type="text/html")


class ChunkContextResponse(BaseModel):
    chunk_text: str
    page_number: int
    char_start: int
    char_end: int
    source: str
    filename: str
    total_pages: int
    html_available: bool
    redistributable: bool


@app.get("/api/chunk/context")
async def get_chunk_context(
    source: str,
    filename: str,
    page_number: int,
    char_start: int,
    char_end: int,
    country: str = DEFAULT_COUNTRY,
):
    """Get chunk context for highlighting in document viewer"""
    country = _known_country(country)
    rag = get_rag(country)
    collection = rag.vector_store.collection

    redistributable, _ = _source_redistribution(country, source)
    html_path = DATA_DIR / country / source / "html" / "full.html"
    html_available = redistributable and html_path.exists()

    total_pages = 0
    all_data = collection.get(include=["metadatas"])
    for metadata in all_data.get("metadatas", []):
        if metadata.get("source") == source and metadata.get("filename") == filename:
            total_pages = metadata.get("total_pages", 0)
            break

    return ChunkContextResponse(
        chunk_text="",
        page_number=page_number,
        char_start=char_start,
        char_end=char_end,
        source=source,
        filename=filename,
        total_pages=total_pages,
        html_available=html_available,
        redistributable=redistributable,
    )


@app.get("/api/documents/{country}/{source}/pdf")
async def get_document_pdf(country: str, source: str, req: Request):
    """Serve the original PDF file (redistributable sources only)"""
    await _guard(req)
    country = _known_country(country)
    _require_redistributable(country, source)
    source_dir = DATA_DIR / country / source

    if not source_dir.exists():
        raise HTTPException(status_code=404, detail=f"Source not found: {country}/{source}")

    pdf_files = list(source_dir.glob("*.pdf"))

    if not pdf_files:
        raise HTTPException(
            status_code=404, detail=f"No PDF found for source: {source}"
        )

    return FileResponse(pdf_files[0], media_type="application/pdf")
