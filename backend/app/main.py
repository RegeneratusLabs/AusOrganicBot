from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import json
import os
import logging

from app.schemas import (
    QueryRequest,
    QueryResponse,
    InputCheckRequest,
    InputCheckResponse,
)
from app.rag.pipeline import RAGPipeline
from app.config import DATA_DIR, CORS_ORIGINS, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ausorganicbot")

_rag: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        logger.info("Initializing RAG pipeline with DeepSeek...")
        _rag = RAGPipeline()
        logger.info("RAG pipeline ready!")
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_rag()
    yield


app = FastAPI(
    title="AusOrganicBot",
    description="AI assistant for Australian organic certification compliance — open source",
    version="2.0.0",
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

    return {
        "documents_loaded": doc_count,
        "llm_connected": llm_healthy,
        "model": rag.llm.model,
    }


# ---------------------------------------------------------------------------
# Chat / Query
# ---------------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    rag = get_rag()
    result = await rag.query(question=request.question, top_k=request.top_k)
    return QueryResponse(**result)


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    rag = get_rag()

    async def generate():
        try:
            async for chunk in rag.query_stream(request.question, request.top_k):
                yield chunk
            yield json.dumps({"done": True}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/input-check", response_model=InputCheckResponse)
async def check_input(request: InputCheckRequest):
    rag = get_rag()
    result = await rag.check_input_approval(
        product_name=request.product_name,
        ingredients=request.ingredients,
        intended_use=request.intended_use,
    )
    return InputCheckResponse(**result)


# ---------------------------------------------------------------------------
# Sources & Documents
# ---------------------------------------------------------------------------


@app.get("/api/sources")
async def list_sources():
    rag = get_rag()
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

    return {"sources": sources}


@app.get("/api/documents")
async def list_documents():
    """List all available documents with metadata"""
    rag = get_rag()
    collection = rag.vector_store.collection
    all_data = collection.get(include=["metadatas"])

    documents = {}
    for metadata in all_data.get("metadatas", []):
        source = metadata.get("source", "unknown")
        filename = metadata.get("filename", "unknown")
        key = f"{source}/{filename}"

        if key not in documents:
            documents[key] = {
                "source": source,
                "filename": filename,
                "total_pages": metadata.get("total_pages", 0),
                "total_chunks": 0,
                "html_available": False,
            }

        documents[key]["total_chunks"] += 1

        html_path = metadata.get("html_path")
        if html_path and os.path.exists(html_path):
            documents[key]["html_available"] = True

    return {"documents": list(documents.values())}


@app.get("/api/documents/{source}/html")
async def get_document_html(source: str):
    """Serve the full HTML version of a document"""
    html_path = DATA_DIR / source / "html" / "full.html"

    if not html_path.exists():
        raise HTTPException(
            status_code=404, detail=f"HTML document not found for source: {source}"
        )

    return FileResponse(html_path, media_type="text/html")


@app.get("/api/documents/{source}/page/{page}")
async def get_document_page(source: str, page: int):
    """Get a specific page of a document"""
    page_path = DATA_DIR / source / "html" / f"page-{page}.html"

    if not page_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Page {page} not found for source: {source}"
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


@app.get("/api/chunk/context")
async def get_chunk_context(
    source: str, filename: str, page_number: int, char_start: int, char_end: int
):
    """Get chunk context for highlighting in document viewer"""
    rag = get_rag()
    collection = rag.vector_store.collection

    html_path = DATA_DIR / source / "html" / "full.html"
    html_available = html_path.exists()

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
    )


@app.get("/api/documents/{source}/pdf")
async def get_document_pdf(source: str):
    """Serve the original PDF file"""
    source_dir = DATA_DIR / source

    if not source_dir.exists():
        raise HTTPException(status_code=404, detail=f"Source not found: {source}")

    pdf_files = list(source_dir.glob("*.pdf"))

    if not pdf_files:
        raise HTTPException(
            status_code=404, detail=f"No PDF found for source: {source}"
        )

    return FileResponse(pdf_files[0], media_type="application/pdf")
