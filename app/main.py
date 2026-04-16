"""FastAPI application — PDF RAG system."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.pdf_processor import process_pdf
from app.rag import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF Summarisation & QA System",
    description="Upload PDFs, summarise them, and ask questions using RAG + Claude.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
_STATIC_DIR = Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Globals (lazy-initialised on first request to avoid startup delay)
# ---------------------------------------------------------------------------

_rag_engine: RAGEngine | None = None
_REGISTRY_PATH = settings.data_dir / "documents.json"


def _get_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        logger.info("Initialising RAG engine …")
        _rag_engine = RAGEngine()
    return _rag_engine


def _load_registry() -> dict:
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict) -> None:
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    doc_id: str | None = None  # scope to a single document (optional)


class SummariseRequest(BaseModel):
    doc_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    index_file = Path(__file__).parent.parent / "static" / "index.html"
    if index_file.exists():
        async with aiofiles.open(index_file, encoding="utf-8") as f:
            return await f.read()
    return "<h1>PDF RAG System is running</h1><p>Place index.html in the static/ directory.</p>"


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ------------------------------------------------------------------
# Document management
# ------------------------------------------------------------------

@app.post("/api/upload", status_code=201)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it for Q&A."""
    # Validate
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Save to disk
    dest = settings.uploads_dir / f"{uuid.uuid4().hex}_{file.filename}"
    size = 0
    async with aiofiles.open(dest, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB read chunks
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                await out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.",
                )
            await out.write(chunk)

    # Process PDF
    try:
        result = process_pdf(
            dest,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("PDF processing failed")
        raise HTTPException(status_code=422, detail=f"Could not process PDF: {exc}") from exc

    doc_id = result["doc_id"]

    # Ingest into vector store
    engine = _get_engine()
    engine.ingest_chunks(result["chunks"])

    # Register document
    registry = _load_registry()
    registry[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "upload_time": datetime.now(timezone.utc).isoformat(),
        "page_count": result["total_pages"],
        "chunk_count": len(result["chunks"]),
        "size_bytes": size,
        "file_path": str(dest),
        # Store text for summarisation (capped at 400 KB)
        "full_text": result["full_text"][:400_000],
    }
    _save_registry(registry)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "page_count": result["total_pages"],
        "chunk_count": len(result["chunks"]),
        "size_bytes": size,
        "message": "PDF uploaded and indexed successfully.",
    }


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    registry = _load_registry()
    docs = [
        {k: v for k, v in doc.items() if k != "full_text"}  # omit large text from listing
        for doc in registry.values()
    ]
    docs.sort(key=lambda d: d.get("upload_time", ""), reverse=True)
    return {"documents": docs, "total": len(docs)}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get metadata for a single document."""
    registry = _load_registry()
    if doc_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found.")
    doc = {k: v for k, v in registry[doc_id].items() if k != "full_text"}
    return doc


@app.delete("/api/documents/{doc_id}", status_code=200)
async def delete_document(doc_id: str):
    """Delete a document and remove its vectors from the store."""
    registry = _load_registry()
    if doc_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove from vector store
    engine = _get_engine()
    removed = engine.delete_document(doc_id)

    # Remove uploaded file
    file_path = Path(registry[doc_id].get("file_path", ""))
    if file_path.exists():
        file_path.unlink()

    # Remove from registry
    del registry[doc_id]
    _save_registry(registry)

    return {"message": "Document deleted.", "vectors_removed": removed}


# ------------------------------------------------------------------
# Q&A — streaming
# ------------------------------------------------------------------

@app.post("/api/query")
async def query(request: QueryRequest):
    """
    Ask a question about the uploaded documents.
    Returns a Server-Sent Events stream.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Validate doc_id if provided
    if request.doc_id:
        registry = _load_registry()
        if request.doc_id not in registry:
            raise HTTPException(status_code=404, detail="Document not found.")

    engine = _get_engine()

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event_str in _wrap_async(
            engine.answer_stream(question, doc_id=request.doc_id)
        ):
            yield f"data: {event_str}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------------------------------------------------------
# Summarisation — streaming
# ------------------------------------------------------------------

@app.post("/api/summarise")
async def summarise(request: SummariseRequest):
    """
    Summarise an uploaded document.
    Returns a Server-Sent Events stream.
    """
    registry = _load_registry()
    if request.doc_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc = registry[request.doc_id]
    full_text = doc.get("full_text", "")
    filename = doc.get("filename", "document")

    if not full_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from this document.")

    engine = _get_engine()

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event_str in _wrap_async(
            engine.summarise_stream(full_text, filename=filename)
        ):
            yield f"data: {event_str}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Utility: run a synchronous generator off the event loop thread
# ---------------------------------------------------------------------------

async def _wrap_async(sync_gen) -> AsyncGenerator[str, None]:
    """
    Drive a synchronous generator in a thread-pool executor so the event loop
    is never blocked.  Items are passed back via an asyncio.Queue.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | object] = asyncio.Queue(maxsize=32)
    _SENTINEL = object()

    def _producer() -> None:
        try:
            for item in sync_gen:
                # put_nowait may block if the queue is full; use the blocking put instead
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(f'{{"type":"error","message":{str(exc)!r}}}'), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

    # Run producer in a thread; don't await — it feeds the queue while we consume
    loop.run_in_executor(None, _producer)

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        yield item  # type: ignore[misc]
