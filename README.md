# PDF Intelligence — RAG Summarisation & Q&A System

A production-ready AI system that ingests PDF documents, builds a semantic vector index, and answers questions using Retrieval-Augmented Generation (RAG) powered by OpenAI GPT-4o.

## Architecture

```
PDF Upload → Text Extraction (PyMuPDF)
          → Chunking (word-level, overlapping)
          → Embeddings (sentence-transformers all-MiniLM-L6-v2)
          → Vector Store (FAISS, persisted to disk)

Question  → Embed query
          → FAISS similarity search → top-k chunks
          → OpenAI GPT-4o (streaming)
          → Grounded answer with source citations
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~90 MB). Subsequent starts are instant.

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

```bash
python run.py
# or: uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a PDF (multipart/form-data) |
| `GET`  | `/api/documents` | List all indexed documents |
| `GET`  | `/api/documents/{doc_id}` | Get document metadata |
| `DELETE` | `/api/documents/{doc_id}` | Delete document + vectors |
| `POST` | `/api/query` | Stream Q&A response (SSE) |
| `POST` | `/api/summarise` | Stream document summary (SSE) |
| `GET`  | `/health` | Health check |

### Example: Q&A request

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology was used in the study?", "doc_id": "abc123"}'
```

Response is a Server-Sent Events stream:
```
data: {"type": "text", "content": "The study used..."}
data: {"type": "sources", "sources": [...]}
data: {"type": "done"}
```

## Features

- **Multi-document support** — upload and query across multiple PDFs
- **Scoped search** — optionally restrict Q&A to a single document
- **Streaming responses** — answers stream token-by-token via SSE
- **Source citations** — every answer shows which pages were used
- **Persistent index** — FAISS index survives server restarts
- **Document summarisation** — full document summaries for large PDFs
- **Production error handling** — proper HTTP status codes, typed errors
- **Clean UI** — dark-mode single-page app with drag-and-drop upload

## Project Structure

```
summarizer/
├── app/
│   ├── main.py          # FastAPI app + all endpoints
│   ├── config.py        # Settings via pydantic-settings
│   ├── pdf_processor.py # PDF extraction + chunking
│   ├── embedder.py      # sentence-transformers wrapper
│   ├── vector_store.py  # FAISS vector store (persisted)
│   └── rag.py           # RAG pipeline + OpenAI streaming
├── static/
│   └── index.html       # Frontend SPA
├── uploads/             # Uploaded PDFs (auto-created)
├── vector_db/           # FAISS index + metadata (auto-created)
├── data/                # Document registry JSON (auto-created)
├── requirements.txt
├── .env.example
└── run.py
```

## Configuration

All settings can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model to use |
| `PORT` | `8000` | Server port |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max PDF size |
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `60` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |

## Future Improvements

### Summarisation Quality
- **Map-Reduce summarisation** — split large documents into sections, summarise each independently, then synthesise a final summary. This removes the current 90K character truncation limit and handles arbitrarily long PDFs without losing content from later pages.
- **Hierarchical chunking** — maintain both fine-grained chunks (for precise Q&A retrieval) and coarser section-level chunks (for summarisation), so each task uses the most appropriate granularity.

### Retrieval & Accuracy
- **Re-ranking** — after FAISS retrieval, apply a cross-encoder model (e.g. `ms-marco-MiniLM`) to re-score and reorder the top-k chunks by relevance before sending to the LLM. This significantly improves answer quality for complex questions.
- **Hybrid search** — combine dense vector search (FAISS) with sparse keyword search (BM25) and merge results. Keyword search catches exact-match terms (names, codes, dates) that embeddings sometimes miss.
- **Metadata filtering** — allow filtering by page range, upload date, or custom tags before running similarity search.

### Scalability
- **Swap FAISS for a managed vector DB** (e.g. Pinecone, Qdrant, Weaviate) to support millions of documents, horizontal scaling, and built-in persistence without managing index files manually.
- **Async PDF processing** — offload PDF extraction and embedding to a background task queue (Celery + Redis) so large uploads don't block the HTTP response.
- **Caching** — cache embeddings for repeated queries and cache summaries per document so re-summarising the same PDF is instant.

### User Experience
- **Conversation memory** — maintain per-session chat history so follow-up questions ("what about the methodology?") are resolved in context rather than treated as standalone queries.
- **Cross-document comparison** — structured endpoint to compare two documents side-by-side (e.g. "How do these two reports differ on climate risk?").
- **Export** — allow downloading Q&A transcripts and summaries as PDF or Markdown.

### Production Hardening
- **Authentication** — add API key or OAuth2 guard on all `/api/*` endpoints before any public deployment.
- **Rate limiting** — per-IP or per-user limits on upload and query endpoints to prevent abuse.
- **Structured logging + tracing** — emit structured JSON logs with a request ID threaded through upload → embed → retrieve → generate, making it easy to trace latency bottlenecks in production.