"""RAG pipeline: retrieval + OpenAI generation."""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Iterator

from openai import OpenAI, APIError

from app.config import settings
from app.embedder import Embedder
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_QA = """\
You are an expert document analyst. You are given excerpts from a PDF document and a question.

Rules:
1. Answer ONLY based on the provided document excerpts.
2. If the answer cannot be found in the excerpts, say so clearly — do not guess.
3. Be precise and cite the relevant page(s) when possible (e.g., "According to page 3 …").
4. Use clear, structured formatting: bullet points, numbered lists, or short paragraphs as appropriate.
5. If the question asks for multiple things (summary + methodology, etc.), address each part separately with a clear label.
"""

_SYSTEM_SUMMARY = """\
You are an expert document summariser. You are given the text of a PDF document.

Produce a comprehensive summary that covers:
- **Overview**: What the document is about (1–2 sentences).
- **Key Topics**: Main subjects, findings, or arguments discussed.
- **Methodology** (if applicable): How the study or analysis was conducted.
- **Key Findings / Conclusions**: The most important outcomes or takeaways.
- **Structure**: Brief description of how the document is organised.

Be concise yet thorough. Use markdown headings and bullet points for readability.
"""

# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page_ref = f"[Page {chunk.get('page_num', '?')}]"
        parts.append(f"--- Excerpt {i} {page_ref} ---\n{chunk['text']}\n")
    return "\n".join(parts)


def _truncate(text: str, max_chars: int = 90_000) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars // 2:
        truncated = truncated[:last_para]
    return truncated + "\n\n[... document truncated for length ...]"


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    def __init__(self):
        self.embedder = Embedder(model_name=settings.embedding_model)
        self.vector_store = VectorStore(
            dimension=self.embedder.dimension,
            persist_dir=settings.vector_db_dir,
        )
        self.client = OpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_chunks(self, chunks: list) -> None:
        """Embed and store document chunks."""
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        metadata = [
            {
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "page_num": c.page_num,
                "text": c.text,
            }
            for c in chunks
        ]
        self.vector_store.add(embeddings, metadata)

    def delete_document(self, doc_id: str) -> int:
        return self.vector_store.delete_document(doc_id)

    # ------------------------------------------------------------------
    # Q&A  (streaming)
    # ------------------------------------------------------------------

    def answer_stream(
        self,
        question: str,
        doc_id: str | None = None,
    ) -> Iterator[str]:
        """
        Retrieve relevant chunks, then stream an OpenAI answer.
        Yields JSON-encoded SSE event strings.
        """
        # 1. Embed query
        query_vec = self.embedder.embed_single(question)

        # 2. Retrieve
        if doc_id:
            chunks = self.vector_store.search_by_doc(query_vec, doc_id=doc_id, k=settings.top_k_results)
        else:
            chunks = self.vector_store.search(query_vec, k=settings.top_k_results)

        if not chunks:
            yield json.dumps({"type": "error", "message": "No relevant content found. Please upload a PDF first."})
            return

        context = _build_context(chunks)
        user_message = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"

        try:
            with self.client.chat.completions.create(
                model=settings.openai_model,
                max_tokens=2048,
                stream=True,
                messages=[
                    {"role": "system", "content": _SYSTEM_QA},
                    {"role": "user", "content": user_message},
                ],
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield json.dumps({"type": "text", "content": delta})

            yield json.dumps({"type": "sources", "sources": _format_sources(chunks)})
            yield json.dumps({"type": "done"})

        except APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            yield json.dumps({"type": "error", "message": f"AI service error: {exc.message}"})
        except Exception as exc:
            logger.exception("Unexpected error during answer_stream")
            yield json.dumps({"type": "error", "message": str(exc)})

    # ------------------------------------------------------------------
    # Summarisation (streaming)
    # ------------------------------------------------------------------

    def summarise_stream(
        self,
        full_text: str,
        filename: str = "document",
    ) -> Iterator[str]:
        """Summarise the document; yields SSE event strings."""
        text = _truncate(full_text)
        user_message = f'Please summarise the following document titled "{filename}":\n\n{text}'

        try:
            with self.client.chat.completions.create(
                model=settings.openai_model,
                max_tokens=2048,
                stream=True,
                messages=[
                    {"role": "system", "content": _SYSTEM_SUMMARY},
                    {"role": "user", "content": user_message},
                ],
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield json.dumps({"type": "text", "content": delta})

            yield json.dumps({"type": "done"})

        except APIError as exc:
            logger.error("OpenAI API error during summarise: %s", exc)
            yield json.dumps({"type": "error", "message": f"AI service error: {exc.message}"})
        except Exception as exc:
            logger.exception("Unexpected error during summarise_stream")
            yield json.dumps({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_sources(chunks: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    sources = []
    for chunk in chunks:
        key = (chunk.get("doc_id"), chunk.get("page_num"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "page": chunk.get("page_num"),
                "score": round(chunk.get("score", 0), 3),
                "excerpt": chunk["text"][:200] + "…" if len(chunk["text"]) > 200 else chunk["text"],
            })
    return sources


