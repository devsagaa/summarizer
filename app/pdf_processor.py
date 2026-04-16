"""PDF text extraction and smart chunking."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class DocumentPage:
    page_num: int
    text: str


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    page_num: int
    text: str
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())


def extract_text(pdf_path: str | Path) -> tuple[list[DocumentPage], int]:
    """
    Extract text from every page of a PDF.
    Returns (pages, total_page_count).
    """
    pages: list[DocumentPage] = []
    pdf_path = Path(pdf_path)

    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
        for page_num, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            # Normalise whitespace: collapse runs of spaces/tabs, preserve newlines
            cleaned = re.sub(r"[ \t]+", " ", raw)
            # Collapse 3+ consecutive newlines to 2
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            if cleaned:
                pages.append(DocumentPage(page_num=page_num, text=cleaned))

    return pages, total_pages


def chunk_pages(
    pages: list[DocumentPage],
    doc_id: str,
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[Chunk]:
    """
    Split page text into overlapping word-level chunks.

    Strategy:
    - Prefer sentence boundaries when possible.
    - Maintain `overlap` words of context between consecutive chunks.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    # Sentence boundary splitter: split after . ! ? followed by whitespace
    _sent_re = re.compile(r"(?<=[.!?])\s+")

    for page in pages:
        sentences = _sent_re.split(page.text)
        # Flatten into a list of words tagged with their source page
        words: list[str] = []
        for sentence in sentences:
            words.extend(sentence.split())

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            window = words[start:end]
            chunk_text = " ".join(window).strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        page_num=page.page_num,
                        text=chunk_text,
                    )
                )
                chunk_index += 1

            if end >= len(words):
                break
            start = end - overlap  # slide back by overlap

    return chunks


def process_pdf(pdf_path: str | Path, chunk_size: int = 400, overlap: int = 60) -> dict:
    """
    Full pipeline: extract → chunk.
    Returns a dict with all extracted data.
    """
    pdf_path = Path(pdf_path)
    doc_id = uuid.uuid4().hex

    pages, total_pages = extract_text(pdf_path)
    chunks = chunk_pages(pages, doc_id, chunk_size=chunk_size, overlap=overlap)

    full_text = "\n\n".join(p.text for p in pages)

    return {
        "doc_id": doc_id,
        "total_pages": total_pages,
        "chunks": chunks,
        "full_text": full_text,
        "char_count": len(full_text),
    }
