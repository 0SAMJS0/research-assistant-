"""Page-aware PDF text extraction and chunking."""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logging.getLogger("pypdf").setLevel(logging.ERROR)

MAX_PAGES = 500


class PDFError(ValueError):
    """The PDF is empty, malformed, encrypted or contains no extractable text."""


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based
    text: str


@dataclass(frozen=True)
class Chunk:
    page: int
    index: int
    text: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_pages(data: bytes) -> tuple[list[PageText], str | None]:
    """Return non-empty page texts and the PDF's metadata title (if any)."""
    if not data:
        raise PDFError("The uploaded file is empty.")
    if not data.lstrip()[:5] == b"%PDF-":
        raise PDFError("The file is not a PDF (missing %PDF header).")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise PDFError("Encrypted PDFs are not supported.")
        if len(reader.pages) == 0:
            raise PDFError("The PDF has no pages.")
        if len(reader.pages) > MAX_PAGES:
            raise PDFError(f"The PDF has more than {MAX_PAGES} pages.")
        pages: list[PageText] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            text = normalize_text(raw)
            if text:
                pages.append(PageText(page=number, text=text))
        title = None
        try:
            if reader.metadata and reader.metadata.title:
                title = str(reader.metadata.title).strip() or None
        except Exception:
            title = None
    except PDFError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise PDFError(f"The PDF could not be parsed ({type(exc).__name__}).") from exc
    if not pages:
        raise PDFError(
            "No extractable text found. The PDF may be image-only (scanned); OCR is not supported."
        )
    return pages, title


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[PageText], chunk_size: int, overlap: int) -> list[Chunk]:
    """Split each page into overlapping chunks on word boundaries; chunks never span pages."""
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        text = page.text
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(page=page.page, index=index, text=piece))
                index += 1
            if end >= len(text):
                break
            next_start = end - overlap
            if overlap:
                space = text.find(" ", next_start, end)
                next_start = space + 1 if space != -1 else next_start
            start = max(next_start, start + 1)
    return chunks
