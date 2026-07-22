"""
data/loader.py
Document loading for PDF, TXT, and Markdown files.

Each loader extracts raw text and returns it with source metadata.
"""

import os
from dataclasses import dataclass, field

from utils.log import get_logger
from config import settings

log = get_logger(__name__)


@dataclass
class LoadedDocument:
    """Represents a loaded document with its text and source metadata."""
    source: str           # Original filename
    content: str          # Extracted text
    page_count: int = 1   # Number of pages (meaningful for PDFs)


def load_file(path: str) -> LoadedDocument:
    """
    Load a single file and extract its text content.

    Supports: .pdf, .txt, .md
    Raises ValueError for unsupported file types.
    """
    ext = os.path.splitext(path)[1].lower()
    filename = os.path.basename(path)

    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(settings.SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        return _load_pdf(path, filename)
    else:
        return _load_text(path, filename)


def load_directory(path: str) -> list[LoadedDocument]:
    """Load all supported files from a directory (non-recursive)."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    documents = []
    for entry in sorted(os.listdir(path)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in settings.SUPPORTED_EXTENSIONS:
            full_path = os.path.join(path, entry)
            try:
                doc = load_file(full_path)
                documents.append(doc)
                log.info(f"[DOC] Loaded {entry} ({len(doc.content)} chars)")
            except Exception as e:
                log.error(f"[ERR] Failed to load {entry}: {e}")

    if not documents:
        log.warning("[WARN]  No supported documents found in directory")

    return documents


def _load_pdf(path: str, filename: str) -> LoadedDocument:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF support. Run: pip install pypdf")

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    content = "\n\n".join(pages)
    if not content.strip():
        raise ValueError(f"No extractable text found in {filename}")

    return LoadedDocument(
        source=filename,
        content=content,
        page_count=len(reader.pages),
    )


def _load_text(path: str, filename: str) -> LoadedDocument:
    """Load a plain text or markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        raise ValueError(f"File is empty: {filename}")

    return LoadedDocument(source=filename, content=content)
