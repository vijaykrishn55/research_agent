"""
data/chunker.py
Text chunking with paragraph-aware splitting and configurable overlap.

Produces Chunk objects that carry source metadata for citation tracking.
"""

from dataclasses import dataclass

from config import settings
from utils.log import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""
    chunk_id: str       # Unique ID: "{source}::chunk_{index}"
    content: str        # The chunk text
    source: str         # Source filename
    chunk_index: int    # Position within the source document

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(
            chunk_id=d["chunk_id"],
            content=d["content"],
            source=d["source"],
            chunk_index=d["chunk_index"],
        )


def chunk_document(
    text: str,
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split document text into overlapping chunks.

    Strategy:
    1. Split on double-newlines (paragraph boundaries) first.
    2. Merge small paragraphs until chunk_size is reached.
    3. Split oversized paragraphs at sentence boundaries.
    4. Apply overlap between consecutive chunks.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    # Merge small paragraphs and split large ones into raw segments
    segments = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            segments.append(para)
        else:
            # Split oversized paragraph at sentence boundaries
            segments.extend(_split_long_text(para, chunk_size))

    # Combine segments into chunks with overlap
    chunks = []
    current_text = ""

    for segment in segments:
        # If adding this segment exceeds chunk_size, finalize current chunk
        candidate = f"{current_text}\n\n{segment}".strip() if current_text else segment

        if len(candidate) > chunk_size and current_text:
            chunks.append(current_text)
            # Start new chunk with overlap from the end of previous
            overlap_text = _get_overlap(current_text, chunk_overlap)
            current_text = f"{overlap_text}\n\n{segment}".strip() if overlap_text else segment
        else:
            current_text = candidate

    # Don't forget the last chunk
    if current_text:
        chunks.append(current_text)

    # Convert to Chunk objects
    result = []
    for i, content in enumerate(chunks):
        result.append(Chunk(
            chunk_id=f"{source}::chunk_{i}",
            content=content,
            source=source,
            chunk_index=i,
        ))

    log.info(f"[+] Chunked {source} -> {len(result)} chunks (avg {_avg_len(result)} chars)")
    return result


def _split_long_text(text: str, max_size: int) -> list[str]:
    """Split long text at sentence boundaries."""
    # Try splitting at sentence-ending punctuation
    sentences = []
    current = ""

    for char in text:
        current += char
        if char in ".!?" and len(current) >= 50:
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    # Merge sentences into segments that fit max_size
    segments = []
    current_segment = ""

    for sentence in sentences:
        candidate = f"{current_segment} {sentence}".strip() if current_segment else sentence
        if len(candidate) > max_size and current_segment:
            segments.append(current_segment)
            current_segment = sentence
        else:
            current_segment = candidate

    if current_segment:
        segments.append(current_segment)

    return segments if segments else [text[:max_size]]


def _get_overlap(text: str, overlap_size: int) -> str:
    """Extract the last `overlap_size` characters for chunk overlap."""
    if len(text) <= overlap_size:
        return text
    # Try to break at a word boundary
    overlap = text[-overlap_size:]
    space_idx = overlap.find(" ")
    if space_idx > 0 and space_idx < len(overlap) // 2:
        overlap = overlap[space_idx + 1:]
    return overlap


def _avg_len(chunks: list[Chunk]) -> int:
    """Average chunk length in characters."""
    if not chunks:
        return 0
    return sum(len(c.content) for c in chunks) // len(chunks)
