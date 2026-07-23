"""
services/exporter.py
Export research answers to Markdown and PDF with citations and evidence.

Markdown export:  Standalone .md with clickable footnote citations and
                  a full evidence appendix.
PDF export:       Converts the Markdown to styled HTML via the `markdown`
                  library, then renders to PDF with xhtml2pdf (pure Python,
                  no GTK dependency on Windows).
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from services.generator import Citation, ResearchAnswer
from utils.log import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_markdown(
    answer: ResearchAnswer,
    citations: list[Citation],
    path: str,
) -> str:
    """Export the research answer to a Markdown file.

    Args:
        answer:    The ResearchAnswer from the pipeline.
        citations: The complete citation list (from CLI gap-filling).
        path:      Output file path (.md).

    Returns:
        The absolute path of the written file.
    """
    md = _build_markdown(answer, citations)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    log.info(f"[EXPORT] Markdown written to {path}")
    return os.path.abspath(path)


def export_pdf(
    answer: ResearchAnswer,
    citations: list[Citation],
    path: str,
) -> str:
    """Export the research answer to a styled PDF file.

    Requires the `markdown` and `xhtml2pdf` packages.  Falls back to
    a clear error message if either is missing.

    Args:
        answer:    The ResearchAnswer from the pipeline.
        citations: The complete citation list (from CLI gap-filling).
        path:      Output file path (.pdf).

    Returns:
        The absolute path of the written file.
    """
    md = _build_markdown(answer, citations)

    # Convert Markdown -> HTML
    try:
        import markdown as md_lib
    except ImportError:
        raise ImportError(
            "The `markdown` package is required for PDF export.  "
            "Install with: pip install markdown"
        )

    html_body = md_lib.markdown(
        md,
        extensions=["tables", "toc", "smarty"],
    )

    # Build full HTML document with embedded CSS
    html_doc = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
{_pdf_css()}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Render HTML -> PDF
    try:
        from xhtml2pdf import pisa
    except ImportError:
        raise ImportError(
            "The `xhtml2pdf` package is required for PDF export.  "
            "Install with: pip install xhtml2pdf"
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        status = pisa.CreatePDF(html_doc, dest=f)

    if status.err:
        raise RuntimeError(
            f"PDF generation failed with {status.err} error(s).  "
            "Check that the HTML content is well-formed."
        )

    log.info(f"[EXPORT] PDF written to {path}")
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------

def _build_markdown(
    answer: ResearchAnswer,
    citations: list[Citation],
) -> str:
    """Build a standalone Markdown document from the research answer."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = answer.metrics.get("mode", "local")

    lines: list[str] = []

    # Title & metadata
    lines.append(f'# Research Brief: "{answer.question}"')
    lines.append("")
    lines.append(
        f"**Date:** {now}  |  **Mode:** {mode}  |  "
        f"**Confidence:** {answer.confidence}  |  "
        f"**Chunks cited:** {answer.chunks_cited}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Answer body — convert inline [N] to Markdown reference links
    body = _rewrite_citations(answer.answer)
    lines.append(body)
    lines.append("")

    # Evidence appendix
    lines.append("---")
    lines.append("")
    lines.append("## Evidence Appendix")
    lines.append("")

    # Build reference-style link definitions for clickable citations
    # and a detailed evidence entry for each citation.
    for c in sorted(citations, key=lambda x: x.citation_id):
        source_label = c.source
        url = _extract_url(c.source)

        # Evidence entry
        lines.append(f"### [{c.citation_id}] {source_label}")
        lines.append("")
        if c.source_type == "web" and url:
            lines.append(f"**URL:** <{url}>")
            lines.append("")
        lines.append(f"**Type:** {c.source_type}")
        lines.append("")

        # Show full text if available, otherwise preview
        evidence_text = c.full_text or c.chunk_preview
        lines.append("> " + evidence_text.replace("\n", "\n> "))
        lines.append("")

    # Reference link definitions (makes [N] citations clickable in MD renderers)
    lines.append("---")
    lines.append("")
    for c in sorted(citations, key=lambda x: x.citation_id):
        url = _extract_url(c.source)
        if url:
            lines.append(f"[ref-{c.citation_id}]: {url}")

    lines.append("")
    return "\n".join(lines)


def _rewrite_citations(text: str) -> str:
    """Convert [N] and [N, M] citation tags to Markdown reference links.

    [1]      -> [[1]][ref-1]
    [3, 11]  -> [[3]][ref-3], [[11]][ref-11]

    Uses a single regex pass to avoid double-wrapping already-converted tags.
    """
    def _replace(match: re.Match) -> str:
        inner = match.group(1)
        ids = [i.strip() for i in inner.split(",")]
        if len(ids) == 1:
            return f"[[{ids[0]}]][ref-{ids[0]}]"
        return ", ".join(f"[[{i}]][ref-{i}]" for i in ids)

    # Single pass handles both [N] and [N, M, ...]
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _replace, text)


def _extract_url(source: str) -> str | None:
    """Extract URL from a source string like '[WEB] Title — https://...'."""
    if not source.startswith("[WEB]"):
        return None
    # URL is everything after the last " — " (em dash)
    parts = source.rsplit(" — ", 1)
    if len(parts) == 2 and parts[1].startswith("http"):
        return parts[1].strip()
    # Fallback: look for http(s) anywhere
    match = re.search(r"https?://\S+", source)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# PDF styling
# ---------------------------------------------------------------------------

def _pdf_css() -> str:
    """Return embedded CSS for the PDF document."""
    return """\
    @page {
        size: A4;
        margin: 2cm;
    }
    body {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #1a1a1a;
    }
    h1 {
        font-size: 18pt;
        color: #111;
        border-bottom: 2px solid #333;
        padding-bottom: 8px;
    }
    h2 {
        font-size: 14pt;
        color: #222;
        margin-top: 24px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 4px;
    }
    h3 {
        font-size: 11pt;
        color: #333;
        margin-top: 16px;
    }
    blockquote {
        border-left: 3px solid #666;
        margin: 8px 0;
        padding: 4px 12px;
        color: #444;
        font-size: 10pt;
        background: #f9f9f9;
    }
    a {
        color: #0066cc;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 20px 0;
    }
    strong {
        color: #111;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 6px 10px;
        text-align: left;
        font-size: 10pt;
    }
    th {
        background: #f0f0f0;
    }
    p {
        margin: 6px 0;
    }
"""
