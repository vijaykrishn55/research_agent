"""
cli.py
Command-line interface for the Research Agent.

Commands:
    python cli.py ingest <path>    — Ingest a file or directory
    python cli.py ask "<question>" — Ask a research question
    python cli.py explain <N>      — Show full evidence for citation [N]
    python cli.py status           — Show index status
    python cli.py clear            — Clear the entire index
"""

import argparse
import json
import os
import re
import sys
import textwrap

from utils.log import setup_logging

setup_logging()


def cmd_ingest(args):
    """Ingest documents from a file or directory."""
    import os
    from services.pipeline import Pipeline

    pipeline = Pipeline()
    path = args.path

    if os.path.isdir(path):
        results = pipeline.ingest_directory(path)
    elif os.path.isfile(path):
        results = [pipeline.ingest_file(path)]
    else:
        print(f"[ERR] Path not found: {path}")
        sys.exit(1)

    print("\n-- Ingestion Results --")
    for r in results:
        status = r["status"]
        icon = "[OK]" if status == "ingested" else "[SKIP]" if status == "skipped" else "[ERR]"
        line = f"  {icon} {r['file']}: {status}"
        if "chunks" in r and r["chunks"]:
            line += f" ({r['chunks']} chunks)"
        if "reason" in r and r["reason"]:
            line += f" — {r['reason']}"
        print(line)

    total = sum(r.get("chunks", 0) for r in results)
    print(f"\n  Total chunks indexed: {total}")


def _extract_citation_ids(text: str) -> set[int]:
    """Parse all [N] and [N, M, ...] citation tags from the answer text."""
    return {int(m) for m in re.findall(r"\[(\d+)\]", text)}


def _strip_synthesis(text: str) -> str:
    """Remove the trailing ## Synthesis section from the LLM answer."""
    return re.sub(
        r"\n*##\s*Synthesis\s*\n.*",
        "",
        text,
        flags=re.DOTALL,
    ).rstrip()


# Detail keys that are rendered as numbered sub-items rather than raw values.
_LIST_DETAIL_KEYS = {"queries", "sources"}


def _render_research_log(events: list[dict]) -> None:
    """Render the pipeline event log as a human-readable nested tree."""
    print("\n\u2500\u2500 Research Log \u2500\u2500")

    for ev in events:
        ts = ev["timestamp"]
        phase = ev["phase"]
        msg = ev["message"]
        dur = ev.get("duration_ms")
        details = ev.get("details", {}) or {}
        level = ev.get("level", "info")

        # Status icon
        icon = "[!]" if level == "error" else "[~]" if level == "warn" else "[+]"

        # Duration badge
        dur_str = f"  ({dur:.0f}ms)" if dur is not None else ""

        # Phase tag — fixed width for alignment
        phase_tag = f"[{phase}]"

        print(f"  {ts}  {icon} {phase_tag:<14} {msg}{dur_str}")

        # Render details
        for key, val in details.items():
            if isinstance(val, list) and key in _LIST_DETAIL_KEYS:
                for i, item in enumerate(val, 1):
                    if isinstance(item, dict) and "query" in item:
                        print(f"          |  {i}. {item['query']}")
                        if "purpose" in item:
                            print(f"          |     -> {item['purpose']}")
                    else:
                        print(f"          |  {i}. {item}")
            elif key not in ("question",):  # skip echoing the question back
                print(f"          |  {key}: {val}")

    print()


def _show_evidence(citations: list, citation_id: int) -> None:
    """Display the full evidence chunk for a specific citation ID."""
    match = None
    for c in citations:
        if c.citation_id == citation_id:
            match = c
            break

    if match is None:
        print(f"\n[ERR] Citation [{citation_id}] not found.")
        return

    print(f"\n── Evidence for [{citation_id}] ──")
    print(f"  Source:  {match.source}")
    print(f"  Type:    {match.source_type}")
    print()

    # Show full text if available, otherwise preview
    text = match.full_text if match.full_text else match.chunk_preview
    print(text)
    print()


def _generate_export_path(base_name: str, fmt: str, export_dir: str = ".") -> str:
    """Generate a timestamped export file path."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^\w\s-]", "", base_name)[:40].strip().replace(" ", "_")
    filename = f"research_{slug}_{ts}.{fmt}"
    return os.path.join(export_dir, filename)


def cmd_ask(args):
    """Ask a research question."""
    from services.pipeline import Pipeline

    pipeline = Pipeline()
    mode = args.mode

    # Only enforce the no-documents guard for local mode.
    # web mode doesn't need local docs; hybrid/auto degrade gracefully.
    if mode == "local" and pipeline.status()["total_chunks"] == 0:
        print("[WARN]  No documents indexed.")
        print("   Run: python cli.py ingest <path>")
        sys.exit(1)

    print(f"\n[?] Researching ({mode} mode): \"{args.question}\"\n")

    answer = pipeline.ask(args.question, top_k=args.top_k, mode=mode)

    # Display research plan if present (hidden with --hide-plan)
    if not args.hide_plan:
        plan = answer.metrics.get("research_plan")
        if plan:
            print("── Research Plan ──")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step['query']}")
                print(f"     → {step['purpose']}")
            print()

    # Strip the Synthesis section from the answer — it repeats the Overview
    display_answer = _strip_synthesis(answer.answer)

    # Print the answer
    print("── Answer ──")
    print(display_answer)

    # Build a complete citation list from ALL [N] IDs in the answer text.
    # This ensures no citation ID referenced in the answer is missing from
    # the display, even if the generator's citation list is incomplete.
    cited_ids = _extract_citation_ids(display_answer)

    # Build lookup from citation objects returned by the generator
    citation_map = {c.citation_id: c for c in answer.citations}

    # Fill gaps for any IDs the generator missed
    from services.generator import Citation
    for cid in sorted(cited_ids):
        if cid not in citation_map:
            citation_map[cid] = Citation(
                citation_id=cid,
                source="[Source not retrieved]",
                chunk_preview="This citation was referenced in the answer but "
                              "its source chunk was not found in the retrieved results.",
                source_type="unknown",
            )

    # Group citations by source type for display
    all_citations = sorted(
        [citation_map[cid] for cid in cited_ids],
        key=lambda c: c.citation_id,
    )

    doc_citations = [c for c in all_citations if c.source_type == "doc"]
    web_citations = [c for c in all_citations if c.source_type == "web"]
    other_citations = [c for c in all_citations if c.source_type not in ("doc", "web")]

    if doc_citations:
        print(f"\n── Document Citations ({len(doc_citations)}) ──")
        for c in doc_citations:
            print(f"  [{c.citation_id}] {c.source}")
            print(f"      \"{c.chunk_preview}\"")

    if web_citations:
        print(f"\n── Web Citations ({len(web_citations)}) ──")
        for c in web_citations:
            print(f"  [{c.citation_id}] {c.source}")
            print(f"      \"{c.chunk_preview}\"")

    if other_citations:
        print(f"\n── Other Citations ({len(other_citations)}) ──")
        for c in other_citations:
            print(f"  [{c.citation_id}] {c.source}")
            print(f"      \"{c.chunk_preview}\"")

    # Print metrics — use the actual unique citation count from the answer
    unique_cited = len(cited_ids)
    print(f"\n── Metrics ──")
    print(f"  Mode:            {answer.metrics.get('mode', mode)}")
    print(f"  Confidence:      {answer.confidence}")
    print(f"  Chunks retrieved: {answer.chunks_retrieved}")
    print(f"  Chunks cited:    {unique_cited}")
    if answer.metrics:
        print(f"  Tokens used:     {answer.metrics.get('tokens_used', 'N/A')}")
        print(f"  Latency:         {answer.metrics.get('total_latency_ms', 'N/A')}ms")
        print(f"  Model:           {answer.metrics.get('model', 'N/A')}")
        print(f"  Provider:        {answer.metrics.get('provider', 'N/A')}")

    # Show the detailed research log if --verbose was passed
    if args.verbose:
        event_log = answer.metrics.get("event_log", [])
        if event_log:
            _render_research_log(event_log)

    # Export to Markdown and/or PDF (optional)
    if args.export:
        from services.exporter import export_markdown, export_pdf
        export_dir = getattr(args, "export_dir", ".") or "."
        base_name = args.question

        for fmt in args.export:
            if args.export_file:
                path = args.export_file
                if not path.endswith(f".{fmt}"):
                    path = f"{path}.{fmt}"
            else:
                path = _generate_export_path(base_name, fmt, export_dir)

            try:
                if fmt == "md":
                    saved = export_markdown(answer, all_citations, path)
                elif fmt == "pdf":
                    saved = export_pdf(answer, all_citations, path)
                else:
                    print(f"[ERR] Unknown export format: {fmt}")
                    continue
                print(f"  [EXPORT] Saved {fmt.upper()}: {saved}")
            except ImportError as e:
                print(f"  [ERR] Export failed: {e}")
            except Exception as e:
                print(f"  [ERR] Export failed: {e}")

    # Output JSON if requested
    if args.json:
        print(f"\n-- JSON Output --")
        # Exclude event_log from metrics to keep JSON clean
        clean_metrics = {k: v for k, v in answer.metrics.items() if k != "event_log"}
        output = {
            "question": answer.question,
            "answer": display_answer,
            "citations": [
                {
                    "id": c.citation_id,
                    "source": c.source,
                    "source_type": c.source_type,
                    "preview": c.chunk_preview,
                }
                for c in all_citations
            ],
            "confidence": answer.confidence,
            "chunks_retrieved": answer.chunks_retrieved,
            "chunks_cited": unique_cited,
            "metrics": clean_metrics,
            "event_log": answer.metrics.get("event_log", []),
        }
        print(json.dumps(output, indent=2))


def cmd_status(args):
    """Show index status."""
    from services.pipeline import Pipeline

    pipeline = Pipeline()
    status = pipeline.status()

    print("\n-- Research Agent Status --")
    print(f"  Index directory:  {status['index_dir']}")
    print(f"  Total chunks:     {status['total_chunks']}")
    print(f"  Total vectors:    {status['total_vectors']}")
    print(f"  Embedding model:  {status['embedding_model']}")
    print(f"  LLM provider:     {status['llm_provider']}")
    print(f"  LLM model:        {status['llm_model']}")

    if status["ingested_files"]:
        print(f"\n  Ingested files ({len(status['ingested_files'])}):")
        for f in status["ingested_files"]:
            print(f"    [DOC] {f}")
    else:
        print("\n  No documents ingested yet.")


def cmd_clear(args):
    """Clear the index."""
    from services.pipeline import Pipeline

    pipeline = Pipeline()
    pipeline.clear()
    print("[DEL]  Index cleared successfully.")


def cmd_explain(args):
    """Show the full evidence chunk for a citation from the last research answer."""
    from services.generator import load_last_answer

    last = load_last_answer()
    if last is None:
        print("[ERR] No previous research found.")
        print("   Run: python cli.py ask \"<question>\" first.")
        sys.exit(1)

    citation_id = args.citation_id

    # Find the matching citation
    match = None
    for c in last.citations:
        if c.citation_id == citation_id:
            match = c
            break

    if match is None:
        available = sorted(c.citation_id for c in last.citations)
        print(f"[ERR] Citation [{citation_id}] not found.")
        print(f"   Available IDs: {available}")
        print(f"   From: \"{last.question}\"")
        sys.exit(1)

    print(f"\n── Evidence for [{citation_id}] ──")
    print(f"  Question: \"{last.question}\"")
    print(f"  Source:    {match.source}")
    print(f"  Type:      {match.source_type}")
    print()

    # Show full text if available, otherwise preview
    text = match.full_text if match.full_text else match.chunk_preview
    print(text)
    print()

    # Also show which parts of the answer cite this ID
    import re
    sentences = re.split(r'(?<=[.!?])\s+', last.answer)
    relevant = [s.strip() for s in sentences if f"[{citation_id}]" in s or f"[{citation_id}," in s or f", {citation_id}]" in s or f", {citation_id}," in s]
    if relevant:
        print(f"── Cited in ──")
        for s in relevant:
            # Clean up markdown headings for display
            clean = s.lstrip("#").strip()
            if clean:
                print(f"  ...{clean}")
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="AI Research Agent — RAG-powered grounded answers with citations",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from a file or directory")
    ingest_parser.add_argument("path", help="Path to a file or directory")
    ingest_parser.set_defaults(func=cmd_ingest)

    # ask
    ask_parser = subparsers.add_parser("ask", help="Ask a research question")
    ask_parser.add_argument("question", help="The research question")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    ask_parser.add_argument(
        "--mode",
        choices=["local", "web", "hybrid", "auto"],
        default="local",
        help="Retrieval mode (default: local)",
    )
    ask_parser.add_argument("--json", action="store_true", help="Also output raw JSON")
    ask_parser.add_argument(
        "--hide-plan", action="store_true",
        help="Hide the research plan from the output (shown by default)",
    )
    ask_parser.add_argument(
        "--verbose", action="store_true",
        help="Show a detailed timestamped log of every pipeline step",
    )
    ask_parser.add_argument(
        "--export", nargs="+", choices=["md", "pdf"],
        metavar="FORMAT",
        help="Export the answer to Markdown and/or PDF (e.g. --export md pdf)",
    )
    ask_parser.add_argument(
        "--export-file", type=str, default=None,
        metavar="PATH",
        help="Custom output filename for export (default: auto-generated)",
    )
    ask_parser.add_argument(
        "--export-dir", type=str, default=".",
        metavar="DIR",
        help="Directory for exported files (default: current directory)",
    )
    ask_parser.set_defaults(func=cmd_ask)

    # status
    status_parser = subparsers.add_parser("status", help="Show index status")
    status_parser.set_defaults(func=cmd_status)

    # clear
    clear_parser = subparsers.add_parser("clear", help="Clear the document index")
    clear_parser.set_defaults(func=cmd_clear)

    # explain
    explain_parser = subparsers.add_parser(
        "explain",
        help="Show full evidence for a citation from the last research answer",
    )
    explain_parser.add_argument(
        "citation_id", type=int, metavar="N",
        help="The citation ID to explain (e.g. 3 for [3])",
    )
    explain_parser.set_defaults(func=cmd_explain)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
