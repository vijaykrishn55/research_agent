"""
cli.py
Command-line interface for the Research Agent.

Commands:
    python cli.py ingest <path>    — Ingest a file or directory
    python cli.py ask "<question>" — Ask a research question
    python cli.py status           — Show index status
    python cli.py clear            — Clear the entire index
"""

import argparse
import json
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

    # Output JSON if requested
    if args.json:
        print(f"\n-- JSON Output --")
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
            "metrics": answer.metrics,
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
    ask_parser.set_defaults(func=cmd_ask)

    # status
    status_parser = subparsers.add_parser("status", help="Show index status")
    status_parser.set_defaults(func=cmd_status)

    # clear
    clear_parser = subparsers.add_parser("clear", help="Clear the document index")
    clear_parser.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
