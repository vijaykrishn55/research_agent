# Research Agent

An AI-powered research assistant that answers questions grounded in your documents — with citations, not hallucinations.

Built on a Retrieval-Augmented Generation (RAG) pipeline: ingest documents, ask questions, get answers backed by evidence.

## Features

- **Document Ingestion** — Accepts PDF, TXT, and Markdown files. Chunks, embeds, and indexes for fast retrieval.
- **Semantic Search** — Finds the most relevant chunks using dense vector similarity (FAISS + sentence-transformers).
- **Grounded Answers** — Generates responses using only retrieved evidence. Every claim is cited.
- **Citation Tracking** — Each answer includes `[N]` references mapped back to source documents.
- **Honest Uncertainty** — Explicitly states when the available evidence is insufficient rather than fabricating answers.
- **Multi-Provider LLM** — Unified interface supporting Groq (default, free tier) and OpenAI.
- **Token Efficiency** — Retrieves only top-k chunks, uses minimal system prompts, caches embeddings and indexes to disk.
- **Dual Interface** — Both CLI and REST API (FastAPI).
- **Web Search** — Optional Tavily integration with `local`, `web`, `hybrid`, and `auto` retrieval modes.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Documents  │────▶│   Loader    │────▶│   Chunker    │────▶│  Embedder   │
│ PDF/TXT/MD  │     │ data/loader │     │ data/chunker │     │  providers/ │
└─────────────┘     └─────────────┘     └──────────────┘     └──────┬──────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer    │◀────│  Generator  │◀────│  Retriever   │◀────│ FAISS Index │
│ + Citations │     │  services/  │     │  services/   │     │   data/     │
└─────────────┘     └──────┬──────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ LLM Provider│
                    │  Groq/OpenAI│
                    └─────────────┘
```

## Project Structure

```
Research-Agent/
├── server.py                  # FastAPI entry point
├── cli.py                     # CLI interface
├── config/
│   └── settings.py            # All configuration (env-driven)
├── data/
│   ├── loader.py              # PDF/TXT/MD document loading
│   ├── chunker.py             # Paragraph-aware text chunking
│   └── index_store.py         # FAISS index with disk persistence
├── providers/
│   ├── base.py                # Abstract LLM provider + factory
│   ├── groq_provider.py       # Groq inference
│   ├── openai_provider.py     # OpenAI inference
│   └── embedder.py            # Sentence-transformer embeddings
├── services/
│   ├── retriever.py           # Semantic search over indexed chunks
│   ├── generator.py           # Grounded answer generation with citations
│   ├── pipeline.py            # RAG pipeline orchestrator
│   ├── research_planner.py    # Query planning for multi-faceted research
│   ├── tavily_search.py       # Tavily web search client (optional)
│   └── exporter.py            # Markdown/PDF export with citations
├── routes/
│   ├── documents.py           # Document upload/management endpoints
│   └── research.py            # Research question endpoint
├── utils/
│   ├── log.py                 # Structured logging
│   └── events.py              # Pipeline event collector (research log)
├── samples/
│   ├── documents/             # Sample research documents
│   ├── questions.json         # Sample research questions
│   └── outputs/               # Sample outputs
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

### Prerequisites

- Python 3.11+
- An LLM API key (Groq recommended — free tier available)

### Setup

```bash
# Clone the repository
cd Research-Agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env and add your API key
# GROQ_API_KEY=your_key_here
```

### Get a Groq API Key (Free)

1. Visit [console.groq.com](https://console.groq.com)
2. Create an account
3. Generate an API key
4. Add it to your `.env` file

## Quick Start

```bash
# 1. Ingest sample documents
python cli.py ingest samples/documents/

# 2. Ask a research question
python cli.py ask "What are the three main paradigms of machine learning?"

# 3. Check what's indexed
python cli.py status
```

## Usage

### CLI

```bash
# Ingest a single file
python cli.py ingest path/to/document.pdf

# Ingest a directory
python cli.py ingest path/to/documents/

# Ask a question
python cli.py ask "How does quantum entanglement work?"

# Ask using web search only (requires TAVILY_API_KEY)
python cli.py ask "Latest AI research" --mode web

# Ask using local + web results combined
python cli.py ask "How does quantum entanglement work?" --mode hybrid

# Auto mode: local first, web fallback if evidence is thin
python cli.py ask "Recent error correction advances" --mode auto

# Ask with custom retrieval depth
python cli.py ask "What causes climate change?" --top-k 3

# Ask with JSON output
python cli.py ask "What is the Transformer architecture?" --json

# Ask without showing the research plan (shown by default in web/hybrid/auto modes)
python cli.py ask "Latest AI research" --mode web --hide-plan

# Ask with verbose pipeline log showing every step with timestamps
python cli.py ask "How does quantum entanglement work?" --verbose

# Export the answer to Markdown (optional)
python cli.py ask "AI ethics overview" --mode web --export md

# Export to both Markdown and PDF with a custom filename
python cli.py ask "Climate change impacts" --export md pdf --export-file climate_report

# Show the full evidence chunk for citation [3] from the last research
python cli.py explain 3

# View index status
python cli.py status

# Clear the index
python cli.py clear
```

### REST API

```bash
# Start the server
python server.py

# Or with uvicorn directly
uvicorn server:app --reload
```

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload files for ingestion |
| `GET` | `/api/documents` | List ingested files |
| `DELETE` | `/api/documents` | Clear the index |
| `POST` | `/api/research` | Ask a research question |
| `GET` | `/api/research/status` | System status |
| `GET` | `/api/research/evidence/{id}` | Full evidence chunk for citation [id] |

#### Example API Calls

```bash
# Upload a document
curl -X POST http://localhost:8000/api/documents/upload \
  -F "files=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the ethical concerns around AI?"}'

# Explain a specific citation from the last research
curl http://localhost:8000/api/research/evidence/3
```

Interactive docs available at: `http://localhost:8000/docs`

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM provider (`groq` or `openai`) |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `1200` | Target chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks (chars) |
| `TOP_K` | `8` | Chunks to retrieve per query |
| `SIMILARITY_THRESHOLD` | `0.25` | Minimum similarity score |
| `MAX_GENERATION_TOKENS` | `1024` | Max tokens for LLM response |
| `INDEX_DIR` | `.research_agent` | Directory for persisted index |
| `TAVILY_API_KEY` | — | Enables web/hybrid/auto retrieval modes |
| `TAVILY_MAX_RESULTS` | `5` | Max web results per query |
| `TAVILY_SEARCH_DEPTH` | `basic` | `basic` or `advanced` |
| `TAVILY_SEARCH_TOPIC` | `general` | `general` or `news` |
| `AUTO_MIN_CHUNKS` | `2` | Min local chunks before auto-mode skips web |
| `AUTO_MIN_SCORE` | `0.45` | Min local score before auto-mode skips web |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.85` | Jaccard overlap threshold for deduplication |

## Token Efficiency

The system is designed to minimize LLM token usage:

1. **Local embeddings** — sentence-transformers runs locally, no API cost for embeddings.
2. **Top-k retrieval** — Only the most relevant chunks are sent to the LLM (default: 5).
3. **Minimal system prompt** — ~60 tokens of instruction overhead.
4. **Disk-persisted index** — Embeddings computed once, reused across sessions.
5. **Skip-on-duplicate** — Already-ingested files are skipped automatically.
6. **No LLM call when empty** — If no relevant chunks are found, returns an honest response without calling the LLM.

## Research Log

Pass `--verbose` to any `ask` command to see a timestamped, step-by-step log of the entire pipeline execution:

```
── Research Log ──
  15:36:23.412  [+] [pipeline]     Starting research (web mode)
          |  mode: web, top_k: 8
  15:36:24.103  [+] [planner]      Classifying query complexity…
  15:36:25.234  [+] [planner]      Query classified as: broad  (1131ms)
          |  complexity: broad
  15:36:25.240  [+] [planner]      Broad query — planning sub-queries…
  15:36:25.890  [+] [planner]      Research plan created — 5 sub-queries  (650ms)
          |  queries:
          |  1. "Qbit Force official website quantum computing"
          |     -> To find the official website…
          |  2. "Qbit Force founders team background"
          |     -> To discover information about the founders…
  15:36:26.100  [+] [search]       Executing 5 concurrent web searches…
  15:36:28.200  [+] [search]       Fetched 5 page(s) for "Qbit Force official…"  (800ms)
          |  sources:
          |  1. Qbit Force — https://linkedin.com/company/qbit-force
          |  2. About Us — https://qbitforcequantum.com/company
  …
  15:36:31.500  [+] [pipeline]     Merge complete — 33 → 25 chunk(s) (8 dropped)
  15:36:31.510  [+] [generation]   Generating answer from 25 chunk(s)…
  15:36:33.200  [+] [generation]   LLM call complete — groq/llama-3.3-70b-versatile  (2254ms)
          |  model: llama-3.3-70b-versatile
          |  tokens_used: 4845
  15:36:33.300  [+] [pipeline]     Research complete  (10163ms)
```

The log is also available in JSON output (`--json`) as the `event_log` field, and via the REST API in `metrics.event_log`.

## Export & Evidence

### Export to Markdown / PDF

Export the final answer as a standalone document with clickable citations and a full evidence appendix. Export is **optional** — it only runs when you pass the flag.

```bash
# Markdown export
python cli.py ask "AI ethics" --mode web --export md

# PDF export (requires xhtml2pdf — pip install xhtml2pdf)
python cli.py ask "AI ethics" --mode web --export pdf

# Both formats, custom filename
python cli.py ask "AI ethics" --export md pdf --export-file ai_ethics_report
```

**Markdown output** includes clickable `[N]` footnote links, a metadata header (date, mode, confidence), and an Evidence Appendix with full source text and URLs.

**PDF output** converts the Markdown to styled HTML (via the `markdown` library) then renders to PDF with `xhtml2pdf` (pure Python, no GTK needed on Windows). Install both:
```bash
pip install markdown xhtml2pdf
```

### Explain the Evidence

Ask "why?" for any citation to see the exact chunk that supports a claim:

```bash
# Show the full evidence for citation [3] from the last research
python cli.py explain 3
```

Via the REST API, call `GET /api/research/evidence/{citation_id}` after submitting a research question:

```bash
curl http://localhost:8000/api/research/evidence/3
```

Returns the full chunk text, source, and source type for the given citation.

## Sample Output

```
🔍 Researching: "What are the main paradigms of machine learning?"

── Answer ──
The three primary paradigms of machine learning are: supervised learning,
which involves training models on labeled datasets [1]; unsupervised
learning, which operates on data without predefined labels to discover
hidden patterns [2]; and reinforcement learning, which trains agents to
make sequential decisions by maximizing cumulative rewards [1][3].

── Citations (3) ──
  [1] artificial_intelligence.md
      "Machine learning (ML) is a subset of AI that enables systems..."
  [2] artificial_intelligence.md
      "**Unsupervised Learning** operates on data without predefined..."
  [3] artificial_intelligence.md
      "**Reinforcement Learning** (RL) trains agents to make sequential..."

── Metrics ──
  Confidence:       high
  Chunks retrieved:  5
  Chunks cited:      3
  Tokens used:       487
  Latency:           1312.7ms
  Model:             llama-3.1-8b-instant
  Provider:          groq
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector store | FAISS | No external DB dependency, fast, well-tested |
| Embeddings | sentence-transformers (local) | Free, no API calls for embeddings |
| Default model | all-MiniLM-L6-v2 | 80MB, fast, good quality for retrieval |
| LLM default | Groq + Llama 3.1 8B | Free tier, fast inference, sufficient for grounded answers |
| Chunking | Paragraph-aware with overlap | Preserves semantic boundaries |
| Index persistence | FAISS binary + JSON metadata | Simple, reliable, no external DB |
| Provider abstraction | ABC with factory | Easy to add new providers |
| CLI + API | Both | CLI for local use, API for integration |

## Trade-offs and Known Limitations

1. **No cross-encoder reranking** — Retrieval uses bi-encoder similarity only. Adding a cross-encoder reranker would improve precision at the cost of latency and complexity.

2. **No recursive directory scanning** — `ingest` processes files in the given directory only, not subdirectories. This is intentional for predictability.

3. **No incremental re-indexing** — If a document is modified after ingestion, the old chunks remain. Use `clear` and re-ingest to update.

4. **Chunk boundary artifacts** — Despite paragraph-aware splitting, some chunks may split mid-thought. Larger chunk sizes reduce this at the cost of retrieval precision.

5. **Single-user design** — The index is a singleton. Multi-user isolation would require per-user index directories.

6. **No streaming** — LLM responses are returned in full, not streamed. Streaming would improve perceived latency for the API.

7. **Embedding model size** — sentence-transformers requires PyTorch (~2GB). For lighter deployments, consider fastembed (ONNX-based) or API-based embeddings.

8. **PDF quality dependency** — Text extraction from scanned/image-based PDFs will fail. Only text-based PDFs are supported.
