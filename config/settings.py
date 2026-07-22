"""
config/settings.py
Central configuration — all tunables live here.

Loads from .env, falls back to sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# -- LLM Provider --------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -- Embedding -----------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# -- Chunking ------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# -- Retrieval -----------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "8"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))

# -- Tavily Web Search (optional) ----------------------------------------------
TAVILY_API_KEY        = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS    = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
TAVILY_SEARCH_DEPTH   = os.getenv("TAVILY_SEARCH_DEPTH", "basic")    # basic | advanced
TAVILY_SEARCH_TOPIC   = os.getenv("TAVILY_SEARCH_TOPIC", "general")  # general | news
TAVILY_OPTIMIZE_QUERY = os.getenv("TAVILY_OPTIMIZE_QUERY", "false").lower() == "true"

# -- Auto-mode thresholds ------------------------------------------------------
AUTO_MIN_CHUNKS = int(os.getenv("AUTO_MIN_CHUNKS", "2"))
AUTO_MIN_SCORE  = float(os.getenv("AUTO_MIN_SCORE", "0.45"))

# -- Deduplication -------------------------------------------------------------
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))

# -- Generation ----------------------------------------------------------------
MAX_GENERATION_TOKENS = int(os.getenv("MAX_GENERATION_TOKENS", "1024"))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))

# -- Storage -------------------------------------------------------------------
INDEX_DIR = os.getenv("INDEX_DIR", ".research_agent")

# -- Server --------------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# -- Supported file types -----------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
