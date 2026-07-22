"""
services/research_planner.py
Research planning for multi-query evidence gathering.

Classifies question complexity and generates diverse, complementary
search queries to improve evidence coverage and quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from providers.base import LLMProvider
from utils.log import get_logger

log = get_logger(__name__)


@dataclass
class PlannedQuery:
    """A single search query with its intended purpose."""
    query: str
    purpose: str


# ---------------------------------------------------------------------------
# Complexity classification
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """Classify whether this research question requires a SIMPLE single search or a BROAD multi-faceted investigation.

SIMPLE: Direct factual lookup with one clear answer.
  Examples: "What year was Python created?", "Who is the CEO of Apple?"

BROAD: Topic requiring multiple perspectives or facets to answer well.
  Examples: "Tell me about Qbit Force", "Latest developments in quantum computing"

Respond with exactly one word: SIMPLE or BROAD

Question: {question}"""


def classify_complexity(question: str, provider: LLMProvider) -> str:
    """
    Classify a question as "simple" or "broad".

    Simple questions get a single Tavily search.
    Broad questions trigger multi-query research planning.

    Returns "simple" or "broad". Defaults to "broad" on error.
    """
    try:
        response = provider.generate(
            prompt=_CLASSIFY_PROMPT.format(question=question),
            system_prompt="You are a research complexity classifier.",
            max_tokens=10,
            temperature=0.0,
        )
        result = response.text.strip().upper()

        if "SIMPLE" in result:
            log.info(f"[PLAN] Complexity: simple — direct search")
            return "simple"
        else:
            log.info(f"[PLAN] Complexity: broad — multi-query planning")
            return "broad"

    except Exception as e:
        log.warning(f"[PLAN] Classification failed ({e}), defaulting to broad")
        return "broad"


# ---------------------------------------------------------------------------
# Research planning
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """You are a research planner. Given a user's research question, generate 3-5 complementary web search queries that together will provide comprehensive evidence.

Each query should target a DIFFERENT facet of the topic. Do NOT simply rephrase the same question.

Return a JSON array where each element has:
- "query": the search string
- "purpose": a short description of what this search aims to find

Example for "Tell me about Qbit Force":
[
  {{"query": "Qbit Force company overview", "purpose": "Find official company description and mission"}},
  {{"query": "Qbit Force products services", "purpose": "Identify what the company offers"}},
  {{"query": "Qbit Force founders team LinkedIn", "purpose": "Find leadership and team information"}},
  {{"query": "Qbit Force latest news funding", "purpose": "Discover recent developments and funding"}}
]

Return ONLY the JSON array, no other text.

Question: {question}"""


def plan_research(question: str, provider: LLMProvider) -> list[PlannedQuery]:
    """
    Generate 3-5 diverse, complementary search queries for a broad question.

    Each query targets a different facet of the topic to maximize evidence
    coverage. Falls back to a single direct query on any error.

    Args:
        question: The user's research question.
        provider: LLM provider for query generation.

    Returns:
        List of PlannedQuery objects with query and purpose.
    """
    try:
        response = provider.generate(
            prompt=_PLAN_PROMPT.format(question=question),
            system_prompt="You are a research planner. Output valid JSON only.",
            max_tokens=300,
            temperature=0.3,
        )

        # Parse JSON from response (handle markdown code blocks)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # Remove opening ```json
            text = text.rsplit("```", 1)[0]  # Remove closing ```
            text = text.strip()

        raw = json.loads(text)

        queries = []
        for item in raw:
            q = item.get("query", "").strip()
            p = item.get("purpose", "").strip()
            if q:
                queries.append(PlannedQuery(query=q, purpose=p or "general search"))

        if not queries:
            raise ValueError("No queries parsed from planner output")

        # Log the plan
        for i, pq in enumerate(queries, 1):
            log.info(f"[PLAN] Query {i}: \"{pq.query}\" — {pq.purpose}")

        return queries

    except Exception as e:
        log.warning(f"[PLAN] Planning failed ({e}), using direct query")
        return [PlannedQuery(query=question, purpose="direct search (fallback)")]
