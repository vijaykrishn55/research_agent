"""
services/research_planner.py
Research planning for multi-query evidence gathering.

Classifies question complexity and generates domain-aware,
complementary search queries to improve evidence coverage.
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
            log.info("[PLAN] Complexity: simple — direct search")
            return "simple"
        else:
            log.info("[PLAN] Complexity: broad — multi-query planning")
            return "broad"

    except Exception as e:
        log.warning(f"[PLAN] Classification failed ({e}), defaulting to broad")
        return "broad"


# ---------------------------------------------------------------------------
# Research planning
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """Given the research question below, generate 3-5 web search queries that will gather the most useful and diverse evidence.

CRITICAL RULES:
- Each query must target a SPECIFIC, DIFFERENT aspect of the topic.
- Tailor queries to the DOMAIN of the subject. Think about what kind of entity this is (company, technology, person, concept, event, etc.) and what information would be most valuable.
- Focus on queries likely to return REAL, SUBSTANTIVE results. Avoid speculative queries about topics that may not exist (e.g., don't search for "funding" unless the subject is likely to have received funding).
- DO NOT use generic templates. Every query should be crafted specifically for THIS subject.

BAD example (generic template applied blindly):
  "Qbit Force competitors", "Qbit Force reviews", "Qbit Force funding"

GOOD example (domain-aware, specific):
  "Qbit Force official website quantum computing"
  "Qbit Force founders team background"
  "Qbit Force quantum hardware technology products"
  "Qbit Force latest news announcements"

Return a JSON array. Each element must have:
- "query": the search string
- "purpose": one sentence explaining what this search aims to discover

Return ONLY the JSON array.

Question: {question}"""


def plan_research(question: str, provider: LLMProvider) -> list[PlannedQuery]:
    """
    Generate 3-5 diverse, domain-aware search queries for a broad question.

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
            max_tokens=400,
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
            log.info(f'[PLAN] Query {i}: "{pq.query}" — {pq.purpose}')

        return queries

    except Exception as e:
        log.warning(f"[PLAN] Planning failed ({e}), using direct query")
        return [PlannedQuery(query=question, purpose="direct search (fallback)")]
