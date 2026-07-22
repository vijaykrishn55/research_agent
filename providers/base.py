"""
providers/base.py
Abstract LLM provider interface and factory.

Every provider returns a uniform ProviderResponse — callers never
need to know which backend produced the answer.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import settings
from utils.log import get_logger

log = get_logger(__name__)


@dataclass
class ProviderResponse:
    """Uniform response from any LLM provider."""
    text: str
    tokens_used: int
    latency_ms: float
    model: str
    provider: str


class LLMProvider(ABC):
    """Abstract interface that all LLM providers implement."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Generate a completion from the LLM."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""
        ...

    def _time_ms(self, start: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return round((time.time() - start) * 1000, 1)


def create_provider(name: str | None = None) -> LLMProvider:
    """
    Factory: create the appropriate LLM provider by name.

    Falls back to settings.LLM_PROVIDER if no name is given.
    Validates that the required API key is set.
    """
    provider_name = (name or settings.LLM_PROVIDER).lower().strip()

    if provider_name == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file. "
                "Get a key at https://console.groq.com"
            )
        from providers.groq_provider import GroqProvider
        return GroqProvider()

    elif provider_name == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to your .env file. "
                "Get a key at https://platform.openai.com"
            )
        from providers.openai_provider import OpenAIProvider
        return OpenAIProvider()

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Supported: groq, openai"
        )
