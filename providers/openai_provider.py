"""
providers/openai_provider.py
OpenAI LLM provider — GPT model family.
"""

import time

from openai import OpenAI

from config import settings
from providers.base import LLMProvider, ProviderResponse
from utils.log import get_logger

log = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI inference provider."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        max_tokens = max_tokens or settings.MAX_GENERATION_TOKENS
        temperature = temperature if temperature is not None else settings.GENERATION_TEMPERATURE

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        log.info(f"[>] OpenAI -> {self.model}")
        start = time.time()

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency = self._time_ms(start)
        text = completion.choices[0].message.content or ""
        tokens = completion.usage.total_tokens if completion.usage else 0

        if not text.strip():
            raise RuntimeError(f"OpenAI {self.model} returned empty response")

        log.info(f"[OK] OpenAI responded ({tokens} tokens, {latency}ms)")

        return ProviderResponse(
            text=text,
            tokens_used=tokens,
            latency_ms=latency,
            model=self.model,
            provider=self.provider_name,
        )
