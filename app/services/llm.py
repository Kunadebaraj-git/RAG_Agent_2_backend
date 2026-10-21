"""LLM Provider interface and Groq implementation."""

import logging
from typing import Optional

from groq import AsyncGroq

from app.config import get_settings

logger = logging.getLogger("docmind")


class LLMProvider:
    """Interface for LLM calls."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq API implementation with multi-model fallback."""

    CANDIDATE_MODELS = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
    ]

    def __init__(self):
        settings = get_settings()
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Set it in your environment or .env file."
            )
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.preferred_model = "qwen/qwen3.8-27b"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response from Groq trying candidate models."""
        models_to_try = [self.preferred_model] + [m for m in self.CANDIDATE_MODELS if m != self.preferred_model]
        last_error = None

        for model in models_to_try:
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.preferred_model = model
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning(f"Groq model {model} failed: {e}. Trying next fallback...")

        logger.exception(f"All Groq models failed: {last_error}")
        raise RuntimeError(f"LLM generation failed: {str(last_error)[:200]}")

    async def generate_with_history(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate with full message history trying candidate models."""
        models_to_try = [self.preferred_model] + [m for m in self.CANDIDATE_MODELS if m != self.preferred_model]
        last_error = None
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        for model in models_to_try:
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=all_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.preferred_model = model
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning(f"Groq model {model} failed with history: {e}. Trying next fallback...")

        logger.exception(f"All Groq models failed: {last_error}")
        raise RuntimeError(f"LLM generation failed: {str(last_error)[:200]}")


# Singleton
_llm_provider: Optional[GroqProvider] = None


def get_llm_provider() -> GroqProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = GroqProvider()
    return _llm_provider
