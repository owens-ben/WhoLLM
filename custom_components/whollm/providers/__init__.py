"""LLM Provider implementations for room presence detection."""

from __future__ import annotations

from .base import BaseLLMProvider
from .crewai import CrewAIProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider

# Provider registry
_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "crewai": CrewAIProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(
    provider_type: str,
    url: str,
    model: str,
    **kwargs,
) -> BaseLLMProvider:
    """Get an LLM provider instance by type."""
    if provider_type not in _PROVIDERS:
        raise ValueError(f"Unknown provider type: {provider_type}. Available providers: {list(_PROVIDERS.keys())}")

    provider_class = _PROVIDERS[provider_type]
    return provider_class(url=url, model=model, **kwargs)


def get_available_providers() -> list[str]:
    """Get list of available provider types."""
    return list(_PROVIDERS.keys())


__all__ = [
    "BaseLLMProvider",
    "OllamaProvider",
    "CrewAIProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "get_provider",
    "get_available_providers",
]
