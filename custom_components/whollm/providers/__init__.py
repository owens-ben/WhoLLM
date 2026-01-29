"""LLM Provider implementations for room presence detection."""
from __future__ import annotations

from .base import BaseLLMProvider
from .ollama import OllamaProvider
from .crewai import CrewAIProvider

# Provider registry
_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "crewai": CrewAIProvider,
    # Stubbed providers - not yet implemented
    # "openai": OpenAIProvider,
    # "anthropic": AnthropicProvider,
    # "local": LocalProvider,
}


def get_provider(
    provider_type: str,
    url: str,
    model: str,
    **kwargs,
) -> BaseLLMProvider:
    """Get an LLM provider instance by type."""
    if provider_type not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Available providers: {list(_PROVIDERS.keys())}"
        )
    
    provider_class = _PROVIDERS[provider_type]
    return provider_class(url=url, model=model, **kwargs)


def get_available_providers() -> list[str]:
    """Get list of available provider types."""
    return list(_PROVIDERS.keys())


__all__ = [
    "BaseLLMProvider",
    "OllamaProvider",
    "CrewAIProvider",
    "get_provider",
    "get_available_providers",
]


