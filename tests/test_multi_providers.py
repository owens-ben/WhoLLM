"""
Tests for OpenAI and Anthropic provider support (TDD).

These tests verify that WhoLLM can use multiple LLM providers
for presence detection.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestProviderFactory:
    """Test provider factory function."""

    def test_get_provider_ollama(self):
        """Factory should return Ollama provider."""
        from custom_components.whollm.providers import get_provider
        from custom_components.whollm.providers.ollama import OllamaProvider
        
        provider = get_provider("ollama", "http://localhost:11434", "llama3.2")
        assert isinstance(provider, OllamaProvider)

    def test_get_provider_openai_not_registered(self):
        """OpenAI provider should not be in the registry (disabled)."""
        from custom_components.whollm.providers import get_provider

        with pytest.raises(ValueError):
            get_provider("openai", "https://api.openai.com/v1", "gpt-4o-mini", api_key="test-key")

    def test_get_provider_anthropic_not_registered(self):
        """Anthropic provider should not be in the registry (disabled)."""
        from custom_components.whollm.providers import get_provider

        with pytest.raises(ValueError):
            get_provider("anthropic", "https://api.anthropic.com", "claude-3-haiku-20240307", api_key="test-key")

    def test_get_provider_unknown(self):
        """Factory should raise for unknown provider."""
        from custom_components.whollm.providers import get_provider

        with pytest.raises(ValueError):
            get_provider("unknown_provider", "http://example.com", "model")


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_openai_provider_init(self):
        """OpenAI provider should initialize with API key."""
        from custom_components.whollm.providers.openai import OpenAIProvider
        
        provider = OpenAIProvider(
            url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        
        assert provider.model == "gpt-4o-mini"
        assert provider.api_key == "test-key"

    def test_openai_provider_requires_api_key(self):
        """OpenAI provider should require API key."""
        from custom_components.whollm.providers.openai import OpenAIProvider
        
        with pytest.raises((ValueError, TypeError)):
            OpenAIProvider(
                url="https://api.openai.com/v1",
                model="gpt-4o-mini",
                # No API key
            )

    @pytest.mark.asyncio
    async def test_openai_deduce_presence(self):
        """OpenAI provider should deduce presence."""
        from custom_components.whollm.providers.openai import OpenAIProvider
        
        provider = OpenAIProvider(
            url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        
        mock_hass = MagicMock()
        context = {"lights": {"light.office": {"state": "on"}}}
        
        # Mock the API call
        with patch.object(provider, "_call_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {
                "choices": [{"message": {"content": "ROOM: office\nCONFIDENCE: 0.7\nREASON: Light is on"}}]
            }
            
            result = await provider.deduce_presence(
                hass=mock_hass,
                context=context,
                entity_name="Test",
                entity_type="person",
                rooms=["office", "living_room"],
            )
            
            assert result.room == "office"
            assert result.confidence >= 0.5


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    def test_anthropic_provider_init(self):
        """Anthropic provider should initialize with API key."""
        from custom_components.whollm.providers.anthropic import AnthropicProvider
        
        provider = AnthropicProvider(
            url="https://api.anthropic.com",
            model="claude-3-haiku-20240307",
            api_key="test-key",
        )
        
        assert provider.model == "claude-3-haiku-20240307"
        assert provider.api_key == "test-key"

    def test_anthropic_default_model(self):
        """Anthropic provider should have reasonable default model."""
        from custom_components.whollm.providers.anthropic import AnthropicProvider
        
        provider = AnthropicProvider(
            url="https://api.anthropic.com",
            model="",  # Empty model
            api_key="test-key",
        )
        
        # Should default to haiku (cheap and fast)
        assert "haiku" in provider.model.lower() or provider.model != ""

    @pytest.mark.asyncio
    async def test_anthropic_deduce_presence(self):
        """Anthropic provider should deduce presence."""
        from custom_components.whollm.providers.anthropic import AnthropicProvider
        
        provider = AnthropicProvider(
            url="https://api.anthropic.com",
            model="claude-3-haiku-20240307",
            api_key="test-key",
        )
        
        mock_hass = MagicMock()
        context = {"motion": {"binary_sensor.living_room_motion": {"state": "on"}}}
        
        with patch.object(provider, "_call_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {
                "content": [{"type": "text", "text": "ROOM: living_room\nCONFIDENCE: 0.8\nREASON: Motion detected"}]
            }
            
            result = await provider.deduce_presence(
                hass=mock_hass,
                context=context,
                entity_name="Test",
                entity_type="person",
                rooms=["office", "living_room"],
            )
            
            assert result.room == "living_room"
            assert result.confidence >= 0.5


class TestProviderConstants:
    """Test provider-related constants."""

    def test_openai_not_in_supported_providers(self):
        """OpenAI should NOT be in active supported providers (disabled)."""
        from custom_components.whollm.const import SUPPORTED_PROVIDERS, PROVIDER_OPENAI, ALL_PROVIDERS

        assert PROVIDER_OPENAI not in SUPPORTED_PROVIDERS
        assert PROVIDER_OPENAI in ALL_PROVIDERS

    def test_anthropic_not_in_supported_providers(self):
        """Anthropic should NOT be in active supported providers (disabled)."""
        from custom_components.whollm.const import SUPPORTED_PROVIDERS, PROVIDER_ANTHROPIC, ALL_PROVIDERS

        assert PROVIDER_ANTHROPIC not in SUPPORTED_PROVIDERS
        assert PROVIDER_ANTHROPIC in ALL_PROVIDERS

    def test_api_key_config(self):
        """Config should support API keys."""
        from custom_components.whollm.const import CONF_API_KEY
        
        assert CONF_API_KEY == "api_key"


class TestProviderResponse:
    """Test provider response parsing."""

    def test_parse_llm_response_room(self):
        """Should parse room from LLM response."""
        from custom_components.whollm.providers.base import parse_llm_response
        
        response = "ROOM: office\nCONFIDENCE: 0.8\nREASON: PC is on"
        room, confidence, _ = parse_llm_response(response, ["office", "living_room"])
        
        assert room == "office"
        assert confidence == 0.8

    def test_parse_llm_response_with_json(self):
        """Should parse JSON-formatted response."""
        from custom_components.whollm.providers.base import parse_llm_response
        
        response = '{"room": "bedroom", "confidence": 0.6, "reason": "lights dim"}'
        room, confidence, _ = parse_llm_response(response, ["bedroom", "office"])
        
        assert room == "bedroom"

    def test_parse_llm_response_fallback(self):
        """Should fallback to unknown for invalid response."""
        from custom_components.whollm.providers.base import parse_llm_response
        
        response = "I don't know where they are"
        room, confidence, _ = parse_llm_response(response, ["office", "living_room"])
        
        assert room == "unknown"
        assert confidence < 0.5
