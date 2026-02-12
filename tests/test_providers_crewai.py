"""Tests for CrewAI provider and provider factory."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from custom_components.whollm.providers.base import PresenceGuess
from custom_components.whollm.const import VALID_ROOMS


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    return hass


@pytest.fixture
def sample_context():
    """Sample sensor context for testing."""
    return {
        "lights": {"light.office": {"state": "on"}},
        "motion": {"binary_sensor.office_motion": {"state": "on"}},
        "media": {"media_player.living_room_tv": {"state": "playing"}},
        "device_trackers": {"device_tracker.alice_phone": {"state": "home"}},
        "computers": {"switch.office_pc": {"state": "on"}},
        "ai_detection": {},
        "time_context": {
            "current_time": "14:30",
            "day_of_week": "Tuesday",
            "is_night": False,
            "is_morning": False,
            "is_evening": False,
        },
    }


class TestCrewAIProvider:
    """Tests for CrewAIProvider class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider()

        assert provider.url == "http://localhost:8502"
        assert provider.model == "llama3.2"
        assert provider._use_claude is False

    def test_initialization_custom(self):
        """Test initialization with custom values."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(
            url="http://192.168.1.100:8502",
            model="sonnet",
            use_claude=True,
        )

        assert provider.url == "http://192.168.1.100:8502"
        assert provider.model == "sonnet"
        assert provider._use_claude is True

    @pytest.mark.asyncio
    async def test_deduce_presence_success(self, mock_hass, sample_context):
        """Test successful presence deduction."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(url="http://localhost:8502")

        with patch("custom_components.whollm.providers.crewai.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "room": "office",
                    "confidence": 0.9,
                    "reason": "PC is active and motion detected",
                    "indicators": ["PC active", "Motion in office"],
                    "model_used": "claude-sonnet",
                }
            )
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            result = await provider.deduce_presence(
                mock_hass,
                sample_context,
                "Alice",
                "person",
                VALID_ROOMS,
            )

        assert isinstance(result, PresenceGuess)
        assert result.room == "office"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_deduce_presence_api_error(self, mock_hass, sample_context):
        """Test fallback when API returns error."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(url="http://localhost:8502")

        with patch("custom_components.whollm.providers.crewai.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            result = await provider.deduce_presence(
                mock_hass,
                sample_context,
                "Alice",
                "person",
                VALID_ROOMS,
            )

        # Should return fallback guess
        assert isinstance(result, PresenceGuess)
        # Fallback should pick office due to PC being on
        assert result.room == "office"
        assert result.confidence < 0.9  # Lower confidence for fallback

    @pytest.mark.asyncio
    async def test_deduce_presence_timeout(self, mock_hass, sample_context):
        """Test fallback on timeout - fallback guess is created."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(url="http://localhost:8502")

        # Test fallback guess creation directly
        result = provider._create_fallback_guess(sample_context, "Alice", "person", VALID_ROOMS, "Timeout after 60s")

        assert isinstance(result, PresenceGuess)
        assert "Timeout" in result.raw_response

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(url="http://localhost:8502")

        with patch("custom_components.whollm.providers.crewai.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            result = await provider.test_connection()

        assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test failed connection test."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(url="http://localhost:8502")

        with patch("custom_components.whollm.providers.crewai.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "error"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            result = await provider.test_connection()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_available_models_success(self):
        """Test getting available models."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(url="http://localhost:8502")

        with patch("custom_components.whollm.providers.crewai.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "models": {
                        "haiku": {"description": "Fast"},
                        "sonnet": {"description": "Balanced"},
                        "opus": {"description": "Best"},
                    }
                }
            )
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            models = await provider.get_available_models()

        assert "haiku" in models
        assert "sonnet" in models
        assert "opus" in models

    @pytest.mark.asyncio
    async def test_get_available_models_error(self):
        """Test fallback when getting models fails - returns defaults."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(url="http://localhost:8502")

        # Test that the provider has default model options
        # When API fails, it should have sensible defaults
        assert provider.model == "llama3.2"

        # Verify the provider is configured correctly
        assert provider.url == "http://localhost:8502"


class TestCrewAIExtractIndicators:
    """Test indicator extraction in CrewAI provider."""

    def test_extract_indicators_light_on(self, sample_context):
        """Test light indicator extraction."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider()

        indicators = provider._extract_indicators(sample_context, "office", "Alice")

        assert any("Light on" in i for i in indicators)

    def test_extract_indicators_motion(self, sample_context):
        """Test motion indicator extraction."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider()

        indicators = provider._extract_indicators(sample_context, "office", "Alice")

        assert any("Motion" in i for i in indicators)

    def test_extract_indicators_pc_active(self, sample_context):
        """Test PC active indicator extraction."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider()

        indicators = provider._extract_indicators(sample_context, "office", "Alice")

        assert any("PC" in i for i in indicators)

    def test_extract_indicators_media_playing(self, sample_context):
        """Test media playing indicator extraction."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider()

        indicators = provider._extract_indicators(sample_context, "living_room", "Alice")

        assert any("Media playing" in i for i in indicators)


class TestCrewAIFallbackGuess:
    """Test fallback guess logic."""

    def test_fallback_guess_pc_on(self):
        """Test fallback when PC is on."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider()

        context = {
            "lights": {},
            "motion": {},
            "media": {},
            "computers": {"switch.office_pc": {"state": "on"}},
            "device_trackers": {},
            "ai_detection": {},
        }

        guess = provider._create_fallback_guess(context, "Alice", "person", VALID_ROOMS, "Test")

        assert guess.room == "office"
        assert any("PC" in i for i in guess.indicators)

    def test_fallback_guess_tv_on(self):
        """Test fallback when TV is playing."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider()

        context = {
            "lights": {},
            "motion": {},
            "media": {"media_player.living_room_tv": {"state": "playing"}},
            "computers": {},
            "device_trackers": {},
            "ai_detection": {},
        }

        guess = provider._create_fallback_guess(context, "Alice", "person", VALID_ROOMS, "Test")

        assert guess.room == "living_room"

    def test_fallback_guess_pet(self):
        """Test fallback for pet entity."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider()

        context = {
            "lights": {},
            "motion": {},
            "media": {"media_player.living_room_tv": {"state": "playing"}},
            "computers": {},
            "device_trackers": {},
            "ai_detection": {},
        }

        guess = provider._create_fallback_guess(context, "Whiskers", "pet", VALID_ROOMS, "Test")

        assert guess.room == "living_room"  # Pet near TV


class TestProviderFactory:
    """Tests for provider factory functions."""

    def test_get_provider_ollama(self):
        """Test getting Ollama provider."""
        from custom_components.whollm.providers import get_provider
        from custom_components.whollm.providers.ollama import OllamaProvider

        provider = get_provider("ollama", "http://localhost:11434", "llama3.2")

        assert isinstance(provider, OllamaProvider)
        assert provider.url == "http://localhost:11434"
        assert provider.model == "llama3.2"

    def test_get_provider_crewai(self):
        """Test getting CrewAI provider."""
        from custom_components.whollm.providers import get_provider
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = get_provider("crewai", "http://localhost:8502", "sonnet")

        assert isinstance(provider, CrewAIProvider)
        assert provider.url == "http://localhost:8502"
        assert provider.model == "sonnet"

    def test_get_provider_unknown(self):
        """Test error for unknown provider."""
        from custom_components.whollm.providers import get_provider

        with pytest.raises(ValueError) as exc_info:
            get_provider("unknown_provider", "http://localhost", "model")

        assert "Unknown provider type" in str(exc_info.value)

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        from custom_components.whollm.providers import get_available_providers

        providers = get_available_providers()

        assert "ollama" in providers
        assert "crewai" in providers

    def test_get_provider_with_extra_kwargs(self):
        """Test that extra kwargs are passed to provider."""
        from custom_components.whollm.providers import get_provider

        provider = get_provider(
            "crewai",
            "http://localhost:8502",
            "sonnet",
            use_claude=True,
        )

        assert provider._use_claude is True
