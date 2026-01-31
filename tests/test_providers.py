"""Tests for LLM providers."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.whollm.providers.base import PresenceGuess, BaseLLMProvider
from custom_components.whollm.providers.ollama import OllamaProvider
from custom_components.whollm.const import VALID_ROOMS


class TestPresenceGuess:
    """Tests for PresenceGuess dataclass."""

    def test_creation(self):
        """Test basic creation."""
        guess = PresenceGuess(
            room="office",
            confidence=0.85,
            raw_response="office",
            indicators=["PC active", "Motion detected"],
        )
        
        assert guess.room == "office"
        assert guess.confidence == 0.85
        assert guess.raw_response == "office"
        assert len(guess.indicators) == 2

    def test_to_dict(self, sample_presence_guess):
        """Test conversion to dictionary."""
        result = sample_presence_guess.to_dict()
        
        assert isinstance(result, dict)
        assert result["room"] == "office"
        assert result["confidence"] == 0.8
        assert "indicators" in result

    def test_empty_indicators(self):
        """Test with empty indicators list."""
        guess = PresenceGuess(
            room="unknown",
            confidence=0.0,
            raw_response="",
            indicators=[],
        )
        
        assert guess.indicators == []
        assert guess.to_dict()["indicators"] == []


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider base class."""

    def test_format_context_includes_time(self, sample_context):
        """Test that context formatting includes time information."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        formatted = provider._format_context_for_prompt(
            context=sample_context,
            entity_name="Alice",
            entity_type="person",
        )
        
        assert "14:30" in formatted
        assert "Tuesday" in formatted

    def test_format_context_includes_lights(self, sample_context):
        """Test that context formatting includes light states."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        formatted = provider._format_context_for_prompt(
            context=sample_context,
            entity_name="Alice",
            entity_type="person",
        )
        
        assert "LIGHTS" in formatted
        assert "Office" in formatted or "office" in formatted.lower()

    def test_format_context_includes_motion(self, sample_context):
        """Test that context formatting includes motion sensors."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        formatted = provider._format_context_for_prompt(
            context=sample_context,
            entity_name="Alice",
            entity_type="person",
        )
        
        assert "MOTION" in formatted
        assert "DETECTED" in formatted

    def test_format_context_includes_computers(self, sample_context):
        """Test that context formatting includes computer states."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        formatted = provider._format_context_for_prompt(
            context=sample_context,
            entity_name="Alice",
            entity_type="person",
        )
        
        assert "COMPUTER" in formatted or "PC" in formatted

    def test_system_prompt_person(self):
        """Test system prompt generation for person."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        prompt = provider._get_system_prompt(
            entity_name="Alice",
            entity_type="person",
            rooms=VALID_ROOMS,
        )
        
        assert "Alice" in prompt
        assert "office" in prompt
        assert "living_room" in prompt
        assert "ONLY ONE word" in prompt

    def test_system_prompt_pet(self):
        """Test system prompt generation for pet."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        prompt = provider._get_system_prompt(
            entity_name="Whiskers",
            entity_type="pet",
            rooms=VALID_ROOMS,
        )
        
        assert "Whiskers" in prompt
        assert "pet" in prompt.lower()
        assert "follow" in prompt.lower()  # Pets follow owners

    def test_night_time_context(self):
        """Test that night time is indicated in context."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        night_context = {
            "lights": {},
            "motion": {},
            "media": {},
            "device_trackers": {},
            "computers": {},
            "ai_detection": {},
            "doors": {},
            "time_context": {
                "current_time": "02:00",
                "day_of_week": "Saturday",
                "is_night": True,
                "is_morning": False,
                "is_evening": False,
            },
        }
        
        formatted = provider._format_context_for_prompt(
            context=night_context,
            entity_name="Alice",
            entity_type="person",
        )
        
        assert "NIGHT" in formatted


class TestOllamaProvider:
    """Tests for OllamaProvider implementation."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = OllamaProvider(
            url="http://localhost:11434",
            model="llama3.2",
        )
        
        assert provider.url == "http://localhost:11434"
        assert provider.model == "llama3.2"

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        with patch("custom_components.whollm.providers.ollama.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
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
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        with patch("custom_components.whollm.providers.ollama.aiohttp.ClientSession") as mock_session:
            # Make the context manager raise an exception
            mock_session.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            
            result = await provider.test_connection()
            
            assert result is False

    def test_extract_indicators_light_on(self):
        """Test indicator extraction for lights."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        context = {
            "lights": {
                "light.office": {"state": "on"},
                "light.bedroom": {"state": "off"},
            },
            "motion": {},
            "media": {},
            "computers": {},
            "device_trackers": {},
            "ai_detection": {},
        }
        
        indicators = provider._extract_indicators(context, "office", "Alice")
        
        assert any("Light on" in i for i in indicators)

    def test_extract_indicators_motion(self):
        """Test indicator extraction for motion."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        context = {
            "lights": {},
            "motion": {
                "binary_sensor.office_motion": {"state": "on"},
            },
            "media": {},
            "computers": {},
            "device_trackers": {},
            "ai_detection": {},
        }
        
        indicators = provider._extract_indicators(context, "office", "Alice")
        
        assert any("Motion" in i for i in indicators)

    def test_create_fallback_guess_pc_on(self):
        """Test fallback guess when PC is on."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        context = {
            "lights": {},
            "motion": {},
            "media": {},
            "computers": {"switch.office_pc": {"state": "on"}},
            "device_trackers": {},
            "ai_detection": {},
        }
        
        guess = provider._create_fallback_guess(
            context=context,
            entity_name="Alice",
            entity_type="person",
            rooms=VALID_ROOMS,
        )
        
        assert guess.room == "office"
        assert any("PC" in i for i in guess.indicators)

    def test_create_fallback_guess_tv_on(self):
        """Test fallback guess when TV is playing."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        context = {
            "lights": {},
            "motion": {},
            "media": {"media_player.living_room_tv": {"state": "playing"}},
            "computers": {},
            "device_trackers": {},
            "ai_detection": {},
        }
        
        guess = provider._create_fallback_guess(
            context=context,
            entity_name="Alice",
            entity_type="person",
            rooms=VALID_ROOMS,
        )
        
        assert guess.room == "living_room"

    def test_create_fallback_guess_pet(self):
        """Test fallback guess for pet."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        context = {
            "lights": {},
            "motion": {},
            "media": {"media_player.living_room_tv": {"state": "playing"}},
            "computers": {},
            "device_trackers": {},
            "ai_detection": {},
        }
        
        guess = provider._create_fallback_guess(
            context=context,
            entity_name="Whiskers",
            entity_type="pet",
            rooms=VALID_ROOMS,
        )
        
        # Pet should be near TV if it's on
        assert guess.room == "living_room"

    def test_camera_name_to_room_mapping(self):
        """Test camera name to room mapping."""
        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")
        
        assert provider._camera_name_to_room("living_room_camera") == "living_room"
        assert provider._camera_name_to_room("bedroom_cam") == "bedroom"
        assert provider._camera_name_to_room("office_camera") == "office"
        assert provider._camera_name_to_room("kitchen_camera") == "kitchen"
        assert provider._camera_name_to_room("front_door") == "entry"
        assert provider._camera_name_to_room("random_camera") == "unknown"
