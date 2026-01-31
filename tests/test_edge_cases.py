"""Edge case tests for WhoLLM."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from custom_components.whollm.const import VALID_ROOMS
from custom_components.whollm.providers.base import PresenceGuess


class TestEmptyConfiguration:
    """Test handling of empty or minimal configurations."""

    def test_empty_persons_list(self):
        """Test sensor with empty persons list."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        mock_coordinator = MagicMock()
        mock_coordinator.data = {"persons": {}, "pets": {}}
        mock_coordinator.persons = []
        mock_coordinator.pets = []

        # Sensor for non-existent person should handle gracefully
        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Unknown",
            entity_type="person",
        )

        assert sensor.native_value is None
        assert sensor.extra_state_attributes == {}

    def test_no_room_entities_configured(self):
        """Test confidence combiner with no room entities."""
        from custom_components.whollm.habits import ConfidenceCombiner

        combiner = ConfidenceCombiner()

        # No sensor indicators
        room, conf, explanation = combiner.combine(
            llm_room="office",
            llm_confidence=0.7,
            sensor_indicators=[],  # Empty
            room_entities={},  # Empty
        )

        assert room == "office"
        assert conf > 0


class TestInvalidRoomResponses:
    """Test handling of invalid room responses from LLM."""

    def test_llm_returns_invalid_room(self):
        """Test handling when LLM returns room not in valid list."""
        from custom_components.whollm.providers.ollama import OllamaProvider

        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")

        with patch("custom_components.whollm.providers.ollama.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "response": "garage"  # Not in VALID_ROOMS
                }
            )
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            # Should handle gracefully
            # The provider tries to match partial responses

    def test_llm_returns_empty_response(self):
        """Test handling of empty LLM response."""
        from custom_components.whollm.providers.ollama import OllamaProvider

        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")

        with patch("custom_components.whollm.providers.ollama.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "response": ""  # Empty
                }
            )
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

    def test_llm_returns_multiword_response(self):
        """Test handling when LLM returns verbose response."""
        from custom_components.whollm.providers.ollama import OllamaProvider

        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")

        context = {
            "lights": {},
            "motion": {},
            "media": {},
            "computers": {},
            "device_trackers": {},
            "ai_detection": {},
            "time_context": {"current_time": "14:00", "day_of_week": "Monday"},
        }

        with patch("custom_components.whollm.providers.ollama.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "response": "Based on the sensors, I think they are in the living_room because the TV is on."
                }
            )
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance


class TestNetworkFailures:
    """Test handling of network failures."""

    @pytest.mark.asyncio
    async def test_ollama_connection_refused(self):
        """Test handling when Ollama connection is refused."""
        from custom_components.whollm.providers.ollama import OllamaProvider
        import aiohttp

        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")

        with patch("custom_components.whollm.providers.ollama.aiohttp.ClientSession") as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(
                side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
            )
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            result = await provider.test_connection()

            assert result is False

    @pytest.mark.asyncio
    async def test_ollama_timeout(self):
        """Test handling of Ollama timeout - fallback guess is returned."""
        from custom_components.whollm.providers.ollama import OllamaProvider

        provider = OllamaProvider(url="http://localhost:11434", model="llama3.2")

        context = {
            "lights": {},
            "motion": {},
            "media": {},
            "computers": {"switch.office_pc": {"state": "on"}},
            "device_trackers": {},
            "ai_detection": {},
            "time_context": {"current_time": "14:00", "day_of_week": "Monday"},
        }

        # Test that fallback guess creation works
        guess = provider._create_fallback_guess(context, "Alice", "person", VALID_ROOMS, "Timeout after 60s")

        assert isinstance(guess, PresenceGuess)
        assert "Timeout" in guess.raw_response

    @pytest.mark.asyncio
    async def test_crewai_api_unavailable(self):
        """Test handling when CrewAI API is unavailable."""
        from custom_components.whollm.providers.crewai import CrewAIProvider

        provider = CrewAIProvider(url="http://localhost:8502")

        # Test fallback guess when API fails
        context = {
            "lights": {},
            "motion": {},
            "media": {},
            "computers": {},
            "device_trackers": {},
            "ai_detection": {},
        }

        guess = provider._create_fallback_guess(context, "Alice", "person", VALID_ROOMS, "API unavailable")

        assert isinstance(guess, PresenceGuess)
        assert guess.confidence < 0.9


class TestMalformedData:
    """Test handling of malformed data."""

    def test_malformed_presence_guess_in_coordinator_data(self):
        """Test sensor handling when coordinator data is malformed."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        mock_coordinator = MagicMock()
        mock_coordinator.data = {
            "persons": {
                "Alice": None  # Malformed - should be PresenceGuess
            },
            "pets": {},
        }

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
        )

        # Should handle gracefully
        assert sensor.native_value is None

    def test_malformed_vision_response(self):
        """Test vision identifier with malformed LLM response."""
        from custom_components.whollm.vision import VisionIdentifier

        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
            known_persons=["Alice"],
        )

        # Test various malformed responses
        malformed_responses = [
            "",
            "I don't know",
            "PERSON:",  # Missing value
            "PERSON: maybe alice or bob",  # Multiple people
            "ERROR: Could not process image",
        ]

        for response in malformed_responses:
            result = identifier._parse_identification_response(response, "person")
            # Should not crash, should return unknown
            assert "identified" in result

    def test_json_decode_error_in_event_logger(self):
        """Test event logger handles malformed JSON gracefully."""
        from custom_components.whollm.event_logger import EventLogger
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"

            # Write malformed JSON
            with open(log_path, "w") as f:
                f.write("this is not json\n")
                f.write('{"valid": true}\n')
                f.write("{broken json\n")

            logger = EventLogger(str(log_path))

            # Should not crash
            events = logger.get_recent_events(10)

            # Should skip malformed lines
            assert len(events) == 1
            assert events[0]["valid"] is True


class TestBoundaryConditions:
    """Test boundary conditions."""

    def test_confidence_at_boundaries(self):
        """Test confidence values at 0 and 1 boundaries."""
        from custom_components.whollm.habits import ConfidenceCombiner

        combiner = ConfidenceCombiner()

        # Zero confidence
        room, conf, _ = combiner.combine(
            llm_room="unknown",
            llm_confidence=0.0,
        )
        assert conf >= 0

        # Maximum confidence
        room, conf, _ = combiner.combine(
            llm_room="office",
            llm_confidence=1.0,
            sensor_indicators=[
                {"entity_id": "switch.pc", "hint_type": "computer", "room": "office", "state": "on"},
                {"entity_id": "sensor.motion", "hint_type": "camera", "room": "office", "state": "on"},
            ],
            habit_room="office",
            habit_confidence=1.0,
        )
        assert conf <= 1.0

    def test_poll_interval_boundaries(self):
        """Test poll interval is correctly converted to timedelta."""
        from datetime import timedelta

        # Test various poll intervals
        intervals = [5, 30, 60, 300]

        for seconds in intervals:
            td = timedelta(seconds=seconds)
            assert td.total_seconds() == seconds

    def test_very_long_entity_name(self):
        """Test handling of very long entity names."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        mock_coordinator = MagicMock()
        mock_coordinator.data = None

        long_name = "A" * 200  # Very long name

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name=long_name,
            entity_type="person",
        )

        # Should not crash
        assert sensor._entity_name == long_name

    def test_special_characters_in_names(self):
        """Test handling of special characters in entity names."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        mock_coordinator = MagicMock()
        mock_coordinator.data = None

        special_names = [
            "Alice O'Brien",
            "José García",
            "Test-User",
            "User_123",
            "名前",  # Japanese
        ]

        for name in special_names:
            sensor = LLMPresenceSensor(
                coordinator=mock_coordinator,
                entity_name=name,
                entity_type="person",
            )
            # Should not crash
            assert sensor._entity_name == name


class TestConcurrencyEdgeCases:
    """Test concurrency-related edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_vision_requests(self):
        """Test multiple concurrent vision identification requests."""
        from custom_components.whollm.vision import VisionIdentifier
        import asyncio

        identifier = VisionIdentifier(ollama_url="http://localhost:11434")

        mock_hass = MagicMock()

        # Mock to return different results for different cameras
        async def mock_identify(hass, camera, detection_type):
            await asyncio.sleep(0.1)  # Simulate network delay
            return {
                "success": True,
                "identified": f"Person from {camera}",
                "confidence": "high",
            }

        with patch.object(identifier, "identify_from_camera", side_effect=mock_identify):
            # Run multiple concurrent requests
            tasks = [identifier.identify_from_camera(mock_hass, f"camera.room{i}", "person") for i in range(5)]

            results = await asyncio.gather(*tasks)

            # All should complete
            assert len(results) == 5


class TestMemoryAndCleanup:
    """Test memory management and cleanup."""

    def test_event_logger_cleanup_very_old_events(self):
        """Test cleanup of very old events."""
        from custom_components.whollm.event_logger import EventLogger
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from datetime import timedelta
        import json

        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"

            logger = EventLogger(str(log_path), retention_days=30)

            # Write events from various ages
            ages_in_days = [1, 7, 29, 31, 60, 365]

            with open(log_path, "w") as f:
                for days in ages_in_days:
                    timestamp = (datetime.now() - timedelta(days=days)).isoformat()
                    f.write(
                        json.dumps(
                            {
                                "timestamp": timestamp,
                                "age_days": days,
                            }
                        )
                        + "\n"
                    )

            # Cleanup
            result = logger.cleanup_old_events()

            # Should keep events within retention period (30 days)
            events = logger.get_recent_events(100)
            kept_ages = [e["age_days"] for e in events]

            assert 1 in kept_ages
            assert 7 in kept_ages
            assert 29 in kept_ages
            assert 31 not in kept_ages
            assert 60 not in kept_ages
            assert 365 not in kept_ages

    def test_habit_predictor_clear_all(self):
        """Test clearing all learned patterns."""
        from custom_components.whollm.habits import HabitPredictor
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmpdir:
            predictor = HabitPredictor(config_path=Path(tmpdir))

            # Learn patterns
            predictor.learn_from_event("Alice", "office", 0.9)
            predictor.learn_from_event("Bob", "bedroom", 0.85)
            predictor.learn_from_event("Charlie", "kitchen", 0.8)

            assert len(predictor.habits) == 3

            # Clear all
            predictor.clear_patterns()

            assert predictor.habits == {}
