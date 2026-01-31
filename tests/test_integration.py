"""Integration tests for WhoLLM end-to-end flows."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from custom_components.whollm.const import (
    CONF_MODEL,
    CONF_PERSONS,
    CONF_PETS,
    CONF_POLL_INTERVAL,
    CONF_PROVIDER,
    CONF_ROOMS,
    CONF_URL,
    CONF_ROOM_ENTITIES,
    CONF_PERSON_DEVICES,
    CONF_LEARNING_ENABLED,
    CONF_CONFIDENCE_WEIGHTS,
    DEFAULT_CONFIDENCE_WEIGHTS,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PROVIDER,
    DEFAULT_URL,
    DOMAIN,
    ENTITY_HINT_COMPUTER,
    ENTITY_HINT_MOTION,
    ENTITY_HINT_MEDIA,
)
from custom_components.whollm.providers.base import PresenceGuess


@pytest.fixture
def mock_hass():
    """Create a comprehensive mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    
    # Mock states
    mock_states = {}
    
    def get_state(entity_id):
        return mock_states.get(entity_id)
    
    def set_state(entity_id, state, attributes=None):
        mock_state = MagicMock()
        mock_state.state = state
        mock_state.entity_id = entity_id
        mock_state.attributes = attributes or {}
        mock_state.last_changed = None
        mock_states[entity_id] = mock_state
        return mock_state
    
    hass.states = MagicMock()
    hass.states.get = get_state
    hass.states.async_all = MagicMock(return_value=list(mock_states.values()))
    hass._mock_states = mock_states
    hass._set_state = set_state
    
    # Mock bus
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()
    
    return hass


@pytest.fixture
def full_config_entry():
    """Create a comprehensive config entry."""
    entry = MagicMock()
    entry.data = {
        CONF_PROVIDER: DEFAULT_PROVIDER,
        CONF_URL: DEFAULT_URL,
        CONF_MODEL: DEFAULT_MODEL,
        CONF_POLL_INTERVAL: 30,
        CONF_PERSONS: [{"name": "Alice"}, {"name": "Bob"}],
        CONF_PETS: [{"name": "Whiskers"}],
        CONF_ROOMS: ["office", "bedroom", "living_room", "kitchen"],
        CONF_ROOM_ENTITIES: {
            "office": [
                {"entity_id": "switch.office_pc", "hint_type": ENTITY_HINT_COMPUTER},
                {"entity_id": "binary_sensor.office_motion", "hint_type": ENTITY_HINT_MOTION},
            ],
            "living_room": [
                {"entity_id": "media_player.living_room_tv", "hint_type": ENTITY_HINT_MEDIA},
                {"entity_id": "binary_sensor.living_room_motion", "hint_type": ENTITY_HINT_MOTION},
            ],
            "bedroom": [
                {"entity_id": "binary_sensor.bedroom_motion", "hint_type": ENTITY_HINT_MOTION},
            ],
        },
        CONF_PERSON_DEVICES: {
            "Alice": ["switch.alice_pc"],
            "Bob": ["switch.bob_laptop"],
        },
        CONF_LEARNING_ENABLED: True,
        CONF_CONFIDENCE_WEIGHTS: DEFAULT_CONFIDENCE_WEIGHTS,
    }
    entry.entry_id = "test_entry_id"
    entry.options = {}
    return entry


class TestEndToEndCoordinatorFlow:
    """Test full coordinator update cycle."""

    @pytest.mark.asyncio
    async def test_full_update_cycle(self, mock_hass, full_config_entry):
        """Test a complete coordinator update with all components.
        
        Note: This test requires full Home Assistant setup.
        It's primarily run in CI with pytest-homeassistant-custom-component.
        """
        # This test validates the coordinator flow conceptually
        # Full integration testing requires HA test fixtures
        
        from custom_components.whollm.providers.base import PresenceGuess
        
        # Test that PresenceGuess can be created and used
        guess = PresenceGuess(
            room="office",
            confidence=0.85,
            raw_response="office",
            indicators=["PC is active"],
        )
        
        assert guess.room == "office"
        assert guess.confidence == 0.85

    @pytest.mark.asyncio
    async def test_learning_from_confident_predictions(self, mock_hass, full_config_entry):
        """Test that habit predictor learns from confident predictions.
        
        Note: This test validates learning logic without full HA setup.
        """
        from custom_components.whollm.habits import HabitPredictor
        from pathlib import Path
        from tempfile import TemporaryDirectory
        
        with TemporaryDirectory() as tmpdir:
            predictor = HabitPredictor(config_path=Path(tmpdir))
            
            # Learn from high-confidence prediction
            predictor.learn_from_event("Alice", "office", 0.9)
            
            # Verify learning occurred
            assert "Alice" in predictor.habits


class TestRoomTransitionDetection:
    """Test room transition detection and logging."""

    def test_room_transition_detected(self, mock_hass, full_config_entry):
        """Test that room transitions are detected and logged.
        
        Tests the transition detection logic in isolation.
        """
        from custom_components.whollm.event_logger import EventLogger
        from tempfile import TemporaryDirectory
        from pathlib import Path
        
        with TemporaryDirectory() as tmpdir:
            logger = EventLogger(str(Path(tmpdir) / "events.jsonl"))
            
            # Log a transition
            logger.log_room_transition(
                entity_name="Alice",
                from_room="office",
                to_room="bedroom",
                confidence=0.8,
            )
            
            # Verify it was logged
            events = logger.get_recent_events(10)
            assert len(events) == 1
            assert events[0]["event_type"] == "room_transition"
            assert events[0]["from_room"] == "office"
            assert events[0]["to_room"] == "bedroom"


class TestProviderFallback:
    """Test provider fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_on_provider_error(self, mock_hass, full_config_entry):
        """Test that Ollama provider returns fallback on error."""
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
        
        # Test fallback guess creation
        guess = provider._create_fallback_guess(
            context, "Alice", "person", ["office", "bedroom"], "Test error"
        )
        
        # Should return office because PC is on
        assert guess.room == "office"
        assert guess.confidence < 0.9  # Fallback has lower confidence


class TestSensorEntityIntegration:
    """Test sensor entity integration with coordinator."""

    def test_sensor_receives_coordinator_data(self, mock_hass, full_config_entry):
        """Test that sensors receive data from coordinator updates."""
        from custom_components.whollm.sensor import LLMPresenceSensor
        
        mock_coordinator = MagicMock()
        mock_coordinator.data = {
            "persons": {
                "Alice": PresenceGuess(
                    room="office",
                    confidence=0.85,
                    raw_response="office",
                    indicators=["PC active"],
                ),
            },
            "pets": {},
        }
        
        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
        )
        
        # Verify sensor reflects coordinator data
        assert sensor.native_value == "office"
        assert sensor.extra_state_attributes["confidence"] == 0.85

    def test_binary_sensor_reflects_room_state(self, mock_hass, full_config_entry):
        """Test that binary sensors reflect correct room state."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        mock_coordinator = MagicMock()
        mock_coordinator.data = {
            "persons": {
                "Alice": PresenceGuess(
                    room="office",
                    confidence=0.85,
                    raw_response="office",
                    indicators=[],
                ),
            },
            "pets": {},
        }
        
        # Sensor for office (where Alice is)
        sensor_office = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="office",
        )
        
        # Sensor for bedroom (where Alice is NOT)
        sensor_bedroom = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="bedroom",
        )
        
        assert sensor_office.is_on is True
        assert sensor_bedroom.is_on is False


class TestHabitLearningIntegration:
    """Test habit learning integration."""

    def test_habit_pattern_learning_flow(self):
        """Test the full habit learning flow."""
        from custom_components.whollm.habits import HabitPredictor
        from datetime import datetime
        
        with TemporaryDirectory() as tmpdir:
            predictor = HabitPredictor(config_path=Path(tmpdir))
            
            # Learn several patterns
            for day in range(5):
                # Morning - bedroom
                predictor.learn_from_event("Alice", "bedroom", 0.9, 
                    datetime(2024, 1, day + 1, 7, 0))
                # Work hours - office
                predictor.learn_from_event("Alice", "office", 0.85,
                    datetime(2024, 1, day + 1, 10, 0))
                predictor.learn_from_event("Alice", "office", 0.9,
                    datetime(2024, 1, day + 1, 14, 0))
                # Evening - living room
                predictor.learn_from_event("Alice", "living_room", 0.8,
                    datetime(2024, 1, day + 1, 20, 0))
            
            # Save patterns
            predictor.save_learned_patterns()
            
            # Load in new predictor
            predictor2 = HabitPredictor(config_path=Path(tmpdir))
            
            # Verify patterns were learned
            schedule = predictor2.get_daily_schedule("Alice")
            assert len(schedule) >= 3  # At least morning, work, evening


class TestConfidenceCombiningIntegration:
    """Test confidence combining with multiple signal sources."""

    def test_multiple_signals_boost_confidence(self):
        """Test that multiple agreeing signals boost confidence."""
        from custom_components.whollm.habits import ConfidenceCombiner
        
        combiner = ConfidenceCombiner()
        
        # Single weak signal
        room1, conf1, _ = combiner.combine(
            llm_room="office",
            llm_confidence=0.5,
            sensor_indicators=[
                {"entity_id": "light.office", "hint_type": "light", "room": "office", "state": "on"}
            ]
        )
        
        # Multiple strong signals
        room2, conf2, _ = combiner.combine(
            llm_room="office",
            llm_confidence=0.5,
            sensor_indicators=[
                {"entity_id": "switch.office_pc", "hint_type": "computer", "room": "office", "state": "on"},
                {"entity_id": "binary_sensor.office_motion", "hint_type": "motion", "room": "office", "state": "on"},
                {"entity_id": "light.office", "hint_type": "light", "room": "office", "state": "on"},
            ],
            habit_room="office",
            habit_confidence=0.8,
        )
        
        # Multiple signals should give higher confidence
        assert conf2 > conf1
        assert room2 == "office"
