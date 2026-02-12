"""Shared fixtures for WhoLLM tests."""

from __future__ import annotations

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, AsyncMock

from custom_components.whollm.habits import HabitPredictor, ConfidenceCombiner
from custom_components.whollm.providers.base import PresenceGuess
from custom_components.whollm.const import (
    CONF_MODEL,
    CONF_PERSON_DEVICES,
    CONF_PERSONS,
    CONF_PETS,
    CONF_POLL_INTERVAL,
    CONF_PROVIDER,
    CONF_ROOM_ENTITIES,
    CONF_ROOMS,
    CONF_URL,
    DEFAULT_CONFIDENCE_WEIGHTS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_URL,
    ENTITY_HINT_COMPUTER,
    ENTITY_HINT_MEDIA,
    ENTITY_HINT_MOTION,
)


@pytest.fixture
def temp_config_dir():
    """Provide a temporary directory for config files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def habit_predictor(temp_config_dir):
    """Provide a fresh HabitPredictor instance."""
    return HabitPredictor(config_path=temp_config_dir)


@pytest.fixture
def confidence_combiner():
    """Provide a ConfidenceCombiner with default weights."""
    return ConfidenceCombiner(weights=DEFAULT_CONFIDENCE_WEIGHTS.copy())


@pytest.fixture
def sample_presence_guess():
    """Provide a sample PresenceGuess for testing."""
    return PresenceGuess(
        room="office",
        confidence=0.8,
        raw_response="office",
        indicators=["PC is active", "Motion detected"],
    )


@pytest.fixture
def sample_context():
    """Provide sample sensor context for testing."""
    return {
        "lights": {
            "light.office": {"state": "on", "brightness": 200, "last_changed": "5m ago"},
            "light.bedroom": {"state": "off", "last_changed": "2h ago"},
        },
        "motion": {
            "binary_sensor.office_motion": {"state": "on", "last_changed": "1m ago"},
            "binary_sensor.bedroom_motion": {"state": "off", "last_changed": "3h ago"},
        },
        "media": {
            "media_player.living_room_tv": {"state": "off", "last_changed": "1h ago"},
        },
        "device_trackers": {
            "device_tracker.alice_phone": {"state": "home"},
        },
        "computers": {
            "switch.office_pc": {"state": "on", "last_changed": "30m ago"},
        },
        "ai_detection": {},
        "doors": {},
        "time_context": {
            "current_time": "14:30",
            "day_of_week": "Tuesday",
            "is_night": False,
            "is_morning": False,
            "is_evening": False,
        },
        "configured_rooms": ["office", "bedroom", "living_room", "kitchen", "bathroom"],
    }


@pytest.fixture
def sample_indicators():
    """Provide sample sensor indicators for confidence combining."""
    return [
        {"entity_id": "switch.office_pc", "hint_type": "computer", "room": "office", "state": "on"},
        {"entity_id": "binary_sensor.office_motion", "hint_type": "motion", "room": "office", "state": "on"},
        {"entity_id": "light.office", "hint_type": "light", "room": "office", "state": "on"},
    ]


@pytest.fixture
def mock_hass():
    """Provide a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.states.async_all = MagicMock(return_value=[])
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()
    return hass


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator with sample data."""
    coordinator = MagicMock()
    coordinator.persons = [{"name": "Alice"}, {"name": "Bob"}]
    coordinator.pets = [{"name": "Whiskers"}]
    coordinator.rooms = ["office", "bedroom", "living_room"]
    coordinator.data = {
        "persons": {
            "Alice": PresenceGuess(
                room="office",
                confidence=0.85,
                raw_response="office",
                indicators=["PC is active"],
            ),
            "Bob": PresenceGuess(
                room="living_room",
                confidence=0.7,
                raw_response="living_room",
                indicators=["TV playing"],
            ),
        },
        "pets": {
            "Whiskers": PresenceGuess(
                room="bedroom",
                confidence=0.6,
                raw_response="bedroom",
                indicators=["Motion"],
            ),
        },
    }
    return coordinator


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry."""
    entry = MagicMock()
    entry.data = {
        CONF_PROVIDER: DEFAULT_PROVIDER,
        CONF_URL: DEFAULT_URL,
        CONF_MODEL: DEFAULT_MODEL,
        CONF_POLL_INTERVAL: 30,
        CONF_PERSONS: [{"name": "Alice"}, {"name": "Bob"}],
        CONF_PETS: [{"name": "Whiskers"}],
        CONF_ROOMS: ["office", "bedroom", "living_room"],
        CONF_ROOM_ENTITIES: {
            "office": [
                {"entity_id": "switch.office_pc", "hint_type": ENTITY_HINT_COMPUTER},
                {"entity_id": "binary_sensor.office_motion", "hint_type": ENTITY_HINT_MOTION},
            ],
            "living_room": [
                {"entity_id": "media_player.living_room_tv", "hint_type": ENTITY_HINT_MEDIA},
            ],
        },
        CONF_PERSON_DEVICES: {
            "Alice": ["switch.alice_pc"],
        },
    }
    entry.entry_id = "test_entry_id"
    entry.options = {}
    return entry
