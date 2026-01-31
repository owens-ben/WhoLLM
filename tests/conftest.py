"""Shared fixtures for WhoLLM tests."""

from __future__ import annotations

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, AsyncMock

from custom_components.whollm.habits import HabitPredictor, ConfidenceCombiner
from custom_components.whollm.providers.base import PresenceGuess
from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS


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
