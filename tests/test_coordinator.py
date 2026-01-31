"""Tests for the WhoLLM coordinator."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import timedelta

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
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_URL,
    ENTITY_HINT_COMPUTER,
    ENTITY_HINT_MEDIA,
    ENTITY_HINT_MOTION,
    VALID_ROOMS,
)


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


class TestCoordinatorHelpers:
    """Test coordinator helper methods without full HA setup."""

    def test_is_entity_active_on(self):
        """Test entity active detection for 'on' state."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)

            assert coordinator._is_entity_active("light.test", "on", "light") is True
            assert coordinator._is_entity_active("light.test", "off", "light") is False

    def test_is_entity_active_motion(self):
        """Test entity active detection for motion sensors."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)

            assert coordinator._is_entity_active("binary_sensor.motion", "on", ENTITY_HINT_MOTION) is True
            assert coordinator._is_entity_active("binary_sensor.motion", "off", ENTITY_HINT_MOTION) is False

    def test_is_entity_active_media_playing(self):
        """Test entity active detection for media players."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)

            assert coordinator._is_entity_active("media_player.tv", "playing", ENTITY_HINT_MEDIA) is True
            assert coordinator._is_entity_active("media_player.tv", "paused", ENTITY_HINT_MEDIA) is True
            assert coordinator._is_entity_active("media_player.tv", "off", ENTITY_HINT_MEDIA) is False
            assert coordinator._is_entity_active("media_player.tv", "idle", ENTITY_HINT_MEDIA) is False

    def test_is_entity_active_device_tracker(self):
        """Test entity active detection for device trackers."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)

            assert coordinator._is_entity_active("device_tracker.phone", "home", "presence") is True
            assert coordinator._is_entity_active("device_tracker.phone", "not_home", "presence") is False


class TestCoordinatorRoomEntityMapping:
    """Test room-entity mapping functionality."""

    def test_find_entity_room(self, mock_hass, mock_config_entry):
        """Test finding which room an entity belongs to."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)
            coordinator.room_entities = mock_config_entry.data[CONF_ROOM_ENTITIES]

            assert coordinator._find_entity_room("switch.office_pc") == "office"
            assert coordinator._find_entity_room("media_player.living_room_tv") == "living_room"
            assert coordinator._find_entity_room("unknown.entity") is None

    def test_get_active_indicators(self, mock_hass, mock_config_entry):
        """Test getting active indicators from room entities."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)
            coordinator.room_entities = mock_config_entry.data[CONF_ROOM_ENTITIES]
            coordinator.hass = mock_hass

            # Mock _get_entity_state and _is_entity_active
            def mock_get_state(entity_id, context):
                states = {
                    "switch.office_pc": "on",
                    "binary_sensor.office_motion": "off",
                    "media_player.living_room_tv": "playing",
                }
                return states.get(entity_id, "unknown")

            coordinator._get_entity_state = mock_get_state

            context = {}
            active = coordinator._get_active_indicators(context)

            # Should have PC (on) and TV (playing), not motion (off)
            entity_ids = [i["entity_id"] for i in active]
            assert "switch.office_pc" in entity_ids
            assert "media_player.living_room_tv" in entity_ids
            assert "binary_sensor.office_motion" not in entity_ids


class TestCoordinatorPersonDevices:
    """Test person-device ownership mapping."""

    def test_get_person_indicators_with_owned_device(self, mock_hass, mock_config_entry):
        """Test that person's owned devices are included in indicators."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)
            coordinator.room_entities = mock_config_entry.data[CONF_ROOM_ENTITIES]
            coordinator.person_devices = mock_config_entry.data[CONF_PERSON_DEVICES]
            coordinator.hass = mock_hass

            # Mock methods
            def mock_get_state(entity_id, context):
                return "on" if entity_id == "switch.alice_pc" else "off"

            def mock_find_room(entity_id):
                return "office" if "pc" in entity_id else None

            coordinator._get_entity_state = mock_get_state
            coordinator._find_entity_room = mock_find_room

            all_indicators = []
            context = {}

            indicators = coordinator._get_person_indicators("Alice", all_indicators, context)

            # Alice's PC should be in indicators with owned_by field
            alice_pc_indicator = next((i for i in indicators if i.get("entity_id") == "switch.alice_pc"), None)
            assert alice_pc_indicator is not None
            assert alice_pc_indicator.get("owned_by") == "Alice"
            assert alice_pc_indicator.get("room") == "office"

    def test_get_person_indicators_without_owned_device(self, mock_hass, mock_config_entry):
        """Test indicators for person without device ownership configured."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)
            coordinator.room_entities = mock_config_entry.data[CONF_ROOM_ENTITIES]
            coordinator.person_devices = mock_config_entry.data[CONF_PERSON_DEVICES]
            coordinator.hass = mock_hass

            coordinator._get_entity_state = lambda e, c: "off"
            coordinator._find_entity_room = lambda e: None

            all_indicators = [
                {"entity_id": "binary_sensor.motion", "hint_type": "motion", "room": "office", "state": "on"}
            ]
            context = {}

            # Bob doesn't have device ownership configured
            indicators = coordinator._get_person_indicators("Bob", all_indicators, context)

            # Should still get general indicators
            assert len(indicators) >= 1
            assert indicators[0]["entity_id"] == "binary_sensor.motion"


class TestCoordinatorRoomTransition:
    """Test room transition detection."""

    def test_check_room_transition_logs_movement(self, mock_hass, mock_config_entry):
        """Test that room transitions are detected."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)
            coordinator._previous_rooms = {"Alice": "bedroom"}
            coordinator.event_logger = MagicMock()

            # Move Alice to office
            coordinator._check_room_transition("Alice", "office", 0.9)

            # Should have logged the transition
            coordinator.event_logger.log_room_transition.assert_called_once_with(
                entity_name="Alice",
                from_room="bedroom",
                to_room="office",
                confidence=0.9,
            )

            # Previous room should be updated
            assert coordinator._previous_rooms["Alice"] == "office"

    def test_check_room_transition_same_room_no_log(self, mock_hass, mock_config_entry):
        """Test that staying in same room doesn't log transition."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)
            coordinator._previous_rooms = {"Alice": "office"}
            coordinator.event_logger = MagicMock()

            # Alice stays in office
            coordinator._check_room_transition("Alice", "office", 0.9)

            # Should NOT have logged a transition
            coordinator.event_logger.log_room_transition.assert_not_called()

    def test_check_room_transition_to_unknown_no_log(self, mock_hass, mock_config_entry):
        """Test that transition to 'unknown' doesn't log."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator

        with patch.object(LLMPresenceCoordinator, "__init__", lambda x, y, z: None):
            coordinator = LLMPresenceCoordinator.__new__(LLMPresenceCoordinator)
            coordinator._previous_rooms = {"Alice": "office"}
            coordinator.event_logger = MagicMock()

            # Unknown room result
            coordinator._check_room_transition("Alice", "unknown", 0.3)

            # Should NOT log transition to unknown
            coordinator.event_logger.log_room_transition.assert_not_called()
