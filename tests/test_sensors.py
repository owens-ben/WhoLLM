"""Tests for WhoLLM sensor entities."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from custom_components.whollm.const import (
    ATTR_CONFIDENCE,
    ATTR_INDICATORS,
    ATTR_RAW_RESPONSE,
    DOMAIN,
)
from custom_components.whollm.providers.base import PresenceGuess


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator with sample data."""
    coordinator = MagicMock()
    coordinator.persons = [{"name": "Alice"}, {"name": "Bob"}]
    coordinator.pets = [{"name": "Whiskers"}]
    coordinator.rooms = ["office", "bedroom", "living_room"]

    # Create mock presence guesses
    alice_guess = PresenceGuess(
        room="office",
        confidence=0.85,
        raw_response="office",
        indicators=["PC is active", "Motion detected"],
    )
    bob_guess = PresenceGuess(
        room="living_room",
        confidence=0.7,
        raw_response="living_room",
        indicators=["TV playing"],
    )
    whiskers_guess = PresenceGuess(
        room="bedroom",
        confidence=0.6,
        raw_response="bedroom",
        indicators=["Motion in bedroom"],
    )

    coordinator.data = {
        "persons": {
            "Alice": alice_guess,
            "Bob": bob_guess,
        },
        "pets": {
            "Whiskers": whiskers_guess,
        },
    }

    return coordinator


class TestLLMPresenceSensor:
    """Tests for LLMPresenceSensor class."""

    def test_sensor_initialization(self, mock_coordinator):
        """Test sensor initializes with correct attributes."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
        )

        assert sensor._entity_name == "Alice"
        assert sensor._entity_type == "person"
        assert sensor._attr_unique_id == f"{DOMAIN}_person_alice_room"
        assert sensor._attr_name == "Alice Room"
        assert sensor._attr_icon == "mdi:account"

    def test_sensor_pet_icon(self, mock_coordinator):
        """Test that pet sensors have paw icon."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Whiskers",
            entity_type="pet",
        )

        assert sensor._attr_icon == "mdi:paw"

    def test_native_value_returns_room(self, mock_coordinator):
        """Test that native_value returns the current room."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
        )

        assert sensor.native_value == "office"

    def test_native_value_with_no_data(self, mock_coordinator):
        """Test native_value returns None when no data available."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        mock_coordinator.data = None

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
        )

        assert sensor.native_value is None

    def test_native_value_unknown_person(self, mock_coordinator):
        """Test native_value returns None for unknown person."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Unknown",
            entity_type="person",
        )

        assert sensor.native_value is None

    def test_extra_state_attributes(self, mock_coordinator):
        """Test extra state attributes are populated correctly."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
        )

        attrs = sensor.extra_state_attributes

        assert attrs[ATTR_CONFIDENCE] == 0.85
        assert attrs[ATTR_RAW_RESPONSE] == "office"
        assert "PC is active" in attrs[ATTR_INDICATORS]
        assert attrs["entity_type"] == "person"

    def test_extra_state_attributes_empty_when_no_data(self, mock_coordinator):
        """Test attributes are empty when no data."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        mock_coordinator.data = None

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
        )

        assert sensor.extra_state_attributes == {}

    def test_pet_sensor_native_value(self, mock_coordinator):
        """Test pet sensor returns correct room."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Whiskers",
            entity_type="pet",
        )

        assert sensor.native_value == "bedroom"

    def test_handle_coordinator_update_callback(self, mock_coordinator):
        """Test that coordinator update callback triggers state write."""
        from custom_components.whollm.sensor import LLMPresenceSensor

        sensor = LLMPresenceSensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
        )

        # Mock async_write_ha_state
        sensor.async_write_ha_state = MagicMock()

        # Call the callback
        sensor._handle_coordinator_update()

        sensor.async_write_ha_state.assert_called_once()


class TestLLMVisionSensor:
    """Tests for LLMVisionSensor class."""

    def test_vision_sensor_initialization(self, mock_coordinator):
        """Test vision sensor initializes correctly."""
        from custom_components.whollm.sensor import LLMVisionSensor

        hass = MagicMock()
        sensor = LLMVisionSensor(hass, mock_coordinator)

        assert sensor._attr_unique_id == f"{DOMAIN}_vision_last_identification"
        assert sensor._attr_name == "Vision Last Identification"
        assert sensor._attr_icon == "mdi:camera-iris"
        assert sensor._identified == "none"

    def test_vision_sensor_native_value(self, mock_coordinator):
        """Test vision sensor native_value."""
        from custom_components.whollm.sensor import LLMVisionSensor

        hass = MagicMock()
        sensor = LLMVisionSensor(hass, mock_coordinator)

        assert sensor.native_value == "none"

        # Simulate identification
        sensor._identified = "Alice"
        assert sensor.native_value == "Alice"

    def test_vision_sensor_attributes(self, mock_coordinator):
        """Test vision sensor extra state attributes."""
        from custom_components.whollm.sensor import LLMVisionSensor

        hass = MagicMock()
        sensor = LLMVisionSensor(hass, mock_coordinator)

        # Set values
        sensor._identified = "Alice"
        sensor._confidence = "high"
        sensor._description = "Person wearing blue shirt"
        sensor._camera = "camera.living_room"
        sensor._detection_type = "person"
        sensor._timestamp = "2024-01-15T14:30:00"
        sensor._raw_response = "PERSON: Alice"

        attrs = sensor.extra_state_attributes

        assert attrs["confidence"] == "high"
        assert attrs["description"] == "Person wearing blue shirt"
        assert attrs["camera"] == "camera.living_room"
        assert attrs["detection_type"] == "person"
        assert attrs["timestamp"] == "2024-01-15T14:30:00"
        assert attrs["raw_response"] == "PERSON: Alice"

    @pytest.mark.asyncio
    async def test_vision_sensor_event_listener(self, mock_coordinator):
        """Test that vision sensor listens for events."""
        from custom_components.whollm.sensor import LLMVisionSensor

        hass = MagicMock()
        hass.bus = MagicMock()
        hass.bus.async_listen = MagicMock(return_value=MagicMock())

        sensor = LLMVisionSensor(hass, mock_coordinator)
        sensor.async_write_ha_state = MagicMock()

        # Call async_added_to_hass
        await sensor.async_added_to_hass()

        # Verify event listener was registered
        hass.bus.async_listen.assert_called_once()
        call_args = hass.bus.async_listen.call_args
        assert call_args[0][0] == f"{DOMAIN}_vision_identification"

    @pytest.mark.asyncio
    async def test_vision_sensor_event_handler(self, mock_coordinator):
        """Test vision sensor event handler updates state."""
        from custom_components.whollm.sensor import LLMVisionSensor

        hass = MagicMock()
        hass.bus = MagicMock()

        # Capture the event handler
        event_handler = None

        def capture_handler(event_name, handler):
            nonlocal event_handler
            event_handler = handler
            return MagicMock()

        hass.bus.async_listen = capture_handler

        sensor = LLMVisionSensor(hass, mock_coordinator)
        sensor.async_write_ha_state = MagicMock()

        await sensor.async_added_to_hass()

        # Simulate event
        mock_event = MagicMock()
        mock_event.data = {
            "identified": "Bob",
            "confidence": "medium",
            "description": "Person at desk",
            "camera": "camera.office",
            "detection_type": "person",
            "raw_response": "PERSON: Bob",
        }

        event_handler(mock_event)

        assert sensor._identified == "Bob"
        assert sensor._confidence == "medium"
        assert sensor._description == "Person at desk"
        sensor.async_write_ha_state.assert_called()

    @pytest.mark.asyncio
    async def test_vision_sensor_unsubscribe_on_remove(self, mock_coordinator):
        """Test that event listener is unsubscribed on removal."""
        from custom_components.whollm.sensor import LLMVisionSensor

        hass = MagicMock()
        mock_unsub = MagicMock()
        hass.bus.async_listen = MagicMock(return_value=mock_unsub)

        sensor = LLMVisionSensor(hass, mock_coordinator)

        await sensor.async_added_to_hass()
        await sensor.async_will_remove_from_hass()

        mock_unsub.assert_called_once()


class TestSensorSetupEntry:
    """Tests for sensor platform setup."""

    @pytest.mark.asyncio
    async def test_async_setup_entry_creates_entities(self, mock_coordinator):
        """Test that setup entry creates sensor entities."""
        from custom_components.whollm.sensor import async_setup_entry

        hass = MagicMock()
        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}

        mock_entry = MagicMock()
        mock_entry.entry_id = "test_entry_id"

        added_entities = []

        def capture_entities(entities):
            added_entities.extend(entities)

        await async_setup_entry(hass, mock_entry, capture_entities)

        # Should have: Alice sensor, Bob sensor, Whiskers sensor, Vision sensor
        assert len(added_entities) == 4

        # Verify entity types
        entity_names = [e._entity_name for e in added_entities if hasattr(e, "_entity_name")]
        assert "Alice" in entity_names
        assert "Bob" in entity_names
        assert "Whiskers" in entity_names
