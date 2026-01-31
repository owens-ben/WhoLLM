"""Tests for WhoLLM binary sensor entities."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from custom_components.whollm.const import ATTR_CONFIDENCE, DOMAIN
from custom_components.whollm.providers.base import PresenceGuess


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator with sample data."""
    coordinator = MagicMock()
    coordinator.persons = [{"name": "Alice"}, {"name": "Bob"}]
    coordinator.pets = [{"name": "Whiskers"}]
    coordinator.rooms = ["office", "bedroom", "living_room"]
    
    alice_guess = PresenceGuess(
        room="office",
        confidence=0.85,
        raw_response="office",
        indicators=["PC is active"],
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
        indicators=["Motion"],
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


class TestLLMPresenceRoomBinarySensor:
    """Tests for LLMPresenceRoomBinarySensor class."""

    def test_initialization_person(self, mock_coordinator):
        """Test binary sensor initializes correctly for person."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="office",
        )
        
        assert sensor._entity_name == "Alice"
        assert sensor._entity_type == "person"
        assert sensor._room == "office"
        assert sensor._attr_unique_id == f"{DOMAIN}_person_alice_office"
        assert sensor._attr_name == "Alice in Office"
        assert sensor._attr_icon == "mdi:account-check"

    def test_initialization_pet(self, mock_coordinator):
        """Test binary sensor initializes correctly for pet."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Whiskers",
            entity_type="pet",
            room="bedroom",
        )
        
        assert sensor._attr_icon == "mdi:paw"
        assert sensor._attr_name == "Whiskers in Bedroom"

    def test_is_on_when_in_room(self, mock_coordinator):
        """Test is_on returns True when entity is in room."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="office",
        )
        
        assert sensor.is_on is True

    def test_is_on_false_when_not_in_room(self, mock_coordinator):
        """Test is_on returns False when entity is not in room."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="bedroom",  # Alice is in office, not bedroom
        )
        
        assert sensor.is_on is False

    def test_is_on_none_when_no_data(self, mock_coordinator):
        """Test is_on returns None when no coordinator data."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        mock_coordinator.data = None
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="office",
        )
        
        assert sensor.is_on is None

    def test_is_on_none_for_unknown_entity(self, mock_coordinator):
        """Test is_on returns None for unknown entity."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Unknown",
            entity_type="person",
            room="office",
        )
        
        assert sensor.is_on is None

    def test_extra_state_attributes_when_in_room(self, mock_coordinator):
        """Test attributes when entity is in the sensor's room."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="office",
        )
        
        attrs = sensor.extra_state_attributes
        
        assert attrs[ATTR_CONFIDENCE] == 0.85  # Full confidence
        assert attrs["current_room"] == "office"
        assert attrs["entity_type"] == "person"

    def test_extra_state_attributes_when_not_in_room(self, mock_coordinator):
        """Test attributes when entity is NOT in the sensor's room."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="bedroom",  # Alice is in office
        )
        
        attrs = sensor.extra_state_attributes
        
        assert attrs[ATTR_CONFIDENCE] == 0.0  # Zero confidence for wrong room
        assert attrs["current_room"] == "office"  # Still shows actual room
        assert attrs["entity_type"] == "person"

    def test_extra_state_attributes_empty_when_no_data(self, mock_coordinator):
        """Test attributes are empty when no data."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        mock_coordinator.data = None
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="office",
        )
        
        assert sensor.extra_state_attributes == {}

    def test_pet_in_correct_room(self, mock_coordinator):
        """Test pet sensor shows correct room state."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        # Whiskers is in bedroom
        sensor_bedroom = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Whiskers",
            entity_type="pet",
            room="bedroom",
        )
        
        sensor_office = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Whiskers",
            entity_type="pet",
            room="office",
        )
        
        assert sensor_bedroom.is_on is True
        assert sensor_office.is_on is False

    def test_handle_coordinator_update(self, mock_coordinator):
        """Test coordinator update callback."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="office",
        )
        
        sensor.async_write_ha_state = MagicMock()
        sensor._handle_coordinator_update()
        
        sensor.async_write_ha_state.assert_called_once()

    def test_room_name_formatting(self, mock_coordinator):
        """Test room name formatting with underscores."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="living_room",
        )
        
        assert sensor._attr_name == "Alice in Living Room"
        assert sensor._attr_unique_id == f"{DOMAIN}_person_alice_living_room"

    def test_entity_name_with_spaces(self, mock_coordinator):
        """Test entity name with spaces is slugified."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice Smith",
            entity_type="person",
            room="office",
        )
        
        assert sensor._attr_unique_id == f"{DOMAIN}_person_alice_smith_office"


class TestBinarySensorSetupEntry:
    """Tests for binary sensor platform setup."""

    @pytest.mark.asyncio
    async def test_async_setup_entry_creates_entities(self, mock_coordinator):
        """Test that setup creates binary sensors for all person/room combinations."""
        from custom_components.whollm.binary_sensor import async_setup_entry
        
        hass = MagicMock()
        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        
        mock_entry = MagicMock()
        mock_entry.entry_id = "test_entry_id"
        
        added_entities = []
        def capture_entities(entities):
            added_entities.extend(entities)
        
        await async_setup_entry(hass, mock_entry, capture_entities)
        
        # Expected: 2 persons × 3 rooms + 1 pet × 3 rooms = 9 binary sensors
        assert len(added_entities) == 9
        
        # Verify we have sensors for each person/room combo
        unique_ids = [e._attr_unique_id for e in added_entities]
        
        # Alice in all rooms
        assert f"{DOMAIN}_person_alice_office" in unique_ids
        assert f"{DOMAIN}_person_alice_bedroom" in unique_ids
        assert f"{DOMAIN}_person_alice_living_room" in unique_ids
        
        # Bob in all rooms
        assert f"{DOMAIN}_person_bob_office" in unique_ids
        assert f"{DOMAIN}_person_bob_bedroom" in unique_ids
        assert f"{DOMAIN}_person_bob_living_room" in unique_ids
        
        # Whiskers in all rooms
        assert f"{DOMAIN}_pet_whiskers_office" in unique_ids
        assert f"{DOMAIN}_pet_whiskers_bedroom" in unique_ids
        assert f"{DOMAIN}_pet_whiskers_living_room" in unique_ids

    @pytest.mark.asyncio
    async def test_async_setup_entry_with_empty_pets(self, mock_coordinator):
        """Test setup with no pets configured."""
        from custom_components.whollm.binary_sensor import async_setup_entry
        
        mock_coordinator.pets = []  # No pets
        
        hass = MagicMock()
        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        
        mock_entry = MagicMock()
        mock_entry.entry_id = "test_entry_id"
        
        added_entities = []
        def capture_entities(entities):
            added_entities.extend(entities)
        
        await async_setup_entry(hass, mock_entry, capture_entities)
        
        # Only 2 persons × 3 rooms = 6 binary sensors
        assert len(added_entities) == 6

    @pytest.mark.asyncio
    async def test_async_setup_entry_with_single_room(self, mock_coordinator):
        """Test setup with only one room configured."""
        from custom_components.whollm.binary_sensor import async_setup_entry
        
        mock_coordinator.rooms = ["office"]  # Only one room
        mock_coordinator.pets = []
        
        hass = MagicMock()
        hass.data = {DOMAIN: {"test_entry_id": mock_coordinator}}
        
        mock_entry = MagicMock()
        mock_entry.entry_id = "test_entry_id"
        
        added_entities = []
        def capture_entities(entities):
            added_entities.extend(entities)
        
        await async_setup_entry(hass, mock_entry, capture_entities)
        
        # 2 persons × 1 room = 2 binary sensors
        assert len(added_entities) == 2


class TestBinarySensorDeviceClass:
    """Test that binary sensors have correct device class."""

    def test_device_class_is_occupancy(self, mock_coordinator):
        """Test that sensors have OCCUPANCY device class."""
        from custom_components.whollm.binary_sensor import LLMPresenceRoomBinarySensor
        from homeassistant.components.binary_sensor import BinarySensorDeviceClass
        
        sensor = LLMPresenceRoomBinarySensor(
            coordinator=mock_coordinator,
            entity_name="Alice",
            entity_type="person",
            room="office",
        )
        
        assert sensor._attr_device_class == BinarySensorDeviceClass.OCCUPANCY
