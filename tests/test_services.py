"""
Tests for Home Assistant services (TDD).

These tests verify WhoLLM services for manual control and data access.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestServiceRegistration:
    """Test service registration."""

    def test_services_module_exists(self):
        """Services module should exist."""
        from custom_components.whollm import services
        assert services is not None

    def test_service_constants_defined(self):
        """Service names should be defined as constants."""
        from custom_components.whollm.services import (
            SERVICE_REFRESH,
            SERVICE_CORRECT_ROOM,
            SERVICE_GET_HISTORY,
            SERVICE_CLEAR_HISTORY,
            SERVICE_GET_PATTERNS,
        )
        
        assert SERVICE_REFRESH == "refresh"
        assert SERVICE_CORRECT_ROOM == "correct_room"
        assert SERVICE_GET_HISTORY == "get_history"
        assert SERVICE_CLEAR_HISTORY == "clear_history"
        assert SERVICE_GET_PATTERNS == "get_patterns"


class TestRefreshService:
    """Test the refresh service."""

    @pytest.mark.asyncio
    async def test_refresh_triggers_update(self):
        """Refresh should trigger coordinator update."""
        from custom_components.whollm.services import async_handle_refresh
        
        mock_coordinator = MagicMock()
        mock_coordinator.async_request_refresh = AsyncMock()
        
        mock_hass = MagicMock()
        mock_hass.data = {"whollm": {"coordinator": mock_coordinator}}
        
        call = MagicMock()
        call.data = {}
        
        await async_handle_refresh(mock_hass, call)
        
        mock_coordinator.async_request_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_with_entity_filter(self):
        """Refresh should support entity filter."""
        from custom_components.whollm.services import async_handle_refresh
        
        mock_coordinator = MagicMock()
        mock_coordinator.async_request_refresh = AsyncMock()
        
        mock_hass = MagicMock()
        mock_hass.data = {"whollm": {"coordinator": mock_coordinator}}
        
        call = MagicMock()
        call.data = {"entity_id": "sensor.ben_room"}
        
        await async_handle_refresh(mock_hass, call)
        
        # Should still trigger refresh
        mock_coordinator.async_request_refresh.assert_called_once()


def _hass_with_coordinator(coordinator=None):
    """Mock hass whose whollm data holds one coordinator, keyed by entry id like the real setup."""
    if coordinator is None:
        coordinator = MagicMock()
        coordinator.persons = [{"name": "Ben"}, {"name": "Mary Jane"}]
        coordinator.pets = [{"name": "Whiskers"}]
        coordinator.rooms = ["office", "kitchen", "living_room", "home-office"]
    hass = MagicMock()
    hass.data = {"whollm": {"entry_1": coordinator, "event_logger": object()}}
    return hass, coordinator


class TestCorrectRoomService:
    """Test the correct_room service."""

    @pytest.mark.asyncio
    async def test_correct_room_updates_state(self):
        """Correct room should learn from the correction and update entity state."""
        from custom_components.whollm.services import async_handle_correct_room

        mock_hass, coordinator = _hass_with_coordinator()
        mock_hass.states.get.return_value = MagicMock(attributes={"confidence": 0.4})

        call = MagicMock()
        call.data = {"entity_id": "sensor.ben_room", "room": "office"}

        await async_handle_correct_room(mock_hass, call)

        coordinator.habit_predictor.learn_from_event.assert_called_once_with(
            entity_name="Ben", room="office", confidence=1.0,
        )
        entity_id, state, attrs = mock_hass.states.async_set.call_args[0]
        assert (entity_id, state) == ("sensor.ben_room", "office")
        assert attrs["source"] == "manual_correction"

    @pytest.mark.asyncio
    async def test_correct_room_maps_multiword_name(self):
        """sensor.mary_jane_room should map back to the configured name 'Mary Jane'."""
        from custom_components.whollm.services import async_handle_correct_room

        mock_hass, coordinator = _hass_with_coordinator()

        call = MagicMock()
        call.data = {"entity_id": "sensor.mary_jane_room", "room": "kitchen"}

        await async_handle_correct_room(mock_hass, call)

        assert coordinator.habit_predictor.learn_from_event.call_args.kwargs["entity_name"] == "Mary Jane"

    @pytest.mark.asyncio
    async def test_correct_room_validates_room(self):
        """An unknown room is rejected before anything is learned or changed."""
        from homeassistant.exceptions import ServiceValidationError

        from custom_components.whollm.services import async_handle_correct_room

        mock_hass, coordinator = _hass_with_coordinator()

        call = MagicMock()
        call.data = {
            "entity_id": "sensor.ben_room",
            "room": "invalid_room_xyz",
        }

        with pytest.raises(ServiceValidationError):
            await async_handle_correct_room(mock_hass, call)

        coordinator.habit_predictor.learn_from_event.assert_not_called()
        mock_hass.states.async_set.assert_not_called()

    @pytest.mark.asyncio
    async def test_correct_room_rejects_unknown_entity(self):
        """A sensor that isn't a configured person or pet is rejected."""
        from homeassistant.exceptions import ServiceValidationError

        from custom_components.whollm.services import async_handle_correct_room

        mock_hass, coordinator = _hass_with_coordinator()

        call = MagicMock()
        call.data = {"entity_id": "sensor.stranger_room", "room": "office"}

        with pytest.raises(ServiceValidationError):
            await async_handle_correct_room(mock_hass, call)

        coordinator.habit_predictor.learn_from_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_correct_room_accepts_display_name_and_away(self):
        """'Living Room' maps to living_room, configured spellings are kept, and away is always valid."""
        from custom_components.whollm.services import async_handle_correct_room

        mock_hass, coordinator = _hass_with_coordinator()

        for room, expected in (("Living Room", "living_room"), ("away", "away"), ("home-office", "home-office")):
            call = MagicMock()
            call.data = {"entity_id": "sensor.ben_room", "room": room}
            await async_handle_correct_room(mock_hass, call)
            assert coordinator.habit_predictor.learn_from_event.call_args.kwargs["room"] == expected

    @pytest.mark.asyncio
    async def test_correct_room_without_coordinator(self):
        """No loaded entry should be a no-op, not a crash."""
        from custom_components.whollm.services import async_handle_correct_room

        mock_hass = MagicMock()
        mock_hass.data = {}

        call = MagicMock()
        call.data = {"entity_id": "sensor.ben_room", "room": "office"}

        await async_handle_correct_room(mock_hass, call)

        mock_hass.states.async_set.assert_not_called()


class TestGetHistoryService:
    """Test the get_history service."""

    @pytest.mark.asyncio
    async def test_get_history_returns_data(self):
        """Get history should return transition data."""
        from custom_components.whollm.services import async_handle_get_history

        mock_hass, coordinator = _hass_with_coordinator()
        coordinator.history_tracker.get_recent_transitions.return_value = [
            {"from_room": "bedroom", "to_room": "kitchen", "timestamp": "2025-01-30T08:00:00"}
        ]

        call = MagicMock()
        call.data = {"entity_name": "Ben", "limit": 10}

        result = await async_handle_get_history(mock_hass, call)

        assert len(result["transitions"]) == 1
        assert result["transitions"][0]["to_room"] == "kitchen"

    @pytest.mark.asyncio
    async def test_get_history_respects_limit(self):
        """Get history should respect limit parameter."""
        from custom_components.whollm.services import async_handle_get_history

        mock_hass, coordinator = _hass_with_coordinator()
        coordinator.history_tracker.get_recent_transitions.return_value = []

        call = MagicMock()
        call.data = {"entity_name": "Ben", "limit": 5}

        await async_handle_get_history(mock_hass, call)

        coordinator.history_tracker.get_recent_transitions.assert_called_with("Ben", limit=5)


class TestClearHistoryService:
    """Test the clear_history service."""

    @pytest.mark.asyncio
    async def test_clear_history_for_entity(self):
        """Clear history should clear for specific entity."""
        from custom_components.whollm.services import async_handle_clear_history

        mock_hass, coordinator = _hass_with_coordinator()

        call = MagicMock()
        call.data = {"entity_name": "Ben"}

        await async_handle_clear_history(mock_hass, call)

        coordinator.history_tracker.clear_history.assert_called_with("Ben")

    @pytest.mark.asyncio
    async def test_clear_history_all(self):
        """Clear history should clear all when no entity specified."""
        from custom_components.whollm.services import async_handle_clear_history

        mock_hass, coordinator = _hass_with_coordinator()

        call = MagicMock()
        call.data = {}

        await async_handle_clear_history(mock_hass, call)

        coordinator.history_tracker.clear_history.assert_called_with(None)


class TestGetPatternsService:
    """Test the get_patterns service."""

    @pytest.mark.asyncio
    async def test_get_patterns_returns_data(self):
        """Get patterns should return detected patterns."""
        from custom_components.whollm.services import async_handle_get_patterns

        mock_hass, coordinator = _hass_with_coordinator()
        coordinator.history_tracker.detect_patterns.return_value = [
            {"type": "morning_routine", "description": "Usually in kitchen in the morning"}
        ]

        call = MagicMock()
        call.data = {"entity_name": "Ben"}

        result = await async_handle_get_patterns(mock_hass, call)

        assert len(result["patterns"]) == 1
        assert result["patterns"][0]["type"] == "morning_routine"


class TestServiceSchemas:
    """Test service schema validation."""

    def test_correct_room_schema(self):
        """Correct room schema should require entity_id and room."""
        from custom_components.whollm.services import CORRECT_ROOM_SCHEMA
        import voluptuous as vol
        
        # Valid data should pass
        valid_data = {"entity_id": "sensor.ben_room", "room": "office"}
        CORRECT_ROOM_SCHEMA(valid_data)
        
        # Missing room should fail
        with pytest.raises(vol.Invalid):
            CORRECT_ROOM_SCHEMA({"entity_id": "sensor.ben_room"})

    def test_get_history_schema(self):
        """Get history schema should require entity_name."""
        from custom_components.whollm.services import GET_HISTORY_SCHEMA
        import voluptuous as vol
        
        # Valid data should pass
        valid_data = {"entity_name": "Ben"}
        GET_HISTORY_SCHEMA(valid_data)
        
        # With optional limit
        GET_HISTORY_SCHEMA({"entity_name": "Ben", "limit": 20})
