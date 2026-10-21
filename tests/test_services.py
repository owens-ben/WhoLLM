"""
Tests for Home Assistant services (TDD).

These tests verify Roomer services for manual control and data access.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestServiceRegistration:
    """Test service registration."""

    def test_services_module_exists(self):
        """Services module should exist."""
        from custom_components.roomer import services
        assert services is not None

    def test_service_constants_defined(self):
        """Service names should be defined as constants."""
        from custom_components.roomer.services import (
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
        from custom_components.roomer.services import async_handle_refresh
        
        mock_coordinator = MagicMock()
        mock_coordinator.async_request_refresh = AsyncMock()
        
        mock_hass = MagicMock()
        mock_hass.data = {"roomer": {"coordinator": mock_coordinator}}
        
        call = MagicMock()
        call.data = {}
        
        await async_handle_refresh(mock_hass, call)
        
        mock_coordinator.async_request_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_with_entity_filter(self):
        """Refresh should support entity filter."""
        from custom_components.roomer.services import async_handle_refresh
        
        mock_coordinator = MagicMock()
        mock_coordinator.async_request_refresh = AsyncMock()
        
        mock_hass = MagicMock()
        mock_hass.data = {"roomer": {"coordinator": mock_coordinator}}
        
        call = MagicMock()
        call.data = {"entity_id": "sensor.ben_room"}
        
        await async_handle_refresh(mock_hass, call)
        
        # Should still trigger refresh
        mock_coordinator.async_request_refresh.assert_called_once()


class TestCorrectRoomService:
    """Test the correct_room service."""

    @pytest.mark.asyncio
    async def test_correct_room_updates_state(self):
        """Correct room should update entity state."""
        from custom_components.roomer.services import async_handle_correct_room
        
        mock_hass = MagicMock()
        mock_hass.states.async_set = MagicMock()
        
        # Mock habit predictor
        mock_predictor = MagicMock()
        mock_predictor.learn_from_event = MagicMock()
        
        with patch("custom_components.roomer.services.get_habit_predictor", return_value=mock_predictor):
            call = MagicMock()
            call.data = {
                "entity_id": "sensor.ben_room",
                "room": "office",
            }
            
            await async_handle_correct_room(mock_hass, call)
            
            # Should learn from correction
            mock_predictor.learn_from_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_correct_room_validates_room(self):
        """Correct room should validate room name."""
        from custom_components.roomer.services import async_handle_correct_room
        
        mock_hass = MagicMock()
        
        call = MagicMock()
        call.data = {
            "entity_id": "sensor.ben_room",
            "room": "invalid_room_xyz",
        }
        
        # Should not raise, but may log warning
        await async_handle_correct_room(mock_hass, call)


class TestGetHistoryService:
    """Test the get_history service."""

    @pytest.mark.asyncio
    async def test_get_history_returns_data(self):
        """Get history should return transition data."""
        from custom_components.roomer.services import async_handle_get_history
        
        mock_tracker = MagicMock()
        mock_tracker.get_recent_transitions.return_value = [
            {"from_room": "bedroom", "to_room": "kitchen", "timestamp": "2025-01-30T08:00:00"}
        ]
        
        mock_hass = MagicMock()
        
        with patch("custom_components.roomer.services.get_history_tracker", return_value=mock_tracker):
            call = MagicMock()
            call.data = {"entity_name": "Ben", "limit": 10}
            
            result = await async_handle_get_history(mock_hass, call)

            # HA only accepts dict service responses, so the list is wrapped
            assert result["entity_name"] == "Ben"
            assert len(result["transitions"]) == 1
            assert result["transitions"][0]["to_room"] == "kitchen"

    @pytest.mark.asyncio
    async def test_get_history_respects_limit(self):
        """Get history should respect limit parameter."""
        from custom_components.roomer.services import async_handle_get_history
        
        mock_tracker = MagicMock()
        mock_tracker.get_recent_transitions.return_value = []
        
        mock_hass = MagicMock()
        
        with patch("custom_components.roomer.services.get_history_tracker", return_value=mock_tracker):
            call = MagicMock()
            call.data = {"entity_name": "Ben", "limit": 5}
            
            await async_handle_get_history(mock_hass, call)
            
            mock_tracker.get_recent_transitions.assert_called_with("Ben", limit=5)


class TestClearHistoryService:
    """Test the clear_history service."""

    @pytest.mark.asyncio
    async def test_clear_history_for_entity(self):
        """Clear history should clear for specific entity."""
        from custom_components.roomer.services import async_handle_clear_history
        
        mock_tracker = MagicMock()
        mock_tracker.clear_history = MagicMock()
        
        mock_hass = MagicMock()
        
        with patch("custom_components.roomer.services.get_history_tracker", return_value=mock_tracker):
            call = MagicMock()
            call.data = {"entity_name": "Ben"}
            
            await async_handle_clear_history(mock_hass, call)
            
            mock_tracker.clear_history.assert_called_with("Ben")

    @pytest.mark.asyncio
    async def test_clear_history_all(self):
        """Clear history should clear all when no entity specified."""
        from custom_components.roomer.services import async_handle_clear_history
        
        mock_tracker = MagicMock()
        mock_tracker.clear_history = MagicMock()
        
        mock_hass = MagicMock()
        
        with patch("custom_components.roomer.services.get_history_tracker", return_value=mock_tracker):
            call = MagicMock()
            call.data = {}
            
            await async_handle_clear_history(mock_hass, call)
            
            mock_tracker.clear_history.assert_called_with(None)


class TestGetPatternsService:
    """Test the get_patterns service."""

    @pytest.mark.asyncio
    async def test_get_patterns_returns_data(self):
        """Get patterns should return detected patterns."""
        from custom_components.roomer.services import async_handle_get_patterns
        
        mock_tracker = MagicMock()
        mock_tracker.detect_patterns.return_value = [
            {"type": "morning_routine", "description": "Usually in kitchen in the morning"}
        ]
        
        mock_hass = MagicMock()
        
        with patch("custom_components.roomer.services.get_history_tracker", return_value=mock_tracker):
            call = MagicMock()
            call.data = {"entity_name": "Ben"}
            
            result = await async_handle_get_patterns(mock_hass, call)

            assert result["entity_name"] == "Ben"
            assert len(result["patterns"]) == 1
            assert result["patterns"][0]["type"] == "morning_routine"


class TestServiceSchemas:
    """Test service schema validation."""

    def test_correct_room_schema(self):
        """Correct room schema should require entity_id and room."""
        from custom_components.roomer.services import CORRECT_ROOM_SCHEMA
        import voluptuous as vol
        
        # Valid data should pass
        valid_data = {"entity_id": "sensor.ben_room", "room": "office"}
        CORRECT_ROOM_SCHEMA(valid_data)
        
        # Missing room should fail
        with pytest.raises(vol.Invalid):
            CORRECT_ROOM_SCHEMA({"entity_id": "sensor.ben_room"})

    def test_get_history_schema(self):
        """Get history schema should require entity_name."""
        from custom_components.roomer.services import GET_HISTORY_SCHEMA
        import voluptuous as vol
        
        # Valid data should pass
        valid_data = {"entity_name": "Ben"}
        GET_HISTORY_SCHEMA(valid_data)
        
        # With optional limit
        GET_HISTORY_SCHEMA({"entity_name": "Ben", "limit": 20})

    def test_get_history_limit_range(self):
        """Limit must stay within what services.yaml advertises (1-100)."""
        from custom_components.roomer.services import GET_HISTORY_SCHEMA
        import voluptuous as vol

        assert GET_HISTORY_SCHEMA({"entity_name": "Ben"})["limit"] == 20
        GET_HISTORY_SCHEMA({"entity_name": "Ben", "limit": 1})
        GET_HISTORY_SCHEMA({"entity_name": "Ben", "limit": 100})

        # limit=0 used to slice as [-0:] and return the whole history
        for bad in (0, -5, 101):
            with pytest.raises(vol.Invalid):
                GET_HISTORY_SCHEMA({"entity_name": "Ben", "limit": bad})


class TestServiceSetup:
    """Test async_setup_services registration."""

    @staticmethod
    def _mock_hass(already_registered=False):
        mock_hass = MagicMock()
        mock_hass.services.has_service = MagicMock(return_value=already_registered)
        mock_hass.services.async_register = MagicMock()
        return mock_hass

    @staticmethod
    def _registrations(mock_hass):
        """Map service name -> kwargs it was registered with."""
        return {c.args[1]: c.kwargs for c in mock_hass.services.async_register.call_args_list}

    @pytest.mark.asyncio
    async def test_registers_all_services(self):
        from custom_components.roomer.services import async_setup_services

        mock_hass = self._mock_hass()
        await async_setup_services(mock_hass)

        assert set(self._registrations(mock_hass)) == {
            "refresh", "correct_room", "get_history", "clear_history", "get_patterns",
        }

    @pytest.mark.asyncio
    async def test_history_and_patterns_are_response_only(self):
        """Without supports_response HA drops the handler's return value."""
        from homeassistant.core import SupportsResponse
        from custom_components.roomer.services import async_setup_services

        mock_hass = self._mock_hass()
        await async_setup_services(mock_hass)
        registered = self._registrations(mock_hass)

        assert registered["get_history"]["supports_response"] == SupportsResponse.ONLY
        assert registered["get_patterns"]["supports_response"] == SupportsResponse.ONLY
        for name in ("refresh", "correct_room", "clear_history"):
            assert "supports_response" not in registered[name]

    @pytest.mark.asyncio
    async def test_setup_is_idempotent(self):
        """A second config entry or a reload must not re-register."""
        from custom_components.roomer.services import async_setup_services

        mock_hass = self._mock_hass(already_registered=True)
        await async_setup_services(mock_hass)

        mock_hass.services.async_register.assert_not_called()


class TestServiceResponsesWithRealHass:
    """Call the services through a real Home Assistant core, as a client would."""

    @pytest.mark.asyncio
    async def test_get_history_and_patterns_via_service_call(self, hass, tmp_path):
        """return_response yields the dicts, and the data is JSON-serializable."""
        import json

        from custom_components.roomer.const import DOMAIN
        from custom_components.roomer.history import HistoryTracker
        from custom_components.roomer.services import HISTORY_TRACKER_KEY, async_setup_services

        tracker = HistoryTracker(config_path=tmp_path)
        tracker.record_transition("Ben", "bedroom", "kitchen", 0.8)
        tracker.record_transition("Ben", "kitchen", "office", 0.9)
        hass.data.setdefault(DOMAIN, {})[HISTORY_TRACKER_KEY] = tracker

        await async_setup_services(hass)

        history = await hass.services.async_call(
            DOMAIN, "get_history", {"entity_name": "Ben", "limit": 1},
            blocking=True, return_response=True,
        )
        assert history["entity_name"] == "Ben"
        assert [t["to_room"] for t in history["transitions"]] == ["office"]
        json.dumps(history)  # what the WebSocket API will do to it

        patterns = await hass.services.async_call(
            DOMAIN, "get_patterns", {"entity_name": "Ben"},
            blocking=True, return_response=True,
        )
        assert patterns == {"entity_name": "Ben", "patterns": []}  # < 10 transitions

    @pytest.mark.asyncio
    async def test_response_only_services_reject_calls_without_return_response(self, hass, tmp_path):
        """get_history has no side effect, so HA refuses a call that ignores the response."""
        from homeassistant.exceptions import ServiceValidationError

        from custom_components.roomer.const import DOMAIN
        from custom_components.roomer.services import async_setup_services

        await async_setup_services(hass)

        with pytest.raises(ServiceValidationError):
            await hass.services.async_call(
                DOMAIN, "get_history", {"entity_name": "Ben"}, blocking=True,
            )

    @pytest.mark.asyncio
    async def test_unknown_person_returns_empty_not_error(self, hass, tmp_path):
        """A name with no history is a normal empty timeline."""
        from custom_components.roomer.const import DOMAIN
        from custom_components.roomer.history import HistoryTracker
        from custom_components.roomer.services import HISTORY_TRACKER_KEY, async_setup_services

        hass.data.setdefault(DOMAIN, {})[HISTORY_TRACKER_KEY] = HistoryTracker(config_path=tmp_path)
        await async_setup_services(hass)

        history = await hass.services.async_call(
            DOMAIN, "get_history", {"entity_name": "Nobody"},
            blocking=True, return_response=True,
        )
        assert history == {"entity_name": "Nobody", "transitions": []}
