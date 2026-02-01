"""
Tests for arrival/departure notifications (TDD).

These tests verify that WhoLLM fires events when people arrive or leave.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestNotificationEvents:
    """Test notification event types."""

    def test_event_constants_defined(self):
        """Event type constants should be defined."""
        from custom_components.whollm.notifications import (
            EVENT_PERSON_ARRIVED,
            EVENT_PERSON_LEFT,
            EVENT_ROOM_CHANGED,
        )
        
        assert EVENT_PERSON_ARRIVED == "whollm_person_arrived"
        assert EVENT_PERSON_LEFT == "whollm_person_left"
        assert EVENT_ROOM_CHANGED == "whollm_room_changed"


class TestNotificationManager:
    """Test the NotificationManager class."""

    def test_manager_initialization(self):
        """Manager should initialize without errors."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        manager = NotificationManager(mock_hass)
        
        assert manager is not None
        assert manager._hass == mock_hass

    def test_track_state_change(self):
        """Manager should track room state changes."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        manager = NotificationManager(mock_hass)
        
        # Initial state
        manager.update_room("Ben", "bedroom")
        
        # Change room
        manager.update_room("Ben", "kitchen")
        
        # Should have recorded both states
        assert manager.get_current_room("Ben") == "kitchen"
        assert manager.get_previous_room("Ben") == "bedroom"

    def test_detect_arrival(self):
        """Manager should detect arrival (away -> home)."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        mock_hass.bus.async_fire = MagicMock()
        
        manager = NotificationManager(mock_hass)
        
        # Set initial state as away
        manager.update_room("Ben", "away")
        
        # Come home
        manager.update_room("Ben", "living_room")
        
        # Should have fired arrival event
        mock_hass.bus.async_fire.assert_called()
        call_args = mock_hass.bus.async_fire.call_args_list
        
        # Find the arrival event
        arrival_calls = [c for c in call_args if "arrived" in c[0][0]]
        assert len(arrival_calls) >= 1

    def test_detect_departure(self):
        """Manager should detect departure (home -> away)."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        mock_hass.bus.async_fire = MagicMock()
        
        manager = NotificationManager(mock_hass)
        
        # Set initial state as home
        manager.update_room("Ben", "office")
        
        # Leave home
        manager.update_room("Ben", "away")
        
        # Should have fired departure event
        mock_hass.bus.async_fire.assert_called()
        call_args = mock_hass.bus.async_fire.call_args_list
        
        # Find the departure event
        departure_calls = [c for c in call_args if "left" in c[0][0]]
        assert len(departure_calls) >= 1

    def test_room_change_event(self):
        """Manager should fire room change events."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        mock_hass.bus.async_fire = MagicMock()
        
        manager = NotificationManager(mock_hass)
        
        # Change rooms (both home)
        manager.update_room("Ben", "bedroom")
        manager.update_room("Ben", "kitchen")
        
        # Should have fired room change event
        mock_hass.bus.async_fire.assert_called()

    def test_no_event_for_same_room(self):
        """Manager should not fire event for same room."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        mock_hass.bus.async_fire = MagicMock()
        
        manager = NotificationManager(mock_hass)
        
        # Set room
        manager.update_room("Ben", "office")
        mock_hass.bus.async_fire.reset_mock()
        
        # Same room again
        manager.update_room("Ben", "office")
        
        # Should not have fired any event
        mock_hass.bus.async_fire.assert_not_called()


class TestEventData:
    """Test event data structure."""

    def test_arrival_event_data(self):
        """Arrival event should include relevant data."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        mock_hass.bus.async_fire = MagicMock()
        
        manager = NotificationManager(mock_hass)
        manager.update_room("Ben", "away")
        manager.update_room("Ben", "living_room")
        
        # Get the event data
        call_args = mock_hass.bus.async_fire.call_args_list
        arrival_call = [c for c in call_args if "arrived" in c[0][0]][0]
        
        event_data = arrival_call[0][1]  # Second positional arg is data
        
        assert "entity_name" in event_data
        assert event_data["entity_name"] == "Ben"
        assert "room" in event_data
        assert event_data["room"] == "living_room"

    def test_departure_event_data(self):
        """Departure event should include relevant data."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        mock_hass.bus.async_fire = MagicMock()
        
        manager = NotificationManager(mock_hass)
        manager.update_room("Ben", "office")
        manager.update_room("Ben", "away")
        
        call_args = mock_hass.bus.async_fire.call_args_list
        departure_call = [c for c in call_args if "left" in c[0][0]][0]
        
        event_data = departure_call[0][1]
        
        assert "entity_name" in event_data
        assert "last_room" in event_data
        assert event_data["last_room"] == "office"


class TestNotificationConfig:
    """Test notification configuration."""

    def test_notifications_enabled_default(self):
        """Notifications should be enabled by default."""
        from custom_components.whollm.const import (
            CONF_NOTIFICATIONS_ENABLED,
            DEFAULT_NOTIFICATIONS_ENABLED,
        )
        
        assert CONF_NOTIFICATIONS_ENABLED == "notifications_enabled"
        assert DEFAULT_NOTIFICATIONS_ENABLED is True

    def test_disable_notifications(self):
        """Should respect notifications_enabled config."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        mock_hass.bus.async_fire = MagicMock()
        
        manager = NotificationManager(mock_hass, enabled=False)
        
        manager.update_room("Ben", "away")
        manager.update_room("Ben", "home")
        
        # Should not fire events when disabled
        mock_hass.bus.async_fire.assert_not_called()


class TestMultiplePeople:
    """Test notifications with multiple people."""

    def test_track_multiple_people(self):
        """Should track multiple people independently."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        manager = NotificationManager(mock_hass)
        
        manager.update_room("Ben", "office")
        manager.update_room("Alex", "bedroom")
        
        assert manager.get_current_room("Ben") == "office"
        assert manager.get_current_room("Alex") == "bedroom"

    def test_independent_arrival_events(self):
        """Each person should get their own arrival event."""
        from custom_components.whollm.notifications import NotificationManager
        
        mock_hass = MagicMock()
        mock_hass.bus.async_fire = MagicMock()
        
        manager = NotificationManager(mock_hass)
        
        # Both away
        manager.update_room("Ben", "away")
        manager.update_room("Alex", "away")
        mock_hass.bus.async_fire.reset_mock()
        
        # Ben arrives
        manager.update_room("Ben", "living_room")
        
        # Check Ben's arrival event
        call_args = mock_hass.bus.async_fire.call_args_list
        arrival_calls = [c for c in call_args if "arrived" in c[0][0]]
        
        assert len(arrival_calls) == 1
        assert arrival_calls[0][0][1]["entity_name"] == "Ben"
