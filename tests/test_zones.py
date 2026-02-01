"""
Tests for multi-zone support (TDD).

These tests verify tracking people across multiple zones (work, gym, etc.)
beyond just home/away.
"""

import pytest
from unittest.mock import MagicMock


class TestZoneConstants:
    """Test zone-related constants."""

    def test_zone_constants_defined(self):
        """Zone constants should be defined."""
        from custom_components.whollm.zones import (
            ZONE_HOME,
            ZONE_AWAY,
            ZONE_WORK,
        )
        
        assert ZONE_HOME == "home"
        assert ZONE_AWAY == "away"
        assert ZONE_WORK == "work"


class TestZoneManager:
    """Test the ZoneManager class."""

    def test_manager_initialization(self):
        """Manager should initialize with default zones."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        manager = ZoneManager(mock_hass)
        
        assert manager is not None

    def test_get_zone_from_device_tracker(self):
        """Should determine zone from device tracker state."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="work")
        
        manager = ZoneManager(mock_hass)
        
        zone = manager.get_zone("device_tracker.ben_phone")
        assert zone == "work"

    def test_home_zone_detection(self):
        """Should detect home zone."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="home")
        
        manager = ZoneManager(mock_hass)
        
        zone = manager.get_zone("device_tracker.ben_phone")
        assert zone == "home"

    def test_not_home_is_away(self):
        """not_home should map to away zone."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="not_home")
        
        manager = ZoneManager(mock_hass)
        
        zone = manager.get_zone("device_tracker.ben_phone")
        assert zone == "away"

    def test_custom_zone_mapping(self):
        """Should support custom zone mappings."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="gym")
        
        manager = ZoneManager(mock_hass)
        
        zone = manager.get_zone("device_tracker.ben_phone")
        assert zone == "gym"


class TestPersonZoneTracking:
    """Test tracking zones for people."""

    def test_get_person_zone(self):
        """Should get zone for a person from their device trackers."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="work")
        
        manager = ZoneManager(mock_hass)
        
        person_devices = {"Ben": ["device_tracker.ben_phone"]}
        
        zone = manager.get_person_zone("Ben", person_devices)
        assert zone == "work"

    def test_multiple_trackers_priority(self):
        """Should handle multiple device trackers per person."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        
        # First tracker says work, second says home
        mock_hass.states.get.side_effect = [
            MagicMock(state="work"),
            MagicMock(state="home"),
        ]
        
        manager = ZoneManager(mock_hass)
        
        person_devices = {"Ben": ["device_tracker.ben_phone", "device_tracker.ben_watch"]}
        
        # Home should take priority (any tracker showing home = home)
        zone = manager.get_person_zone("Ben", person_devices)
        assert zone == "home"

    def test_no_trackers_returns_unknown(self):
        """Should return unknown when no trackers configured."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        manager = ZoneManager(mock_hass)
        
        person_devices = {}
        
        zone = manager.get_person_zone("Ben", person_devices)
        assert zone == "unknown"


class TestZoneAttributes:
    """Test zone attributes and metadata."""

    def test_zone_friendly_name(self):
        """Should provide friendly names for zones."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        manager = ZoneManager(mock_hass)
        
        assert manager.get_zone_friendly_name("work") == "Work"
        assert manager.get_zone_friendly_name("home") == "Home"
        assert manager.get_zone_friendly_name("gym") == "Gym"

    def test_is_at_home(self):
        """Should check if zone is home."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        manager = ZoneManager(mock_hass)
        
        assert manager.is_at_home("home") is True
        assert manager.is_at_home("work") is False
        assert manager.is_at_home("away") is False

    def test_is_known_zone(self):
        """Should check if zone is known."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        manager = ZoneManager(mock_hass)
        
        assert manager.is_known_zone("home") is True
        assert manager.is_known_zone("work") is True
        assert manager.is_known_zone("unknown") is False


class TestZoneIntegration:
    """Test zone integration with WhoLLM."""

    def test_zone_affects_room_prediction(self):
        """When person is at work, room should be 'work' not home rooms."""
        from custom_components.whollm.zones import ZoneManager, ZONE_WORK
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="work")
        
        manager = ZoneManager(mock_hass)
        
        zone = manager.get_zone("device_tracker.ben_phone")
        
        # Zone should override room prediction
        assert zone == ZONE_WORK
        assert manager.should_skip_room_prediction(zone) is True

    def test_home_zone_allows_room_prediction(self):
        """When person is home, room prediction should continue."""
        from custom_components.whollm.zones import ZoneManager
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="home")
        
        manager = ZoneManager(mock_hass)
        
        zone = manager.get_zone("device_tracker.ben_phone")
        
        # Home zone should allow room prediction
        assert manager.should_skip_room_prediction(zone) is False
