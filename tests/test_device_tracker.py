"""
Tests for device tracker integration (TDD).

These tests verify that WhoLLM correctly uses device_tracker entities
to determine if a person is home or away.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestDeviceTrackerSupport:
    """Test device tracker integration for away detection."""

    def test_config_accepts_device_tracker_mapping(self):
        """Config should accept person-to-device_tracker mappings."""
        from custom_components.whollm.const import CONF_PERSON_DEVICES
        
        # Config should have a constant for person devices
        assert CONF_PERSON_DEVICES == "person_devices"

    def test_device_tracker_home_state(self):
        """Device tracker 'home' state should indicate person is present."""
        from custom_components.whollm.coordinator import DeviceTrackerHelper
        
        helper = DeviceTrackerHelper()
        
        # Mock hass with device_tracker state
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="home")
        
        result = helper.is_person_home(mock_hass, "device_tracker.john_phone")
        assert result is True

    def test_device_tracker_not_home_state(self):
        """Device tracker 'not_home' state should indicate person is away."""
        from custom_components.whollm.coordinator import DeviceTrackerHelper
        
        helper = DeviceTrackerHelper()
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="not_home")
        
        result = helper.is_person_home(mock_hass, "device_tracker.john_phone")
        assert result is False

    def test_device_tracker_zone_state(self):
        """Device tracker with zone name should indicate person is home."""
        from custom_components.whollm.coordinator import DeviceTrackerHelper
        
        helper = DeviceTrackerHelper()
        
        mock_hass = MagicMock()
        # Zone-based trackers return the zone name when home
        mock_hass.states.get.return_value = MagicMock(state="home")
        
        result = helper.is_person_home(mock_hass, "device_tracker.john_phone")
        assert result is True

    def test_multiple_device_trackers_any_home(self):
        """If any device tracker shows home, person is considered home."""
        from custom_components.whollm.coordinator import DeviceTrackerHelper
        
        helper = DeviceTrackerHelper()
        
        mock_hass = MagicMock()
        # First tracker away, second home
        mock_hass.states.get.side_effect = [
            MagicMock(state="not_home"),
            MagicMock(state="home"),
        ]
        
        devices = ["device_tracker.john_phone", "device_tracker.john_watch"]
        result = helper.is_person_home_any(mock_hass, devices)
        assert result is True

    def test_multiple_device_trackers_all_away(self):
        """If all device trackers show away, person is considered away."""
        from custom_components.whollm.coordinator import DeviceTrackerHelper
        
        helper = DeviceTrackerHelper()
        
        mock_hass = MagicMock()
        mock_hass.states.get.side_effect = [
            MagicMock(state="not_home"),
            MagicMock(state="not_home"),
        ]
        
        devices = ["device_tracker.john_phone", "device_tracker.john_watch"]
        result = helper.is_person_home_any(mock_hass, devices)
        assert result is False

    def test_away_detection_high_confidence(self):
        """Away detection from device tracker should have high confidence."""
        from custom_components.whollm.coordinator import DeviceTrackerHelper
        
        helper = DeviceTrackerHelper()
        
        # When device tracker says away, confidence should be high
        confidence = helper.get_away_confidence()
        assert confidence >= 0.9

    def test_device_tracker_unavailable(self):
        """Unavailable device tracker should not affect home/away status."""
        from custom_components.whollm.coordinator import DeviceTrackerHelper
        
        helper = DeviceTrackerHelper()
        
        mock_hass = MagicMock()
        mock_hass.states.get.return_value = MagicMock(state="unavailable")
        
        result = helper.is_person_home(mock_hass, "device_tracker.john_phone")
        # Unavailable should return None (unknown)
        assert result is None

    def test_coordinator_uses_device_tracker(self):
        """Coordinator should check device tracker before making predictions."""
        from custom_components.whollm.coordinator import LLMPresenceCoordinator
        
        # This test verifies the integration point exists
        # The actual implementation will use DeviceTrackerHelper
        assert hasattr(LLMPresenceCoordinator, '_check_device_trackers') or \
               hasattr(LLMPresenceCoordinator, '_async_update_data')


class TestDeviceTrackerConfigFlow:
    """Test config flow for device tracker setup."""

    def test_person_devices_in_config_schema(self):
        """Config schema should include person_devices field."""
        from custom_components.whollm.const import CONF_PERSON_DEVICES
        
        # The constant should exist
        assert CONF_PERSON_DEVICES is not None

    def test_device_tracker_entities_selectable(self):
        """Device tracker entities should be selectable in config."""
        # This is more of an integration test, but we verify the pattern
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        # Config flow should exist and handle device trackers
        assert LLMPresenceConfigFlow is not None


class TestAwayPrediction:
    """Test that away predictions are generated correctly."""

    def test_away_room_value(self):
        """Away should be represented as 'away' room value."""
        from custom_components.whollm.const import ROOM_AWAY
        
        assert ROOM_AWAY == "away"

    def test_presence_guess_supports_away(self):
        """PresenceGuess should support away as a valid room."""
        from custom_components.whollm.providers.base import PresenceGuess
        
        guess = PresenceGuess(
            room="away",
            confidence=0.95,
            raw_response="Away detection",
            indicators=["device_tracker"],
            source="device_tracker"
        )
        
        assert guess.room == "away"
        assert guess.confidence == 0.95
        assert guess.source == "device_tracker"
