"""Multi-zone support for WhoLLM.

Tracks people across multiple zones beyond just home/away:
- Home: At home (room detection active)
- Work: At workplace
- Gym: At gym
- School: At school
- Custom zones: Any HA zone
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Standard zone constants
ZONE_HOME = "home"
ZONE_AWAY = "away"
ZONE_WORK = "work"
ZONE_GYM = "gym"
ZONE_SCHOOL = "school"
ZONE_UNKNOWN = "unknown"

# Map device tracker states to zones
DEVICE_TRACKER_STATE_MAP = {
    "home": ZONE_HOME,
    "not_home": ZONE_AWAY,
}


class ZoneManager:
    """Manages zone tracking for people.
    
    Determines which zone (home, work, gym, etc.) a person is in
    based on their device tracker states.
    """
    
    def __init__(self, hass: HomeAssistant):
        """Initialize the zone manager.
        
        Args:
            hass: Home Assistant instance
        """
        self._hass = hass
    
    def get_zone(self, device_tracker_id: str) -> str:
        """Get the zone for a device tracker.
        
        Args:
            device_tracker_id: Entity ID of the device tracker
            
        Returns:
            Zone name (home, away, work, gym, etc.)
        """
        state = self._hass.states.get(device_tracker_id)
        
        if state is None:
            return ZONE_UNKNOWN
        
        zone_state = state.state.lower()
        
        # Map standard states
        if zone_state in DEVICE_TRACKER_STATE_MAP:
            return DEVICE_TRACKER_STATE_MAP[zone_state]
        
        # Custom zones (work, gym, school, etc.) pass through
        return zone_state
    
    def get_person_zone(
        self,
        person_name: str,
        person_devices: dict[str, list[str]],
    ) -> str:
        """Get the zone for a person from their device trackers.
        
        If multiple trackers, home takes priority (any tracker showing
        home means person is home).
        
        Args:
            person_name: Name of the person
            person_devices: Mapping of person names to device tracker IDs
            
        Returns:
            Zone name
        """
        devices = person_devices.get(person_name, [])
        
        # Filter to device trackers
        trackers = [d for d in devices if d.startswith("device_tracker.")]
        
        if not trackers:
            return ZONE_UNKNOWN
        
        zones = []
        for tracker in trackers:
            zone = self.get_zone(tracker)
            zones.append(zone)
        
        # Home takes priority
        if ZONE_HOME in zones:
            return ZONE_HOME
        
        # Any non-away zone is better than away
        non_away_zones = [z for z in zones if z not in (ZONE_AWAY, ZONE_UNKNOWN)]
        if non_away_zones:
            return non_away_zones[0]
        
        # Default to away if all trackers show away
        if ZONE_AWAY in zones:
            return ZONE_AWAY
        
        return ZONE_UNKNOWN
    
    def get_zone_friendly_name(self, zone: str) -> str:
        """Get a friendly display name for a zone.
        
        Args:
            zone: Zone name
            
        Returns:
            Friendly name (capitalized)
        """
        return zone.replace("_", " ").title()
    
    def is_at_home(self, zone: str) -> bool:
        """Check if a zone indicates being at home.
        
        Args:
            zone: Zone name
            
        Returns:
            True if at home
        """
        return zone == ZONE_HOME
    
    def is_known_zone(self, zone: str) -> bool:
        """Check if a zone is a known/named zone.
        
        Args:
            zone: Zone name
            
        Returns:
            True if known zone (not unknown or away)
        """
        return zone not in (ZONE_UNKNOWN,)
    
    def should_skip_room_prediction(self, zone: str) -> bool:
        """Check if room prediction should be skipped for a zone.
        
        Room prediction only makes sense when at home. For other
        zones (work, gym, etc.), the zone IS the location.
        
        Args:
            zone: Zone name
            
        Returns:
            True if room prediction should be skipped
        """
        return zone != ZONE_HOME
    
    def get_all_zones(self) -> list[str]:
        """Get list of all standard zones.
        
        Returns:
            List of zone names
        """
        return [ZONE_HOME, ZONE_AWAY, ZONE_WORK, ZONE_GYM, ZONE_SCHOOL]


# Global instance
_zone_manager: ZoneManager | None = None


def get_zone_manager(hass: HomeAssistant | None = None) -> ZoneManager | None:
    """Get or create the global zone manager."""
    global _zone_manager
    
    if _zone_manager is None and hass is not None:
        _zone_manager = ZoneManager(hass)
    
    return _zone_manager
