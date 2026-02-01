"""Notification system for WhoLLM.

Fires Home Assistant events when people arrive, leave, or change rooms.
Events can be used to trigger automations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Event types
EVENT_PERSON_ARRIVED = "whollm_person_arrived"
EVENT_PERSON_LEFT = "whollm_person_left"
EVENT_ROOM_CHANGED = "whollm_room_changed"

# Special room values
ROOM_AWAY = "away"
ROOM_UNKNOWN = "unknown"


class NotificationManager:
    """Manages arrival/departure notifications.
    
    Tracks room state changes and fires events when:
    - Someone arrives home (away -> any room)
    - Someone leaves home (any room -> away)
    - Someone changes rooms
    """
    
    def __init__(self, hass: HomeAssistant, enabled: bool = True):
        """Initialize the notification manager.
        
        Args:
            hass: Home Assistant instance
            enabled: Whether notifications are enabled
        """
        self._hass = hass
        self._enabled = enabled
        
        # Track current and previous room for each entity
        self._current_rooms: dict[str, str] = {}
        self._previous_rooms: dict[str, str] = {}
    
    def update_room(
        self,
        entity_name: str,
        new_room: str,
        confidence: float = 0.0,
    ) -> None:
        """Update room state and fire events if needed.
        
        Args:
            entity_name: Name of the person
            new_room: New room name
            confidence: Confidence of the detection
        """
        old_room = self._current_rooms.get(entity_name)
        
        # No change
        if old_room == new_room:
            return
        
        # Update state
        self._previous_rooms[entity_name] = old_room
        self._current_rooms[entity_name] = new_room
        
        if not self._enabled:
            return
        
        timestamp = datetime.now().isoformat()
        
        # Check for arrival (away/unknown -> home)
        if self._is_away(old_room) and not self._is_away(new_room):
            self._fire_arrival_event(entity_name, new_room, confidence, timestamp)
        
        # Check for departure (home -> away)
        elif not self._is_away(old_room) and self._is_away(new_room) and old_room is not None:
            self._fire_departure_event(entity_name, old_room, timestamp)
        
        # Room change (both home, different rooms)
        elif old_room is not None and not self._is_away(old_room) and not self._is_away(new_room):
            self._fire_room_change_event(entity_name, old_room, new_room, confidence, timestamp)
    
    def _is_away(self, room: str | None) -> bool:
        """Check if a room value indicates away/not home."""
        if room is None:
            return True
        return room.lower() in (ROOM_AWAY, ROOM_UNKNOWN, "not_home")
    
    def _fire_arrival_event(
        self,
        entity_name: str,
        room: str,
        confidence: float,
        timestamp: str,
    ) -> None:
        """Fire person arrived event."""
        event_data = {
            "entity_name": entity_name,
            "room": room,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        
        self._hass.bus.async_fire(EVENT_PERSON_ARRIVED, event_data)
        _LOGGER.info("%s arrived home (in %s)", entity_name, room)
    
    def _fire_departure_event(
        self,
        entity_name: str,
        last_room: str,
        timestamp: str,
    ) -> None:
        """Fire person left event."""
        event_data = {
            "entity_name": entity_name,
            "last_room": last_room,
            "timestamp": timestamp,
        }
        
        self._hass.bus.async_fire(EVENT_PERSON_LEFT, event_data)
        _LOGGER.info("%s left home (was in %s)", entity_name, last_room)
    
    def _fire_room_change_event(
        self,
        entity_name: str,
        from_room: str,
        to_room: str,
        confidence: float,
        timestamp: str,
    ) -> None:
        """Fire room changed event."""
        event_data = {
            "entity_name": entity_name,
            "from_room": from_room,
            "to_room": to_room,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        
        self._hass.bus.async_fire(EVENT_ROOM_CHANGED, event_data)
        _LOGGER.debug("%s moved from %s to %s", entity_name, from_room, to_room)
    
    def get_current_room(self, entity_name: str) -> str | None:
        """Get current room for an entity."""
        return self._current_rooms.get(entity_name)
    
    def get_previous_room(self, entity_name: str) -> str | None:
        """Get previous room for an entity."""
        return self._previous_rooms.get(entity_name)
    
    def is_home(self, entity_name: str) -> bool:
        """Check if an entity is currently home."""
        room = self._current_rooms.get(entity_name)
        return room is not None and not self._is_away(room)
    
    def get_all_home(self) -> list[str]:
        """Get list of all entities currently home."""
        return [
            name for name, room in self._current_rooms.items()
            if not self._is_away(room)
        ]
    
    def get_all_away(self) -> list[str]:
        """Get list of all entities currently away."""
        return [
            name for name, room in self._current_rooms.items()
            if self._is_away(room)
        ]


# Global instance
_notification_manager: NotificationManager | None = None


def get_notification_manager(
    hass: HomeAssistant | None = None,
    enabled: bool = True,
) -> NotificationManager | None:
    """Get or create the global notification manager."""
    global _notification_manager
    
    if _notification_manager is None and hass is not None:
        _notification_manager = NotificationManager(hass, enabled)
    
    return _notification_manager


def reset_notification_manager() -> None:
    """Reset the global notification manager (for testing)."""
    global _notification_manager
    _notification_manager = None
