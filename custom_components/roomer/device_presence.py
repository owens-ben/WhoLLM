"""Room-level device presence for Roomer.

Bluetooth room-presence tools such as Bermuda, ESPresense and room-assistant
report which room a phone, watch or tag is in, usually as an entity whose state
is the room or area name (for example ``sensor.alice_phone_area`` = ``Office``).
When an entity like that is assigned to a person or pet under Person Device
Ownership, the room it reports belongs to that person alone, so it is used
directly instead of the shared room sensors and the LLM.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .const import (
    DEVICE_ROOM_CONFIDENCE,
    DEVICE_ROOM_SPLIT_CONFIDENCE,
    ROOM_AWAY,
    ROOM_UNKNOWN,
)
from .providers.base import PresenceGuess

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, State

_LOGGER = logging.getLogger(__name__)

SOURCE_DEVICE_ROOM = "device_room"

# Attributes that may carry the room when the state itself is not a room,
# for example a device_tracker that is "home" with an "area" attribute.
ROOM_ATTRIBUTES = ("area_id", "area_name", "area", "room")

# States that mean "no room right now" rather than a room name
NO_ROOM_STATES = {"", "unknown", "unavailable", "none", "home", "not_home", "away", "on", "off"}

# States that say the device is not here at all. Such an entity is never read as
# a room, even when it still carries a stale room attribute.
ABSENT_STATES = {"", "unknown", "unavailable", "none", "not_home", "away"}


def room_key(value: Any) -> str:
    """Reduce a room or area name to a comparable key ("Living Room" -> "living_room")."""
    text = re.sub(r"[\s-]+", "_", unicodedata.normalize("NFC", str(value)).strip().lower())
    return re.sub(r"[^\w]", "", text)


def _as_datetime(value: Any) -> datetime | None:
    """Return value if it is a datetime, else None."""
    return value if isinstance(value, datetime) else None


@dataclass
class DeviceRoomReading:
    """A device that currently reports one of the configured rooms."""

    entity_id: str
    room: str  # The configured room it matched
    reported: str  # What the device actually reported
    # When it started reporting this room: the state's last change, or for a room
    # read from an attribute (the state stays "home") when the entity last updated
    as_of: datetime | None


class DeviceRoomLocator:
    """Work out which configured room a person's or pet's devices report."""

    def __init__(self, rooms: Iterable[str]) -> None:
        """Initialize with the configured rooms."""
        self._rooms = {room_key(room): room for room in rooms if room not in (ROOM_AWAY, ROOM_UNKNOWN)}
        self._reported_unmatched: set[tuple[str, str]] = set()

    def match_room(self, value: Any) -> str | None:
        """Return the configured room a reported value names, if any."""
        if not isinstance(value, str) or value.strip().lower() in NO_ROOM_STATES:
            return None
        return self._rooms.get(room_key(value))

    def read(self, state: State) -> DeviceRoomReading | None:
        """Return the room a single entity reports, from its state or room attributes."""
        if _reports_zone(state) or str(state.state).strip().lower() in ABSENT_STATES:
            return None
        room = self.match_room(state.state)
        if room:
            return DeviceRoomReading(
                entity_id=state.entity_id,
                room=room,
                reported=state.state,
                as_of=_as_datetime(state.last_changed),
            )
        for attr in ROOM_ATTRIBUTES:
            value = state.attributes.get(attr)
            room = self.match_room(value)
            if room:
                return DeviceRoomReading(
                    entity_id=state.entity_id,
                    room=room,
                    reported=value,
                    as_of=_as_datetime(state.last_updated),
                )
        self._warn_if_unmatched_room(state)
        return None

    def locate(self, hass: HomeAssistant, entity_ids: Iterable[str]) -> list[DeviceRoomReading]:
        """Return a reading for each entity that currently reports a configured room."""
        readings = []
        for entity_id in entity_ids:
            state = hass.states.get(entity_id)
            if state is None:
                continue
            reading = self.read(state)
            if reading:
                readings.append(reading)
        return readings

    def guess(self, readings: list[DeviceRoomReading]) -> PresenceGuess:
        """Turn one or more device readings into a presence guess.

        If the devices disagree (a phone left on the desk while the watch moves
        with its owner), the room most devices report wins, and a tie goes to the
        device that started reporting its room most recently. Devices without a
        timestamp lose ties, and equal ones keep the order they were configured in.
        """
        newest_first = sorted(readings, key=_reading_sort_key, reverse=True)
        votes = Counter(reading.room for reading in readings)
        top_votes = max(votes.values())
        room = next(reading.room for reading in newest_first if votes[reading.room] == top_votes)
        confidence = DEVICE_ROOM_CONFIDENCE if len(votes) == 1 else DEVICE_ROOM_SPLIT_CONFIDENCE

        return PresenceGuess(
            room=room,
            confidence=confidence,
            raw_response="; ".join(f"{r.entity_id} reports {r.reported}" for r in newest_first),
            indicators=[f"device: {r.entity_id.split('.')[-1]} in {r.room}" for r in newest_first],
            source=SOURCE_DEVICE_ROOM,
        )

    def _warn_if_unmatched_room(self, state: State) -> None:
        """Warn once when a room-presence entity reports a room Roomer doesn't have."""
        value = state.state
        if not isinstance(value, str) or value.strip().lower() in NO_ROOM_STATES:
            return
        domain, _, object_id = state.entity_id.partition(".")
        is_room_sensor = domain == "sensor" and object_id.endswith(("_area", "_room"))
        if not (is_room_sensor or domain == "device_tracker"):
            return
        if (state.entity_id, value) in self._reported_unmatched:
            return
        self._reported_unmatched.add((state.entity_id, value))
        _LOGGER.warning(
            "%s reports '%s', which doesn't match any Roomer room (%s). "
            "Name the room or area the same in both places to use it for room presence",
            state.entity_id,
            value,
            ", ".join(sorted(self._rooms.values())),
        )


def _reports_zone(state: State) -> bool:
    """Return True for GPS trackers, whose state is an HA zone ("Office", "Work"), not a room."""
    return state.attributes.get("source_type") == "gps" or "latitude" in state.attributes


def _reading_sort_key(reading: DeviceRoomReading) -> float:
    """Sort key putting readings without a timestamp last."""
    return reading.as_of.timestamp() if reading.as_of else float("-inf")


def seen_since_away(
    hass: HomeAssistant,
    readings: list[DeviceRoomReading],
    away_tracker_ids: Iterable[str],
) -> bool:
    """Return True if a device reported a room after the person's trackers went away.

    Home/away trackers and room-presence devices can disagree for a while. On
    arrival, Bluetooth usually sees the phone before Wi-Fi or GPS marks it home.
    On departure, some room sensors keep their last room until they time out.
    Whichever changed most recently describes the present, and without
    timestamps to compare, the tracker wins.
    """
    newest_reading = max((r.as_of for r in readings if r.as_of), default=None)
    if newest_reading is None:
        return False

    away_changes = []
    for tracker_id in away_tracker_ids:
        state = hass.states.get(tracker_id)
        if state is None or state.state in ("unavailable", "unknown"):
            continue
        changed = _as_datetime(state.last_changed)
        if changed:
            away_changes.append(changed)
    if not away_changes:
        return False

    return newest_reading > max(away_changes)
