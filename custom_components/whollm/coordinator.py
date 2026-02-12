"""Data update coordinator for WhoLLM."""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    CONF_CONFIDENCE_WEIGHTS,
    CONF_LEARNING_ENABLED,
    CONF_MODEL,
    CONF_PERSON_DEVICES,
    CONF_PERSONS,
    CONF_PETS,
    CONF_POLL_INTERVAL,
    CONF_PROVIDER,
    CONF_ROOM_ENTITIES,
    CONF_ROOMS,
    CONF_TIMEOUT,
    CONF_URL,
    DEFAULT_CONFIDENCE_WEIGHTS,
    DEFAULT_LEARNING_ENABLED,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PROVIDER,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    DEVICE_TRACKER_AWAY_CONFIDENCE,
    DEVICE_TRACKER_HOME,
    DEVICE_TRACKER_NOT_HOME,
    DEVICE_TRACKER_UNAVAILABLE,
    DOMAIN,
    ENTITY_HINT_COMPUTER,
    ENTITY_HINT_MEDIA,
    ROOM_AWAY,
    VALID_ROOMS,
)
from .event_logger import EventLogger
from .habits import ConfidenceCombiner, HabitPredictor
from .providers import get_provider
from .providers.base import PresenceGuess

_LOGGER = logging.getLogger(__name__)


class DeviceTrackerHelper:
    """Helper class for device tracker integration."""

    def is_person_home(self, hass: HomeAssistant, device_tracker_id: str) -> bool | None:
        """Check if a single device tracker indicates person is home.
        
        Returns:
            True if home, False if away, None if unavailable/unknown
        """
        state = hass.states.get(device_tracker_id)
        if not state:
            return None
        
        state_value = state.state.lower()
        
        if state_value in (DEVICE_TRACKER_UNAVAILABLE, "unknown"):
            return None
        
        # "home" or zone names indicate presence
        if state_value == DEVICE_TRACKER_HOME:
            return True
        
        # "not_home" indicates away
        if state_value == DEVICE_TRACKER_NOT_HOME:
            return False
        
        # Any other zone name (e.g., "work", "school") means not at home
        # but we return True only for "home"
        return False

    def is_person_home_any(self, hass: HomeAssistant, device_tracker_ids: list[str]) -> bool | None:
        """Check if any device tracker indicates person is home.
        
        Returns True if ANY tracker shows home.
        Returns False if ALL trackers show away.
        Returns None if all trackers are unavailable.
        """
        if not device_tracker_ids:
            return None
        
        has_valid_state = False
        
        for tracker_id in device_tracker_ids:
            result = self.is_person_home(hass, tracker_id)
            if result is True:
                return True
            if result is False:
                has_valid_state = True
        
        # If we had at least one valid "not_home", return False
        if has_valid_state:
            return False
        
        # All trackers unavailable
        return None

    def get_away_confidence(self) -> float:
        """Get confidence level for device tracker based away detection."""
        return DEVICE_TRACKER_AWAY_CONFIDENCE

    def check_person_device_trackers(
        self, 
        hass: HomeAssistant, 
        person_name: str, 
        person_devices: dict[str, list[str]]
    ) -> tuple[bool | None, list[str]]:
        """Check all device trackers for a person.
        
        Returns:
            (is_home, list_of_tracker_ids_checked)
        """
        # Get device trackers from person_devices config
        devices = person_devices.get(person_name, [])
        
        # Filter to only device_tracker entities
        trackers = [d for d in devices if d.startswith("device_tracker.")]
        
        if not trackers:
            return None, []
        
        is_home = self.is_person_home_any(hass, trackers)
        return is_home, trackers


class LLMPresenceCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage LLM presence detection."""

    @staticmethod
    def _get_config(entry: ConfigEntry, key: str, default: Any = None) -> Any:
        """Get config value from options first, then data, then default."""
        return entry.options.get(key, entry.data.get(key, default))

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        event_logger: EventLogger | None = None,
        habit_predictor: HabitPredictor | None = None,
        confidence_combiner: ConfidenceCombiner | None = None,
    ) -> None:
        """Initialize the coordinator."""
        self.entry = entry
        self.provider_type = self._get_config(entry, CONF_PROVIDER, DEFAULT_PROVIDER)
        self.provider_url = self._get_config(entry, CONF_URL, DEFAULT_URL)
        self.model = self._get_config(entry, CONF_MODEL, DEFAULT_MODEL)
        self.persons = self._get_config(entry, CONF_PERSONS, [])
        self.pets = self._get_config(entry, CONF_PETS, [])
        self.rooms = self._get_config(entry, CONF_ROOMS, VALID_ROOMS)
        self.room_entities: dict[str, list[dict]] = self._get_config(entry, CONF_ROOM_ENTITIES, {})
        self.person_devices: dict[str, list[str]] = self._get_config(entry, CONF_PERSON_DEVICES, {})
        self.confidence_weights = self._get_config(entry, CONF_CONFIDENCE_WEIGHTS, DEFAULT_CONFIDENCE_WEIGHTS)
        self.learning_enabled = self._get_config(entry, CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED)
        poll_interval = self._get_config(entry, CONF_POLL_INTERVAL, DEFAULT_POLL_INTERVAL)

        # Initialize the LLM provider
        self.timeout = self._get_config(entry, CONF_TIMEOUT, DEFAULT_TIMEOUT)
        self.provider = get_provider(
            provider_type=self.provider_type,
            url=self.provider_url,
            model=self.model,
            timeout=self.timeout,
        )

        # Accept injected instances or create defaults
        self.event_logger = event_logger or EventLogger()
        self.habit_predictor = habit_predictor or HabitPredictor()
        self.confidence_combiner = confidence_combiner or ConfidenceCombiner(self.confidence_weights)

        # Device tracker helper for away detection
        self.device_tracker_helper = DeviceTrackerHelper()

        # Track previous rooms for transition detection
        self._previous_rooms: dict[str, str] = {}

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=poll_interval),
        )

        _LOGGER.info(
            "WhoLLM coordinator initialized: provider=%s, rooms=%d, room_entities=%d configured",
            self.provider_type,
            len(self.rooms),
            len(self.room_entities),
        )

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from LLM provider."""

        start_time = time.time()

        try:
            # Gather sensor context from Home Assistant
            context = await self._gather_sensor_context()

            # Get active indicators from configured room entities
            active_indicators = self._get_active_indicators(context)

            # Query LLM for each person/pet
            results: dict[str, Any] = {
                "persons": {},
                "pets": {},
                "last_context": context,
            }

            for person in self.persons:
                person_name = person.get("name", "unknown")
                guess = await self._process_entity(
                    person_name, "person", context, active_indicators,
                )
                results["persons"][person_name] = guess

            for pet in self.pets:
                pet_name = pet.get("name", "unknown")
                guess = await self._process_entity(
                    pet_name, "pet", context, active_indicators, is_pet=True,
                )
                results["pets"][pet_name] = guess

            elapsed = time.time() - start_time
            _LOGGER.debug("Finished fetching whollm data in %.3f seconds (success: True)", elapsed)

            return results

        except Exception as err:
            _LOGGER.error("Error fetching LLM presence data: %s", err)
            raise UpdateFailed(f"Error communicating with LLM: {err}") from err

    async def _process_entity(
        self,
        name: str,
        entity_type: str,
        context: dict[str, Any],
        active_indicators: list[dict],
        is_pet: bool = False,
    ) -> PresenceGuess:
        """Process a single person or pet through the presence detection pipeline."""

        # Device tracker check (persons only)
        if not is_pet:
            is_home, trackers_checked = self.device_tracker_helper.check_person_device_trackers(
                self.hass, name, self.person_devices
            )
            if is_home is False:
                guess = PresenceGuess(
                    room=ROOM_AWAY,
                    confidence=self.device_tracker_helper.get_away_confidence(),
                    raw_response="Device tracker indicates not home",
                    indicators=[f"device_tracker: {t}" for t in trackers_checked],
                    source="device_tracker",
                )
                _LOGGER.debug("%s is away (device tracker: %s)", name, trackers_checked)
                await self._log_presence_event(name, entity_type, guess, context)
                await self._check_room_transition(name, guess.room, guess.confidence)
                return guess

        # Get habit hint (learned patterns)
        habit_hint = self.habit_predictor.get_habit_hint(name, entity_type)
        habit_context = self.habit_predictor.get_habit_context_for_prompt(name, entity_type)

        # Get indicators
        if is_pet:
            entity_indicators = [i for i in active_indicators if i.get("hint_type") != ENTITY_HINT_COMPUTER]
        else:
            entity_indicators = self._get_person_indicators(name, active_indicators, context)

        # Build context for LLM
        context_for_llm = {
            **context,
            "habit_hint": habit_context,
            "active_indicators": [
                f"{i['hint_type']}: {i['entity_id']} in {i['room']}" for i in entity_indicators
            ],
        }

        # Query LLM
        guess = await self.provider.deduce_presence(
            hass=self.hass,
            context=context_for_llm,
            entity_name=name,
            entity_type=entity_type,
            rooms=self.rooms,
        )

        # Combine confidence from multiple sources
        final_room, final_confidence, explanation = self.confidence_combiner.combine(
            llm_room=guess.room,
            llm_confidence=guess.confidence,
            habit_room=habit_hint.get("predicted_room"),
            habit_confidence=habit_hint.get("confidence", 0),
            sensor_indicators=entity_indicators,
            entity_name=name,
            room_entities=self.room_entities,
        )

        # Update guess with combined result
        if guess.room != final_room:
            _LOGGER.debug(
                "Confidence combiner changed %s from %s to %s (reason: %s)",
                name, guess.room, final_room, explanation,
            )
        guess.room = final_room
        guess.confidence = final_confidence
        indicator_list = [f"{i['hint_type']}: {i['entity_id'].split('.')[-1]}" for i in entity_indicators[:5]]
        if explanation:
            indicator_list.append(explanation)
        guess.indicators = indicator_list

        # Learn from this event if confident
        if self.learning_enabled and final_confidence >= 0.7:
            self.habit_predictor.learn_from_event(
                entity_name=name,
                room=final_room,
                confidence=final_confidence,
            )

        # Log the event and check for room transitions
        await self._log_presence_event(name, entity_type, guess, context)
        await self._check_room_transition(name, guess.room, guess.confidence)

        return guess

    def _get_active_indicators(self, context: dict[str, Any]) -> list[dict]:
        """Get list of active indicators based on configured room entities.

        Returns list of dicts with: entity_id, hint_type, room, state
        """
        active = []

        for room, entities in self.room_entities.items():
            for entity_config in entities:
                entity_id = entity_config.get("entity_id")
                hint_type = entity_config.get("hint_type", "appliance")

                # Get current state from context or HA directly
                state = self._get_entity_state(entity_id, context)

                if self._is_entity_active(entity_id, state, hint_type):
                    active.append(
                        {
                            "entity_id": entity_id,
                            "hint_type": hint_type,
                            "room": room,
                            "state": state,
                        }
                    )

        return active

    def _get_entity_state(self, entity_id: str, context: dict[str, Any]) -> str:
        """Get entity state from context or Home Assistant."""
        # Check context first
        for category in ["lights", "motion", "media", "computers", "doors", "ai_detection"]:
            if entity_id in context.get(category, {}):
                return context[category][entity_id].get("state", "unknown")

        # Fall back to HA state
        state = self.hass.states.get(entity_id)
        return state.state if state else "unknown"

    def _is_entity_active(self, entity_id: str, state: str, hint_type: str) -> bool:
        """Determine if an entity is in an 'active' state."""
        state_lower = state.lower() if state else ""

        # Media players are special - playing is active, paused is semi-active
        if hint_type == ENTITY_HINT_MEDIA:
            return state_lower in ["playing", "paused", "on"]

        # Most entities: "on", "home", "detected" are active
        active_states = ["on", "home", "detected", "playing", "open", "occupied"]
        return state_lower in active_states

    def _get_person_indicators(
        self,
        person_name: str,
        all_indicators: list[dict],
        context: dict[str, Any],
    ) -> list[dict]:
        """Get indicators relevant to a specific person.

        Includes:
        - All general indicators (motion, media, etc.)
        - Person's owned devices (if active, strongly suggests their location)
        """
        indicators = list(all_indicators)  # Copy all indicators

        # Check person's owned devices
        owned_devices = self.person_devices.get(person_name, [])
        for device_id in owned_devices:
            state = self._get_entity_state(device_id, context)
            if self._is_entity_active(device_id, state, ENTITY_HINT_COMPUTER):
                # Find which room this device is in
                room = self._find_entity_room(device_id)
                if room:
                    indicators.append(
                        {
                            "entity_id": device_id,
                            "hint_type": ENTITY_HINT_COMPUTER,
                            "room": room,
                            "state": state,
                            "owned_by": person_name,
                        }
                    )

        return indicators

    def _find_entity_room(self, entity_id: str) -> str | None:
        """Find which room an entity belongs to based on config."""
        for room, entities in self.room_entities.items():
            for entity_config in entities:
                if entity_config.get("entity_id") == entity_id:
                    return room
        return None

    async def _gather_sensor_context(self) -> dict[str, Any]:
        """Gather current state of only configured entities.

        Only reads entities from room_entities config and person_devices,
        instead of scanning all HA entities.
        """
        now = dt_util.now()

        context: dict[str, Any] = {
            "lights": {},
            "motion": {},
            "media": {},
            "device_trackers": {},
            "doors": {},
            "climate": {},
            "computers": {},
            "ai_detection": {},
            "time_context": {
                "current_time": now.strftime("%H:%M"),
                "day_of_week": now.strftime("%A"),
                "is_night": now.hour >= 22 or now.hour < 6,
                "is_morning": 6 <= now.hour < 10,
                "is_evening": 18 <= now.hour < 22,
            },
            "configured_rooms": self.rooms,
        }

        # Collect all entity IDs we need to read
        entity_ids: set[str] = set()

        # From room_entities config
        for _room, entities in self.room_entities.items():
            for entity_config in entities:
                entity_id = entity_config.get("entity_id")
                if entity_id:
                    entity_ids.add(entity_id)

        # From person_devices config
        for _person, devices in self.person_devices.items():
            for device_id in devices:
                entity_ids.add(device_id)

        # Read and categorize each configured entity
        for entity_id in entity_ids:
            state = self.hass.states.get(entity_id)
            if not state:
                continue

            time_since_change = self._format_time_since(now, state.last_changed)

            # Find hint_type from room_entities config
            hint_type = self._get_entity_hint_type(entity_id)

            # Categorize by hint_type first, then by domain as fallback
            self._categorize_entity(
                context, entity_id, state, hint_type, time_since_change,
            )

        return context

    def _format_time_since(self, now, last_changed) -> str | None:
        """Format time since last state change as human-readable string."""
        if not last_changed:
            return None
        delta = now - last_changed
        seconds = delta.total_seconds()
        if seconds < 60:
            return f"{int(seconds)}s ago"
        if seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        return f"{int(seconds / 3600)}h ago"

    def _get_entity_hint_type(self, entity_id: str) -> str | None:
        """Look up the configured hint_type for an entity."""
        for _room, entities in self.room_entities.items():
            for entity_config in entities:
                if entity_config.get("entity_id") == entity_id:
                    return entity_config.get("hint_type", "appliance")
        return None

    def _categorize_entity(
        self,
        context: dict[str, Any],
        entity_id: str,
        state,
        hint_type: str | None,
        time_since_change: str | None,
    ) -> None:
        """Categorize an entity into the appropriate context bucket."""
        # Use hint_type if configured, otherwise infer from domain/name
        if hint_type == ENTITY_HINT_COMPUTER:
            room = self._find_entity_room(entity_id)
            context["computers"][entity_id] = {
                "state": state.state,
                "room": room or "unknown",
            }
        elif hint_type == "light" or entity_id.startswith("light."):
            context["lights"][entity_id] = {
                "state": state.state,
                "last_changed": time_since_change,
                "brightness": state.attributes.get("brightness"),
            }
        elif hint_type == "motion" or (
            entity_id.startswith("binary_sensor.") and "motion" in entity_id.lower()
        ):
            context["motion"][entity_id] = {
                "state": state.state,
                "last_changed": time_since_change,
            }
        elif hint_type == "door" or (
            entity_id.startswith("binary_sensor.")
            and any(x in entity_id.lower() for x in ["door", "window", "contact"])
        ):
            context["doors"][entity_id] = {
                "state": state.state,
                "last_changed": time_since_change,
            }
        elif hint_type == "media" or entity_id.startswith("media_player."):
            media_info = {
                "state": state.state,
                "last_changed": time_since_change,
            }
            if state.attributes.get("media_title"):
                media_info["playing"] = state.attributes.get("media_title")
            if state.attributes.get("app_name"):
                media_info["app"] = state.attributes.get("app_name")
            context["media"][entity_id] = media_info
        elif hint_type == "camera" or (
            entity_id.startswith("binary_sensor.")
            and any(x in entity_id.lower() for x in ["_person", "_animal", "_pet", "_human"])
        ):
            context["ai_detection"][entity_id] = {
                "state": state.state,
                "last_changed": time_since_change,
            }
        elif hint_type == "climate" or (
            entity_id.startswith("sensor.")
            and any(x in entity_id.lower() for x in ["temperature", "humidity"])
        ):
            context["climate"][entity_id] = {
                "state": state.state,
                "unit": state.attributes.get("unit_of_measurement", ""),
            }
        elif hint_type == "presence" or entity_id.startswith(("device_tracker.", "person.")):
            context["device_trackers"][entity_id] = {
                "state": state.state,
                "last_changed": time_since_change,
            }
        # Default: skip uncategorized entities rather than pollute context

    async def _log_presence_event(
        self,
        entity_name: str,
        entity_type: str,
        guess: Any,
        context: dict[str, Any],
    ) -> None:
        """Log presence event for ML training."""
        try:
            await self.event_logger.async_log_presence_event(
                entity_name=entity_name,
                entity_type=entity_type,
                room=guess.room,
                confidence=guess.confidence,
                raw_response=guess.raw_response,
                indicators=guess.indicators,
                sensor_context=context,
                detection_method="llm",
            )
        except OSError as err:
            _LOGGER.warning("Failed to log presence event: %s", err)

    async def _check_room_transition(
        self,
        entity_name: str,
        new_room: str,
        confidence: float,
    ) -> None:
        """Check if entity moved rooms and log transition."""
        previous_room = self._previous_rooms.get(entity_name)

        if previous_room and previous_room != new_room and new_room != "unknown":
            try:
                await self.event_logger.async_log_room_transition(
                    entity_name=entity_name,
                    from_room=previous_room,
                    to_room=new_room,
                    confidence=confidence,
                )
                _LOGGER.debug("Room transition: %s moved from %s to %s", entity_name, previous_room, new_room)
            except OSError as err:
                _LOGGER.warning("Failed to log room transition: %s", err)

        self._previous_rooms[entity_name] = new_room
