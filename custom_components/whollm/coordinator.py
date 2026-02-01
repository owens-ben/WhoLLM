"""Data update coordinator for WhoLLM."""

from __future__ import annotations

import logging
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
    CONF_URL,
    DEFAULT_CONFIDENCE_WEIGHTS,
    DEFAULT_LEARNING_ENABLED,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PROVIDER,
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
from .event_logger import get_event_logger
from .habits import get_confidence_combiner, get_habit_predictor
from .providers import get_provider

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

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        self.entry = entry
        self.provider_type = entry.data.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        self.provider_url = entry.data.get(CONF_URL, DEFAULT_URL)
        self.model = entry.data.get(CONF_MODEL, DEFAULT_MODEL)
        self.persons = entry.data.get(CONF_PERSONS, [])
        self.pets = entry.data.get(CONF_PETS, [])
        self.rooms = entry.data.get(CONF_ROOMS, VALID_ROOMS)

        # Room-entity mappings from config
        self.room_entities: dict[str, list[dict]] = entry.data.get(CONF_ROOM_ENTITIES, {})

        # Person-device ownership mappings
        self.person_devices: dict[str, list[str]] = entry.data.get(CONF_PERSON_DEVICES, {})

        # Confidence weights
        self.confidence_weights = entry.data.get(CONF_CONFIDENCE_WEIGHTS, DEFAULT_CONFIDENCE_WEIGHTS)

        # Learning enabled
        self.learning_enabled = entry.data.get(CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED)

        poll_interval = entry.data.get(CONF_POLL_INTERVAL, DEFAULT_POLL_INTERVAL)

        # Initialize the LLM provider
        self.provider = get_provider(
            provider_type=self.provider_type,
            url=self.provider_url,
            model=self.model,
        )

        # Initialize event logger and habit predictor
        self.event_logger = get_event_logger()
        self.habit_predictor = get_habit_predictor()
        self.confidence_combiner = get_confidence_combiner(self.confidence_weights)

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
        import time

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

                # Check device trackers first for away detection
                is_home, trackers_checked = self.device_tracker_helper.check_person_device_trackers(
                    self.hass, person_name, self.person_devices
                )

                # If device tracker shows person is away, skip LLM and return away
                if is_home is False:
                    from .providers.base import PresenceGuess
                    
                    guess = PresenceGuess(
                        room=ROOM_AWAY,
                        confidence=self.device_tracker_helper.get_away_confidence(),
                        raw_response="Device tracker indicates not home",
                        indicators=[f"device_tracker: {t}" for t in trackers_checked],
                        source="device_tracker",
                    )
                    _LOGGER.debug(
                        "%s is away (device tracker: %s)",
                        person_name,
                        trackers_checked,
                    )
                    
                    # Log and track transition
                    self._log_presence_event(person_name, "person", guess, context)
                    self._check_room_transition(person_name, guess.room, guess.confidence)
                    results["persons"][person_name] = guess
                    continue

                # Get habit hint for this person (learned patterns)
                habit_hint = self.habit_predictor.get_habit_hint(person_name, "person")
                habit_context = self.habit_predictor.get_habit_context_for_prompt(person_name, "person")

                # Get indicators relevant to this person
                person_indicators = self._get_person_indicators(person_name, active_indicators, context)

                # Build context for LLM
                context_for_llm = {
                    **context,
                    "habit_hint": habit_context,
                    "active_indicators": [
                        f"{i['hint_type']}: {i['entity_id']} in {i['room']}" for i in person_indicators
                    ],
                }

                # Query LLM
                guess = await self.provider.deduce_presence(
                    hass=self.hass,
                    context=context_for_llm,
                    entity_name=person_name,
                    entity_type="person",
                    rooms=self.rooms,
                )

                # Combine confidence from multiple sources
                final_room, final_confidence, explanation = self.confidence_combiner.combine(
                    llm_room=guess.room,
                    llm_confidence=guess.confidence,
                    habit_room=habit_hint.get("predicted_room"),
                    habit_confidence=habit_hint.get("confidence", 0),
                    sensor_indicators=person_indicators,
                    entity_name=person_name,
                    room_entities=self.room_entities,
                )

                # Update guess with combined result
                if guess.room != final_room:
                    _LOGGER.debug(
                        "Confidence combiner changed %s from %s to %s (reason: %s)",
                        person_name,
                        guess.room,
                        final_room,
                        explanation,
                    )
                guess.room = final_room
                guess.confidence = final_confidence
                guess.indicators = (
                    [f"{i['hint_type']}: {i['entity_id'].split('.')[-1]}" for i in person_indicators[:5]]
                    + [explanation]
                    if explanation
                    else []
                )

                # Learn from this event if confident
                if self.learning_enabled and final_confidence >= 0.7:
                    self.habit_predictor.learn_from_event(
                        entity_name=person_name,
                        room=final_room,
                        confidence=final_confidence,
                    )

                # Log the event
                self._log_presence_event(person_name, "person", guess, context)

                # Check for room transitions
                self._check_room_transition(person_name, guess.room, guess.confidence)

                results["persons"][person_name] = guess

            # Process pets similarly
            for pet in self.pets:
                pet_name = pet.get("name", "unknown")

                habit_hint = self.habit_predictor.get_habit_hint(pet_name, "pet")
                habit_context = self.habit_predictor.get_habit_context_for_prompt(pet_name, "pet")

                # Pets don't have owned devices, use general indicators
                pet_indicators = [i for i in active_indicators if i.get("hint_type") != ENTITY_HINT_COMPUTER]

                context_for_llm = {
                    **context,
                    "habit_hint": habit_context,
                    "active_indicators": [f"{i['hint_type']}: {i['entity_id']} in {i['room']}" for i in pet_indicators],
                }

                guess = await self.provider.deduce_presence(
                    hass=self.hass,
                    context=context_for_llm,
                    entity_name=pet_name,
                    entity_type="pet",
                    rooms=self.rooms,
                )

                final_room, final_confidence, explanation = self.confidence_combiner.combine(
                    llm_room=guess.room,
                    llm_confidence=guess.confidence,
                    habit_room=habit_hint.get("predicted_room"),
                    habit_confidence=habit_hint.get("confidence", 0),
                    sensor_indicators=pet_indicators,
                    entity_name=pet_name,
                    room_entities=self.room_entities,
                )

                if guess.room != final_room:
                    _LOGGER.debug(
                        "Confidence combiner changed %s from %s to %s (reason: %s)",
                        pet_name,
                        guess.room,
                        final_room,
                        explanation,
                    )
                guess.room = final_room
                guess.confidence = final_confidence

                if self.learning_enabled and final_confidence >= 0.7:
                    self.habit_predictor.learn_from_event(
                        entity_name=pet_name,
                        room=final_room,
                        confidence=final_confidence,
                    )

                self._log_presence_event(pet_name, "pet", guess, context)
                self._check_room_transition(pet_name, guess.room, guess.confidence)

                results["pets"][pet_name] = guess

            elapsed = time.time() - start_time
            _LOGGER.debug("Finished fetching whollm data in %.3f seconds (success: True)", elapsed)

            return results

        except Exception as err:
            _LOGGER.error("Error fetching LLM presence data: %s", err)
            raise UpdateFailed(f"Error communicating with LLM: {err}") from err

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
        """Gather current state of all relevant sensors."""
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

        # Get all entity states
        states = self.hass.states.async_all()

        for state in states:
            entity_id = state.entity_id
            last_changed = state.last_changed

            # Calculate time since change
            time_since_change = None
            if last_changed:
                delta = now - last_changed
                if delta.total_seconds() < 60:
                    time_since_change = f"{int(delta.total_seconds())}s ago"
                elif delta.total_seconds() < 3600:
                    time_since_change = f"{int(delta.total_seconds() / 60)}m ago"
                else:
                    time_since_change = f"{int(delta.total_seconds() / 3600)}h ago"

            # Categorize entities
            if entity_id.startswith("light."):
                context["lights"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                    "brightness": state.attributes.get("brightness"),
                }

            elif entity_id.startswith("binary_sensor.") and "motion" in entity_id.lower():
                context["motion"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }

            elif entity_id.startswith("binary_sensor.") and any(
                x in entity_id.lower() for x in ["door", "window", "contact"]
            ):
                context["doors"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }

            elif entity_id.startswith("media_player."):
                media_info = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }
                if state.attributes.get("media_title"):
                    media_info["playing"] = state.attributes.get("media_title")
                if state.attributes.get("app_name"):
                    media_info["app"] = state.attributes.get("app_name")
                context["media"][entity_id] = media_info

            elif entity_id.startswith(("device_tracker.", "person.")):
                context["device_trackers"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }

            # AI detection sensors (camera-based person/animal detection)
            elif entity_id.startswith("binary_sensor.") and any(
                x in entity_id.lower() for x in ["_person", "_animal", "_pet", "_human"]
            ):
                context["ai_detection"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }

        # Also include configured entities that might not match the patterns above
        for room, entities in self.room_entities.items():
            for entity_config in entities:
                entity_id = entity_config.get("entity_id")
                hint_type = entity_config.get("hint_type", "appliance")

                if entity_id and entity_id not in context.get("computers", {}):
                    state = self.hass.states.get(entity_id)
                    if state:
                        if hint_type == ENTITY_HINT_COMPUTER:
                            context["computers"][entity_id] = {
                                "state": state.state,
                                "room": room,
                            }

        return context

    def _log_presence_event(
        self,
        entity_name: str,
        entity_type: str,
        guess: Any,
        context: dict[str, Any],
    ) -> None:
        """Log presence event for ML training."""
        try:
            self.event_logger.log_presence_event(
                entity_name=entity_name,
                entity_type=entity_type,
                room=guess.room,
                confidence=guess.confidence,
                raw_response=guess.raw_response,
                indicators=guess.indicators,
                sensor_context=context,
                detection_method="llm",
            )
        except Exception as err:
            _LOGGER.warning("Failed to log presence event: %s", err)

    def _check_room_transition(
        self,
        entity_name: str,
        new_room: str,
        confidence: float,
    ) -> None:
        """Check if entity moved rooms and log transition."""
        previous_room = self._previous_rooms.get(entity_name)

        if previous_room and previous_room != new_room and new_room != "unknown":
            try:
                self.event_logger.log_room_transition(
                    entity_name=entity_name,
                    from_room=previous_room,
                    to_room=new_room,
                    confidence=confidence,
                )
                _LOGGER.debug("Room transition: %s moved from %s to %s", entity_name, previous_room, new_room)
            except Exception as err:
                _LOGGER.warning("Failed to log room transition: %s", err)

        self._previous_rooms[entity_name] = new_room
