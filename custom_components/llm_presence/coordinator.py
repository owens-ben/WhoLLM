"""Data update coordinator for LLM Presence."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    CONF_MODEL,
    CONF_PERSONS,
    CONF_PETS,
    CONF_POLL_INTERVAL,
    CONF_PROVIDER,
    CONF_ROOMS,
    CONF_URL,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PROVIDER,
    DEFAULT_URL,
    DOMAIN,
    VALID_ROOMS,
)
from .providers import get_provider
from .event_logger import get_event_logger
from .habits import get_habit_predictor, get_confidence_combiner
from .ml_predictor import get_ml_predictor

_LOGGER = logging.getLogger(__name__)


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
        self.confidence_combiner = get_confidence_combiner()
        
        # Initialize ML predictor
        self.ml_predictor = get_ml_predictor(hass)
        
        # Track previous rooms for transition detection
        self._previous_rooms: dict[str, str] = {}
        
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=poll_interval),
        )
        
        _LOGGER.info(
            "LLM Presence coordinator initialized: provider=%s, url=%s, model=%s, event_logging=enabled",
            self.provider_type,
            self.provider_url,
            self.model,
        )

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from LLM provider."""
        try:
            # Gather sensor context from Home Assistant
            context = await self._gather_sensor_context()
            
            # Query LLM for each person/pet
            results: dict[str, Any] = {
                "persons": {},
                "pets": {},
                "last_context": context,
            }
            
            # Pre-compute shared context for all persons
            media_context = self._get_media_context(context)
            pc_is_on = self._is_pc_on(context)
            
            for person in self.persons:
                person_name = person.get("name", "unknown")
                
                # Get habit hint for this person
                habit_hint = self.habit_predictor.get_habit_hint(person_name, "person")
                habit_context = self.habit_predictor.get_habit_context_for_prompt(person_name, "person")
                
                # Add behavioral indicators to context
                behavioral_indicators = self._get_behavioral_indicators(
                    person_name, "person", context, media_context, pc_is_on
                )
                
                # Add habit context and behavioral indicators to the sensor context for the LLM
                context_with_habits = {
                    **context, 
                    "habit_hint": habit_context,
                    "behavioral_indicators": behavioral_indicators,
                }
                
                guess = await self.provider.deduce_presence(
                    hass=self.hass,
                    context=context_with_habits,
                    entity_name=person_name,
                    entity_type="person",
                    rooms=self.rooms,
                )
                
                # Add behavioral indicators to the guess
                guess.indicators.extend(behavioral_indicators)
                
                # Combine confidence from multiple sources (with new media/PC context)
                final_room, final_confidence, explanation = self.confidence_combiner.combine(
                    llm_room=guess.room,
                    llm_confidence=guess.confidence,
                    habit_room=habit_hint.get("predicted_room"),
                    habit_confidence=habit_hint.get("confidence", 0),
                    sensor_indicators=guess.indicators,
                    camera_ai_room=self._get_camera_ai_room(context, "person"),
                    media_context=media_context,
                    entity_name=person_name,
                    pc_is_on=pc_is_on,
                )
                
                # Update guess with combined confidence
                guess.confidence = final_confidence
                if guess.room != final_room:
                    _LOGGER.debug(
                        "Confidence combiner changed %s from %s to %s (reason: %s)",
                        person_name, guess.room, final_room, explanation
                    )
                    guess.room = final_room
                
                # ML prediction as additional signal
                ml_result = self.ml_predictor.predict(
                    entity_name=person_name,
                    context=context,
                    llm_confidence=guess.confidence,
                    indicators=guess.indicators,
                )
                
                if ml_result['method'] == 'ml':
                    # If ML is confident and disagrees, consider it
                    if ml_result['confidence'] > guess.confidence and ml_result['room'] != guess.room:
                        _LOGGER.info(
                            "ML overriding %s: %s (%.0f%%) -> %s (%.0f%%)",
                            person_name, guess.room, guess.confidence * 100,
                            ml_result['room'], ml_result['confidence'] * 100
                        )
                        guess.room = ml_result['room']
                        guess.confidence = ml_result['confidence']
                        guess.indicators.append(f"🤖 ML prediction: {ml_result['room']}")
                    
                    # Send notification if uncertain
                    if ml_result['uncertain'] and guess.confidence < 0.5:
                        await self.ml_predictor.notify_uncertainty(
                            entity_name=person_name,
                            predicted_room=guess.room,
                            confidence=guess.confidence,
                            probabilities=ml_result['probabilities'],
                            context=context,
                        )
                
                # Log the event for ML training
                self._log_presence_event(person_name, "person", guess, context)
                
                # Check for room transitions
                self._check_room_transition(person_name, guess.room, guess.confidence)
                
                results["persons"][person_name] = guess
            
            for pet in self.pets:
                pet_name = pet.get("name", "unknown")
                
                # Get habit hint for this pet
                habit_hint = self.habit_predictor.get_habit_hint(pet_name, "pet")
                habit_context = self.habit_predictor.get_habit_context_for_prompt(pet_name, "pet")
                
                # Add behavioral indicators for pets (follows people)
                behavioral_indicators = self._get_behavioral_indicators(
                    pet_name, "pet", context, media_context, pc_is_on
                )
                
                context_with_habits = {
                    **context, 
                    "habit_hint": habit_context,
                    "behavioral_indicators": behavioral_indicators,
                }
                
                guess = await self.provider.deduce_presence(
                    hass=self.hass,
                    context=context_with_habits,
                    entity_name=pet_name,
                    entity_type="pet",
                    rooms=self.rooms,
                )
                
                # Add behavioral indicators to the guess
                guess.indicators.extend(behavioral_indicators)
                
                # Combine confidence (with new media/PC context for pets too)
                final_room, final_confidence, explanation = self.confidence_combiner.combine(
                    llm_room=guess.room,
                    llm_confidence=guess.confidence,
                    habit_room=habit_hint.get("predicted_room"),
                    habit_confidence=habit_hint.get("confidence", 0),
                    sensor_indicators=guess.indicators,
                    camera_ai_room=self._get_camera_ai_room(context, "animal"),
                    media_context=media_context,
                    entity_name=pet_name,
                    pc_is_on=pc_is_on,
                )
                
                guess.confidence = final_confidence
                if guess.room != final_room:
                    _LOGGER.debug(
                        "Confidence combiner changed %s from %s to %s (reason: %s)",
                        pet_name, guess.room, final_room, explanation
                    )
                    guess.room = final_room
                
                # ML prediction for pets
                ml_result = self.ml_predictor.predict(
                    entity_name=pet_name,
                    context=context,
                    llm_confidence=guess.confidence,
                    indicators=guess.indicators,
                )
                
                if ml_result['method'] == 'ml':
                    if ml_result['confidence'] > guess.confidence and ml_result['room'] != guess.room:
                        _LOGGER.info(
                            "ML overriding %s: %s (%.0f%%) -> %s (%.0f%%)",
                            pet_name, guess.room, guess.confidence * 100,
                            ml_result['room'], ml_result['confidence'] * 100
                        )
                        guess.room = ml_result['room']
                        guess.confidence = ml_result['confidence']
                        guess.indicators.append(f"🤖 ML prediction: {ml_result['room']}")
                    
                    # Pets: only notify if very uncertain (they're hard to track)
                    if ml_result['uncertain'] and guess.confidence < 0.3:
                        await self.ml_predictor.notify_uncertainty(
                            entity_name=pet_name,
                            predicted_room=guess.room,
                            confidence=guess.confidence,
                            probabilities=ml_result['probabilities'],
                            context=context,
                        )
                
                # Log the event
                self._log_presence_event(pet_name, "pet", guess, context)
                
                # Check for room transitions
                self._check_room_transition(pet_name, guess.room, guess.confidence)
                
                results["pets"][pet_name] = guess
            
            return results
            
        except Exception as err:
            _LOGGER.error("Error fetching LLM presence data: %s", err)
            raise UpdateFailed(f"Error communicating with LLM: {err}") from err

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
            "computers": {},  # PC/workstation indicators
            "ai_detection": {},  # AI-based person/animal detection from cameras
            "time_context": {
                "current_time": now.strftime("%H:%M"),
                "day_of_week": now.strftime("%A"),
                "is_night": now.hour >= 22 or now.hour < 6,
                "is_morning": 6 <= now.hour < 10,
                "is_evening": 18 <= now.hour < 22,
            },
        }
        
        # Get all entity states
        states = self.hass.states.async_all()
        
        for state in states:
            entity_id = state.entity_id
            last_changed = state.last_changed
            
            # Calculate how long ago the state changed
            time_since_change = None
            if last_changed:
                delta = now - last_changed
                if delta.total_seconds() < 60:
                    time_since_change = f"{int(delta.total_seconds())}s ago"
                elif delta.total_seconds() < 3600:
                    time_since_change = f"{int(delta.total_seconds() / 60)}m ago"
                else:
                    time_since_change = f"{int(delta.total_seconds() / 3600)}h ago"
            
            # Collect lights
            if entity_id.startswith("light."):
                context["lights"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                    "brightness": state.attributes.get("brightness"),
                }
            
            # Collect motion sensors
            elif entity_id.startswith("binary_sensor.") and "motion" in entity_id.lower():
                context["motion"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }
            
            # Collect door/window sensors
            elif entity_id.startswith("binary_sensor.") and any(x in entity_id.lower() for x in ["door", "window", "contact"]):
                context["doors"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }
            
            # Collect media players with more detail
            elif entity_id.startswith("media_player."):
                media_info = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }
                # Add what's playing if available
                if state.attributes.get("media_title"):
                    media_info["playing"] = state.attributes.get("media_title")
                if state.attributes.get("app_name"):
                    media_info["app"] = state.attributes.get("app_name")
                if state.attributes.get("source"):
                    media_info["source"] = state.attributes.get("source")
                context["media"][entity_id] = media_info
            
            # Collect device trackers and persons
            elif entity_id.startswith(("device_tracker.", "person.")):
                context["device_trackers"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                }
            
            # Collect climate/temperature sensors for room occupancy hints
            elif entity_id.startswith("sensor.") and any(x in entity_id.lower() for x in ["temperature", "humidity"]):
                # Only include room-specific sensors
                if any(room in entity_id.lower() for room in ["living", "bedroom", "office", "kitchen", "bathroom"]):
                    context["climate"][entity_id] = {
                        "state": state.state,
                        "unit": state.attributes.get("unit_of_measurement"),
                    }
            
            # Collect PC/computer indicators (switches, binary sensors, device trackers)
            elif any(x in entity_id.lower() for x in ["_pc", "pc_", "computer", "desktop", "workstation"]):
                if entity_id.startswith(("switch.", "binary_sensor.", "device_tracker.")):
                    context["computers"][entity_id] = {
                        "state": state.state,
                        "last_changed": time_since_change,
                        "friendly_name": state.attributes.get("friendly_name", ""),
                    }
            
            # Collect AI-based person/animal detection from cameras (e.g., E1 Zoom)
            # These are VERY strong indicators of presence!
            elif entity_id.startswith("binary_sensor.") and any(x in entity_id.lower() for x in ["_person", "_animal", "_pet", "_human"]):
                # Try to determine which room the camera is in from entity name
                camera_name = entity_id.replace("binary_sensor.", "").replace("_person", "").replace("_animal", "").replace("_pet", "").replace("_human", "")
                detection_type = "person" if "person" in entity_id.lower() or "human" in entity_id.lower() else "animal"
                context["ai_detection"][entity_id] = {
                    "state": state.state,
                    "last_changed": time_since_change,
                    "camera": camera_name,
                    "detection_type": detection_type,
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
                _LOGGER.debug(
                    "Room transition: %s moved from %s to %s",
                    entity_name, previous_room, new_room
                )
            except Exception as err:
                _LOGGER.warning("Failed to log room transition: %s", err)
        
        self._previous_rooms[entity_name] = new_room

    def _get_camera_ai_room(
        self,
        context: dict[str, Any],
        detection_type: str,
    ) -> str | None:
        """Get room where camera AI detected a person/animal."""
        ai_detections = context.get("ai_detection", {})
        
        for entity_id, data in ai_detections.items():
            if data.get("state") == "on":
                det_type = data.get("detection_type", "")
                if detection_type == "person" and det_type == "person":
                    # Try to map camera to room
                    camera = data.get("camera", "").lower()
                    return self._camera_to_room(camera)
                elif detection_type == "animal" and det_type == "animal":
                    camera = data.get("camera", "").lower()
                    return self._camera_to_room(camera)
        
        return None

    def _camera_to_room(self, camera_name: str) -> str | None:
        """Map camera name to room name."""
        # Common mappings - customize for your setup
        camera_room_map = {
            "e1_zoom": "living_room",  # Adjust based on where your E1 Zoom is
            "living": "living_room",
            "kitchen": "kitchen",
            "bedroom": "bedroom",
            "office": "office",
            "front": "entry",
            "entry": "entry",
            "hallway": "entry",
        }
        
        for key, room in camera_room_map.items():
            if key in camera_name:
                return room
        
        return None

    def _get_media_context(self, context: dict[str, Any]) -> dict[str, str]:
        """Extract media player states by room.
        
        Returns:
            Dict mapping room names to media state ('playing', 'paused', 'off')
        """
        media_context = {}
        
        for entity_id, data in context.get("media", {}).items():
            state = data.get("state", "off")
            entity_lower = entity_id.lower()
            
            # Map media player to room
            if "living" in entity_lower or "tv" in entity_lower:
                room = "living_room"
            elif "bedroom" in entity_lower:
                room = "bedroom"
            elif "office" in entity_lower:
                room = "office"
            elif "kitchen" in entity_lower:
                room = "kitchen"
            elif "bathroom" in entity_lower:
                room = "bathroom"
            else:
                continue
            
            # Only update if this state is stronger than existing
            current = media_context.get(room, "off")
            if state == "playing":
                media_context[room] = "playing"
            elif state == "paused" and current != "playing":
                media_context[room] = "paused"
        
        return media_context

    def _is_pc_on(self, context: dict[str, Any]) -> bool:
        """Check if any PC/computer is active.
        
        Returns:
            True if a PC is detected as on/home
        """
        computers = context.get("computers", {})
        for entity_id, data in computers.items():
            if data.get("state") in ["on", "home"]:
                return True
        
        # Also check device trackers for ping-based PC detection
        trackers = context.get("device_trackers", {})
        for entity_id, data in trackers.items():
            if "pc" in entity_id.lower() or "computer" in entity_id.lower():
                if data.get("state") == "home":
                    return True
        
        return False

    def _get_behavioral_indicators(
        self,
        entity_name: str,
        entity_type: str,
        context: dict[str, Any],
        media_context: dict[str, str],
        pc_is_on: bool,
    ) -> list[str]:
        """Generate behavioral indicators based on cross-person/cross-room logic.
        
        These are logical inferences like:
        - If a PC is on, someone is likely in the office
        - If TV is playing, someone is likely in the living room
        - Pets follow people
        
        Customize this method for your household's specific patterns.
        """
        indicators = []
        
        if entity_type == "person":
            # PC activity suggests office presence
            if pc_is_on:
                indicators.append("🎯 PC is active - likely someone in office")
            
            # Media playing suggests living room presence
            if media_context.get("living_room") == "playing":
                indicators.append("📺 Living room TV/media playing")
            if media_context.get("bedroom") == "playing":
                indicators.append("📺 Bedroom TV/media playing")
        
        elif entity_type == "pet":
            # Pets follow people and have time-based patterns
            time_ctx = context.get("time_context", {})
            
            if time_ctx.get("is_night"):
                indicators.append(f"🐾 Late night - {entity_name} likely in bedroom with owners")
            elif time_ctx.get("is_morning"):
                indicators.append(f"🐾 Morning - {entity_name} following people around")
            elif pc_is_on and media_context.get("living_room") == "playing":
                # People in different rooms - pet could be with either
                indicators.append(f"🐾 Activity in multiple rooms - {entity_name} could be in either")
            elif pc_is_on:
                indicators.append(f"🐾 PC active - {entity_name} may be in office with owner")
            elif media_context.get("living_room") == "playing":
                indicators.append(f"🐾 TV on - {entity_name} likely in living room with family")
        
        return indicators


