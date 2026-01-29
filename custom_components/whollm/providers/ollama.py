"""Ollama LLM provider for room presence detection."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import aiohttp

from .base import BaseLLMProvider, PresenceGuess

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""
    
    async def deduce_presence(
        self,
        hass: HomeAssistant,
        context: dict[str, Any],
        entity_name: str,
        entity_type: str,
        rooms: list[str],
    ) -> PresenceGuess:
        """Query Ollama to deduce room presence."""
        prompt = self._format_context_for_prompt(context, entity_name, entity_type)
        system_prompt = self._get_system_prompt(entity_name, entity_type, rooms)
        
        # Log the prompt being sent to Ollama
        _LOGGER.debug(
            "=== OLLAMA QUERY for %s (%s) ===\nSYSTEM PROMPT:\n%s\n\nUSER PROMPT:\n%s",
            entity_name, entity_type, system_prompt, prompt
        )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent responses
                            "num_predict": 20,   # Short response expected
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=60),  # Increased from 30s to 60s
                ) as response:
                    if response.status != 200:
                        _LOGGER.error("Ollama API error: %s", response.status)
                        # On API error, use fallback based on strong indicators
                        return self._create_fallback_guess(context, entity_name, entity_type, rooms)
                    
                    result = await response.json()
                    raw_answer = result.get("response", "").strip()
                    answer = raw_answer.lower().replace(" ", "_")
                    
                    # Log the response from Ollama
                    _LOGGER.debug(
                        "=== OLLAMA RESPONSE for %s ===\nRaw: '%s'\nParsed: '%s'",
                        entity_name, raw_answer, answer
                    )
                    
                    # Validate response
                    room = "unknown"
                    confidence = 0.5
                    
                    if answer in rooms:
                        room = answer
                        confidence = 0.8
                    else:
                        # Try to match partial responses
                        for valid_room in rooms:
                            if valid_room in answer:
                                room = valid_room
                                confidence = 0.6
                                break
                    
                    # Gather indicators that contributed to this guess
                    indicators = self._extract_indicators(context, room, entity_name)
                    
                    return PresenceGuess(
                        room=room,
                        confidence=confidence,
                        raw_response=raw_answer,
                        indicators=indicators,
                    )
                    
        except aiohttp.ClientError as err:
            _LOGGER.error("Error communicating with Ollama: %s", err)
            # Use fallback on connection error
            return self._create_fallback_guess(context, entity_name, entity_type, rooms)
        except TimeoutError:
            _LOGGER.warning("Ollama timeout after 60s for %s - using fallback", entity_name)
            return self._create_fallback_guess(context, entity_name, entity_type, rooms, "Timeout after 60s")
        except Exception as err:
            _LOGGER.error("Unexpected error querying Ollama: %s", err)
            return self._create_fallback_guess(context, entity_name, entity_type, rooms, str(err))
    
    async def test_connection(self) -> bool:
        """Test if Ollama is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception as err:
            _LOGGER.error("Failed to connect to Ollama: %s", err)
            return False
    
    async def get_available_models(self) -> list[str]:
        """Get list of available models from Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    models = data.get("models", [])
                    return [m.get("name", "") for m in models if m.get("name")]
        except Exception as err:
            _LOGGER.error("Failed to get Ollama models: %s", err)
            return []
    
    def _extract_indicators(
        self,
        context: dict[str, Any],
        room: str,
        entity_name: str | None = None,
    ) -> list[str]:
        """Extract indicators that support the room guess."""
        indicators = []
        room_lower = room.lower().replace("_", " ")
        
        # Check lights
        for entity_id, data in context.get("lights", {}).items():
            if room_lower in entity_id.lower() and data.get("state") == "on":
                indicators.append(f"Light on: {entity_id}")
        
        # Check motion
        for entity_id, data in context.get("motion", {}).items():
            if room_lower in entity_id.lower() and data.get("state") == "on":
                indicators.append(f"Motion detected: {entity_id}")
        
        # Check media - strong indicator!
        for entity_id, data in context.get("media", {}).items():
            if data.get("state") == "playing":
                # Determine which room the media player is in
                media_room = "unknown"
                if "living" in entity_id.lower() or "tv" in entity_id.lower():
                    media_room = "living_room"
                elif "bedroom" in entity_id.lower():
                    media_room = "bedroom"
                elif "office" in entity_id.lower():
                    media_room = "office"
                
                if media_room == room.lower().replace(" ", "_"):
                    indicators.append(f"Media playing: {entity_id}")
        
        # Check computers/PCs - PC on is a strong office indicator
        pc_is_on = False
        for entity_id, data in context.get("computers", {}).items():
            if data.get("state") in ["on", "home"]:
                pc_is_on = True
                if room_lower == "office":
                    friendly_name = data.get("friendly_name", entity_id)
                    indicators.append(f"PC active: {friendly_name}")
        
        # Also check device trackers for PC
        for entity_id, data in context.get("device_trackers", {}).items():
            if "pc" in entity_id.lower() and data.get("state") == "home":
                pc_is_on = True
                if room_lower == "office":
                    indicators.append(f"PC online: {entity_id}")
        
        # Check camera AI detection
        for entity_id, data in context.get("ai_detection", {}).items():
            if data.get("state") == "on":
                camera = data.get("camera", "unknown")
                detection_type = data.get("detection_type", "unknown")
                # Map camera to room
                camera_room = self._camera_name_to_room(camera)
                if camera_room == room.lower().replace(" ", "_"):
                    indicators.append(f"{detection_type.title()} detected by camera: {camera}")
        
        return indicators

    def _camera_name_to_room(self, camera_name: str) -> str:
        """Map camera name to room."""
        camera_lower = camera_name.lower()
        if "living" in camera_lower or "e1" in camera_lower:
            return "living_room"
        elif "bedroom" in camera_lower:
            return "bedroom"
        elif "office" in camera_lower:
            return "office"
        elif "kitchen" in camera_lower:
            return "kitchen"
        elif "front" in camera_lower or "entry" in camera_lower:
            return "entry"
        return "unknown"

    def _create_fallback_guess(
        self,
        context: dict[str, Any],
        entity_name: str,
        entity_type: str,
        rooms: list[str],
        error_msg: str = "Fallback",
    ) -> PresenceGuess:
        """Create a fallback guess based on strong indicators when LLM fails.
        
        Uses deterministic logic based on sensor state to make a reasonable guess.
        """
        indicators = []
        entity_lower = entity_name.lower()
        
        # Check if PC is on (strong office indicator)
        pc_is_on = False
        for entity_id, data in context.get("computers", {}).items():
            if data.get("state") in ["on", "home"]:
                pc_is_on = True
                break
        for entity_id, data in context.get("device_trackers", {}).items():
            if "pc" in entity_id.lower() and data.get("state") == "home":
                pc_is_on = True
                break
        
        # Check media state
        living_room_tv_on = False
        for entity_id, data in context.get("media", {}).items():
            if ("living" in entity_id.lower() or "tv" in entity_id.lower()) and data.get("state") == "playing":
                living_room_tv_on = True
                break
        
        # Determine fallback room based on strong indicators
        room = "unknown"
        confidence = 0.3  # Low base confidence for fallback
        
        if entity_type == "person":
            # Use strong indicators for any person
            if pc_is_on:
                room = "office"
                confidence = 0.5
                indicators.append("PC is active (fallback logic)")
            elif living_room_tv_on:
                room = "living_room"
                confidence = 0.55
                indicators.append("Living room TV playing (fallback logic)")
            else:
                # Default to common areas
                room = "living_room"
                confidence = 0.3
                    
        elif entity_type == "pet":
            # Pets follow people
            if living_room_tv_on:
                room = "living_room"
                confidence = 0.4
                indicators.append("TV on - pet likely with family")
            elif pc_is_on:
                room = "office"
                confidence = 0.35
                indicators.append("PC active - pet may be with owner")
            else:
                room = "bedroom"
                confidence = 0.3
        
        # Extract additional indicators
        indicators.extend(self._extract_indicators(context, room, entity_name))
        
        _LOGGER.info(
            "Fallback guess for %s: %s (confidence: %.1f%%) - %s",
            entity_name, room, confidence * 100, error_msg
        )
        
        return PresenceGuess(
            room=room,
            confidence=confidence,
            raw_response=error_msg,
            indicators=indicators,
        )


