"""CrewAI API provider for room presence detection.

Integrates with the homelab CrewAI API for intelligent presence detection
using Claude or Ollama models with better context awareness.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import aiohttp

from .base import BaseLLMProvider, PresenceGuess

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class CrewAIProvider(BaseLLMProvider):
    """CrewAI API provider implementation.
    
    This provider connects to the local CrewAI API (typically at port 8502)
    which provides access to Claude models with better context awareness
    for homelab-specific tasks.
    """
    
    def __init__(self, url: str = "http://192.168.1.100:8502", model: str = "llama3.2", **kwargs) -> None:
        """Initialize the CrewAI provider.
        
        Args:
            url: CrewAI API URL (default: http://192.168.1.100:8502)
            model: Model to use - for Ollama: "llama3.2", for Claude: "sonnet", "haiku", or "opus"
        """
        super().__init__(url, model, **kwargs)
        # Default to Ollama (use_claude=False) unless explicitly set
        self._use_claude = kwargs.get("use_claude", False)
    
    async def deduce_presence(
        self,
        hass: HomeAssistant,
        context: dict[str, Any],
        entity_name: str,
        entity_type: str,
        rooms: list[str],
    ) -> PresenceGuess:
        """Query CrewAI API to deduce room presence using the dedicated endpoint."""
        
        _LOGGER.debug(
            "=== CREWAI QUERY for %s (%s) via /presence/detect ===",
            entity_name, entity_type
        )
        
        try:
            async with aiohttp.ClientSession() as session:
                # Use the dedicated presence detection endpoint
                async with session.post(
                    f"{self.url}/presence/detect",
                    json={
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "sensor_context": context,
                        "rooms": rooms,
                        "use_claude": self._use_claude,
                        "model": self.model,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        _LOGGER.error("CrewAI API error: %s", response.status)
                        return self._create_fallback_guess(context, entity_name, entity_type, rooms)
                    
                    result = await response.json()
                    
                    room = result.get("room", "unknown")
                    confidence = result.get("confidence", 0.5)
                    reason = result.get("reason", "")
                    api_indicators = result.get("indicators", [])
                    model_used = result.get("model_used", "unknown")
                    
                    _LOGGER.debug(
                        "=== CREWAI RESPONSE for %s ===\nRoom: %s, Confidence: %.0f%%, Model: %s\nReason: %s",
                        entity_name, room, confidence * 100, model_used, reason
                    )
                    
                    # Gather additional indicators
                    indicators = self._extract_indicators(context, room, entity_name)
                    
                    # Add API indicators
                    for ind in api_indicators:
                        if ind not in indicators:
                            indicators.append(ind)
                    
                    # Add reason as an indicator
                    if reason:
                        indicators.insert(0, f"Model {model_used}: {reason}")
                    
                    return PresenceGuess(
                        room=room,
                        confidence=confidence,
                        raw_response=f"{room} ({confidence*100:.0f}%): {reason}",
                        indicators=indicators,
                    )
                    
        except aiohttp.ClientError as err:
            _LOGGER.error("Error communicating with CrewAI: %s", err)
            return self._create_fallback_guess(context, entity_name, entity_type, rooms)
        except TimeoutError:
            _LOGGER.warning("CrewAI timeout for %s - using fallback", entity_name)
            return self._create_fallback_guess(context, entity_name, entity_type, rooms, "Timeout")
        except Exception as err:
            _LOGGER.error("Unexpected error querying CrewAI: %s", err)
            return self._create_fallback_guess(context, entity_name, entity_type, rooms, str(err))
    
    async def test_connection(self) -> bool:
        """Test if CrewAI API is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "ok"
                    return False
        except Exception as err:
            _LOGGER.error("Failed to connect to CrewAI: %s", err)
            return False
    
    async def get_available_models(self) -> list[str]:
        """Get list of available models from CrewAI."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", {})
                        return list(models.keys())
                    return ["haiku", "sonnet", "opus"]
        except Exception:
            return ["haiku", "sonnet", "opus"]
    
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
        
        # Check media
        for entity_id, data in context.get("media", {}).items():
            if data.get("state") == "playing":
                media_room = "unknown"
                if "living" in entity_id.lower() or "tv" in entity_id.lower():
                    media_room = "living_room"
                elif "bedroom" in entity_id.lower():
                    media_room = "bedroom"
                elif "office" in entity_id.lower():
                    media_room = "office"
                
                if media_room == room.lower().replace(" ", "_"):
                    indicators.append(f"Media playing: {entity_id}")
        
        # Check computers/PCs
        for entity_id, data in context.get("computers", {}).items():
            if data.get("state") in ["on", "home"] and room_lower == "office":
                friendly_name = data.get("friendly_name", entity_id)
                indicators.append(f"PC active: {friendly_name}")
        
        # Check device trackers for PC
        for entity_id, data in context.get("device_trackers", {}).items():
            if "pc" in entity_id.lower() and data.get("state") == "home" and room_lower == "office":
                indicators.append(f"PC online: {entity_id}")
        
        # Check camera AI detection
        for entity_id, data in context.get("ai_detection", {}).items():
            if data.get("state") == "on":
                camera = data.get("camera", "unknown")
                detection_type = data.get("detection_type", "unknown")
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
        """Create a fallback guess based on strong indicators when API fails."""
        indicators = []
        entity_lower = entity_name.lower()
        
        # Check if PC is on
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
        confidence = 0.3
        
        if entity_type == "person":
            # Use strong indicators for any person
            if pc_is_on:
                room = "office"
                confidence = 0.5
                indicators.append("PC is active (fallback)")
            elif living_room_tv_on:
                room = "living_room"
                confidence = 0.55
                indicators.append("TV playing (fallback)")
            else:
                room = "living_room"
                confidence = 0.3
        elif entity_type == "pet":
            if living_room_tv_on:
                room = "living_room"
                confidence = 0.4
                indicators.append("TV on - pet with family")
            elif pc_is_on:
                room = "office"
                confidence = 0.35
                indicators.append("PC active - pet may be with owner")
            else:
                room = "bedroom"
                confidence = 0.3
        
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
