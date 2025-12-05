"""Base class for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


@dataclass
class PresenceGuess:
    """Result of a presence detection query."""
    
    room: str
    confidence: float  # 0.0 to 1.0
    raw_response: str
    indicators: list[str]  # What signals led to this guess
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "room": self.room,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "indicators": self.indicators,
        }


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, url: str, model: str, **kwargs) -> None:
        """Initialize the provider."""
        self.url = url
        self.model = model
        self._extra_config = kwargs
    
    @abstractmethod
    async def deduce_presence(
        self,
        hass: HomeAssistant,
        context: dict[str, Any],
        entity_name: str,
        entity_type: str,  # "person" or "pet"
        rooms: list[str],
    ) -> PresenceGuess:
        """Query the LLM to deduce room presence.
        
        Args:
            hass: Home Assistant instance
            context: Dictionary of sensor states (lights, motion, media, trackers)
            entity_name: Name of the person or pet
            entity_type: "person" or "pet"
            rooms: List of valid room names
            
        Returns:
            PresenceGuess with room, confidence, and supporting data
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the provider is reachable and working.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> list[str]:
        """Get list of available models from the provider.
        
        Returns:
            List of model names
        """
        pass
    
    def _format_context_for_prompt(
        self,
        context: dict[str, Any],
        entity_name: str,
        entity_type: str,
    ) -> str:
        """Format sensor context into a human-readable prompt.
        
        Can be overridden by subclasses for provider-specific formatting.
        """
        lines = [f"Current home sensor states for {entity_type} '{entity_name}':\n"]
        
        # Lights
        lines.append("LIGHTS:")
        for entity_id, data in context.get("lights", {}).items():
            room = entity_id.replace("light.", "").replace("_", " ").title()
            lines.append(f"  - {room}: {data.get('state', 'unknown')}")
        
        # Motion
        lines.append("\nMOTION SENSORS:")
        for entity_id, data in context.get("motion", {}).items():
            room = entity_id.replace("binary_sensor.", "").replace("_motion", "").replace("_", " ").title()
            status = "detected" if data.get("state") == "on" else "no motion"
            lines.append(f"  - {room}: {status}")
        
        # Media
        lines.append("\nMEDIA DEVICES:")
        for entity_id, data in context.get("media", {}).items():
            device = entity_id.replace("media_player.", "").replace("_", " ").title()
            lines.append(f"  - {device}: {data.get('state', 'unknown')}")
        
        # Device Trackers
        lines.append("\nDEVICE TRACKERS:")
        for entity_id, data in context.get("device_trackers", {}).items():
            device = entity_id.replace("device_tracker.", "").replace("person.", "").replace("_", " ").title()
            lines.append(f"  - {device}: {data.get('state', 'unknown')}")
        
        return "\n".join(lines)
    
    def _get_system_prompt(
        self,
        entity_name: str,
        entity_type: str,
        rooms: list[str],
    ) -> str:
        """Get the system prompt for presence detection.
        
        Can be overridden by subclasses for provider-specific prompts.
        """
        rooms_str = ", ".join(rooms)
        
        if entity_type == "pet":
            return f"""You are a home presence detection assistant for pets. Based on sensor data, determine which room the pet '{entity_name}' is most likely in.

Rules for pets:
1. Pets tend to follow their owners or stay in warm/comfortable spots
2. Motion in a room without lights on might indicate pet movement
3. Pets don't use media devices, but may be near them for warmth
4. If no clear indicators, guess based on typical pet behavior patterns
5. Consider time of day - pets often sleep during certain hours

Respond with ONLY ONE of these exact words: {rooms_str}
No explanation, just the room name."""
        
        return f"""You are a home presence detection assistant. Based on sensor data, determine which room {entity_name} is most likely in.

Rules:
1. Lights ON is a strong indicator of presence
2. Motion sensors can timeout when sitting still - don't rely on them alone
3. Media playing (TV, computer) is a very strong indicator
4. If device tracker shows "not_home" or "away", respond with "away"
5. Consider cross-room signals - if office lights are on and bedroom lights are off, person is likely in office

Respond with ONLY ONE of these exact words: {rooms_str}
No explanation, just the room name."""


