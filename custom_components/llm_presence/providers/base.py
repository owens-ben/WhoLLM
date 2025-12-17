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
        lines = []
        
        # Time context first - very important for behavior patterns
        time_ctx = context.get("time_context", {})
        lines.append(f"CURRENT TIME: {time_ctx.get('current_time', 'unknown')} ({time_ctx.get('day_of_week', 'unknown')})")
        if time_ctx.get("is_night"):
            lines.append("  ** It is NIGHTTIME (10pm-6am) - people are likely sleeping or getting ready for bed **")
        elif time_ctx.get("is_morning"):
            lines.append("  ** It is MORNING (6am-10am) - people are likely waking up or getting ready **")
        elif time_ctx.get("is_evening"):
            lines.append("  ** It is EVENING (6pm-10pm) - people are likely relaxing, watching TV, or winding down **")
        
        lines.append("")
        
        # Lights - with recency info
        lines.append("LIGHTS (room: state, when changed):")
        for entity_id, data in context.get("lights", {}).items():
            room = entity_id.replace("light.", "").replace("_", " ").title()
            state = data.get('state', 'unknown')
            changed = data.get('last_changed', '')
            if state == 'on':
                brightness = data.get('brightness')
                if brightness:
                    pct = round(brightness / 255 * 100)
                    lines.append(f"  - {room}: ON ({pct}% brightness) - changed {changed}")
                else:
                    lines.append(f"  - {room}: ON - changed {changed}")
            elif state == 'off' and changed:
                lines.append(f"  - {room}: off - changed {changed}")
        
        # Motion - with recency (very important!)
        lines.append("\nMOTION SENSORS (recent motion is key indicator!):")
        for entity_id, data in context.get("motion", {}).items():
            room = entity_id.replace("binary_sensor.", "").replace("_motion", "").replace("_", " ").title()
            state = data.get("state")
            changed = data.get('last_changed', '')
            if state == "on":
                lines.append(f"  - {room}: MOTION DETECTED NOW!")
            else:
                lines.append(f"  - {room}: no motion (last detected {changed})")
        
        # AI Detection from cameras (VERY reliable - camera AI detected person/animal)
        if context.get("ai_detection"):
            lines.append("\nCAMERA AI DETECTION (very reliable - camera visually confirmed!):")
            for entity_id, data in context.get("ai_detection", {}).items():
                camera = data.get("camera", "unknown").replace("_", " ").title()
                detection_type = data.get("detection_type", "unknown")
                state = data.get("state")
                changed = data.get('last_changed', '')
                if state == "on":
                    if detection_type == "person":
                        lines.append(f"  - {camera}: 🚨 PERSON DETECTED NOW by camera AI!")
                    else:
                        lines.append(f"  - {camera}: 🐾 ANIMAL/PET DETECTED NOW by camera AI!")
                else:
                    lines.append(f"  - {camera}: no {detection_type} detected (last seen {changed})")
        
        # Doors - for entry/exit tracking
        if context.get("doors"):
            lines.append("\nDOORS/WINDOWS:")
            for entity_id, data in context.get("doors", {}).items():
                name = entity_id.replace("binary_sensor.", "").replace("_", " ").title()
                state = "OPEN" if data.get("state") == "on" else "closed"
                changed = data.get('last_changed', '')
                lines.append(f"  - {name}: {state} - changed {changed}")
        
        # Media - with what's playing (very strong indicator!)
        lines.append("\nMEDIA DEVICES (TV/speakers on = strong presence indicator):")
        for entity_id, data in context.get("media", {}).items():
            device = entity_id.replace("media_player.", "").replace("_", " ").title()
            state = data.get('state', 'unknown')
            changed = data.get('last_changed', '')
            
            if state == "playing":
                playing = data.get('playing', '')
                app = data.get('app', '')
                source = data.get('source', '')
                info_parts = [f"PLAYING"]
                if app:
                    info_parts.append(f"app={app}")
                if playing:
                    info_parts.append(f"'{playing}'")
                if source:
                    info_parts.append(f"source={source}")
                lines.append(f"  - {device}: {' '.join(info_parts)}")
            elif state == "paused":
                lines.append(f"  - {device}: PAUSED (someone was just watching) - changed {changed}")
            elif state == "idle":
                lines.append(f"  - {device}: idle - changed {changed}")
            elif state == "off":
                lines.append(f"  - {device}: off - changed {changed}")
        
        # Computers/PCs (very strong office indicator!)
        if context.get("computers"):
            lines.append("\nCOMPUTERS/WORKSTATIONS (PC on = person likely at desk in OFFICE!):")
            for entity_id, data in context.get("computers", {}).items():
                # Use friendly_name if available, otherwise parse entity_id
                friendly_name = data.get('friendly_name', '')
                name = friendly_name if friendly_name else entity_id.split(".")[-1].replace("_", " ").title()
                state = data.get('state', 'unknown')
                changed = data.get('last_changed', '')
                if state in ["on", "home"]:
                    lines.append(f"  - {name}: 🖥️ ON/ACTIVE - person is at their desk! - {changed}")
                else:
                    lines.append(f"  - {name}: off/away - {changed}")
        
        # Device Trackers
        lines.append("\nDEVICE TRACKERS (home/away status):")
        for entity_id, data in context.get("device_trackers", {}).items():
            device = entity_id.replace("device_tracker.", "").replace("person.", "").replace("_", " ").title()
            state = data.get('state', 'unknown')
            if state == "home":
                lines.append(f"  - {device}: HOME")
            elif state in ["not_home", "away"]:
                lines.append(f"  - {device}: AWAY/NOT HOME")
            else:
                lines.append(f"  - {device}: {state}")
        
        # Add habit hint if available
        habit_hint = context.get("habit_hint", "")
        if habit_hint:
            lines.append(habit_hint)
        
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
            return f"""You are a smart home presence detection AI. Determine which room the pet '{entity_name}' is most likely in based on sensor data.

REASONING RULES FOR PETS:
1. Pets follow their owners - if owners are in a room, pet is likely there too
2. At night, pets often sleep in bedrooms or their favorite spots
3. Motion in a dark room might be a pet moving around
4. Pets don't use TVs/media, but may be near warm electronics
5. If a door just opened, pet might have moved through it
6. Consider time of day - pets sleep more during certain hours

OUTPUT: Respond with ONLY ONE word from this list: {rooms_str}
No explanation, no punctuation, just the room name."""
        
        return f"""You are a smart home presence detection AI. Determine which room {entity_name} is most likely in based on sensor data.

REASONING RULES (in priority order):
1. DEVICE TRACKER: If their phone/device shows "not_home" or "away", respond "away"
2. CAMERA AI DETECTION: If camera AI detects "PERSON DETECTED NOW" - this is VERY reliable, person is definitely in that room!
3. TV/MEDIA PLAYING: If a room's TV is playing/paused, person is DEFINITELY there
4. PC/COMPUTER ON: If someone's PC is on/active, they're likely at their desk (office)
5. LIGHTS ON: Room with lights on is a STRONG indicator - if office lights are on, person is likely in office even at night
6. MOTION: Recent motion detection (within last few minutes) is a good indicator
7. CROSS-ROOM LOGIC: 
   - If office lights ON and bedroom lights OFF → person is in office
   - If bedroom lights ON and office lights OFF → person is in bedroom
   - If bathroom light ON → person might be in bathroom temporarily
8. TIME CONTEXT (use as TIE-BREAKER only, not primary indicator):
   - Late night with NO lights on anywhere: likely bedroom/sleeping
   - If bedroom TV just turned OFF at night: person is going to sleep

IMPORTANT: Camera AI detection and actual sensor data OVERRIDE time-of-day assumptions!
If camera sees a person in living room at 2am, trust the camera!

OUTPUT: Respond with ONLY ONE word from this list: {rooms_str}
No explanation, no punctuation, just the room name."""


