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
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        _LOGGER.error("Ollama API error: %s", response.status)
                        return PresenceGuess(
                            room="unknown",
                            confidence=0.0,
                            raw_response=f"API error: {response.status}",
                            indicators=[],
                        )
                    
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
                    indicators = self._extract_indicators(context, room)
                    
                    return PresenceGuess(
                        room=room,
                        confidence=confidence,
                        raw_response=raw_answer,
                        indicators=indicators,
                    )
                    
        except aiohttp.ClientError as err:
            _LOGGER.error("Error communicating with Ollama: %s", err)
            return PresenceGuess(
                room="unknown",
                confidence=0.0,
                raw_response=str(err),
                indicators=[],
            )
        except Exception as err:
            _LOGGER.error("Unexpected error querying Ollama: %s", err)
            return PresenceGuess(
                room="unknown",
                confidence=0.0,
                raw_response=str(err),
                indicators=[],
            )
    
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
            if room_lower in entity_id.lower() and data.get("state") == "playing":
                indicators.append(f"Media playing: {entity_id}")
        
        # Check computers/PCs - PC on is a strong office indicator
        if room_lower == "office":
            for entity_id, data in context.get("computers", {}).items():
                if data.get("state") in ["on", "home"]:
                    friendly_name = data.get("friendly_name", entity_id)
                    indicators.append(f"PC active: {friendly_name}")
        
        return indicators


