"""Anthropic Claude API provider for room presence detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from .base import BaseLLMProvider, PresenceGuess, parse_llm_response

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider for presence detection.
    
    Uses the Anthropic Messages API.
    """

    def __init__(self, url: str, model: str, **kwargs) -> None:
        """Initialize the Anthropic provider.
        
        Args:
            url: API endpoint URL (default: Anthropic's API)
            model: Model name (e.g., claude-3-haiku-20240307, claude-3-sonnet-20240229)
            api_key: Required API key for authentication
        """
        # Default to haiku if no model specified (cheap and fast)
        actual_model = model or DEFAULT_ANTHROPIC_MODEL
        super().__init__(url or DEFAULT_ANTHROPIC_URL, actual_model, **kwargs)
        
        self.api_key = kwargs.get("api_key")
        if not self.api_key:
            raise ValueError("Anthropic provider requires an api_key")
        
        # Use messages endpoint
        self.endpoint = f"{self.url.rstrip('/')}/v1/messages"

    async def _call_api(self, messages: list[dict], system: str = "") -> dict:
        """Call the Anthropic API.
        
        Args:
            messages: List of message dicts with role and content
            system: System prompt
            
        Returns:
            API response dict
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.3,
        }
        
        if system:
            payload["system"] = system
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("Anthropic API error %d: %s", response.status, error_text)
                    raise Exception(f"Anthropic API error: {response.status}")
                
                return await response.json()

    async def deduce_presence(
        self,
        hass: HomeAssistant,
        context: dict[str, Any],
        entity_name: str,
        entity_type: str,
        rooms: list[str],
    ) -> PresenceGuess:
        """Query Anthropic Claude to deduce room presence."""
        # Build the prompt
        prompt = self._format_context_for_prompt(context, entity_name, entity_type)
        
        system = """You are a home presence detection system. Based on sensor data, determine which room a person is most likely in.

Respond in this EXACT format:
ROOM: <room_name>
CONFIDENCE: <0.0 to 1.0>
REASON: <brief explanation>

Only use room names from the provided list. If uncertain, use "unknown" with low confidence."""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self._call_api(messages, system=system)
            
            # Extract response content
            content_blocks = response.get("content", [])
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content = block.get("text", "")
                    break
            
            _LOGGER.debug("Anthropic response for %s: %s", entity_name, content[:100])
            
            # Parse the response
            room, confidence, reason = parse_llm_response(content, rooms)
            
            return PresenceGuess(
                room=room,
                confidence=confidence,
                raw_response=content,
                indicators=[reason] if reason else [],
                source="anthropic",
            )
            
        except Exception as err:
            _LOGGER.warning("Anthropic API error for %s: %s", entity_name, err)
            return self._create_fallback_guess(entity_name, str(err))

    def _create_fallback_guess(self, entity_name: str, error: str) -> PresenceGuess:
        """Create a fallback guess when API fails."""
        return PresenceGuess(
            room="unknown",
            confidence=0.0,
            raw_response=f"API error: {error}",
            indicators=["Anthropic API unavailable"],
            source="anthropic_fallback",
        )

    async def test_connection(self) -> bool:
        """Test if the Anthropic API is reachable."""
        try:
            messages = [{"role": "user", "content": "Say 'ok'"}]
            response = await self._call_api(messages, system="Be brief.")
            return "content" in response
        except Exception as err:
            _LOGGER.warning("Anthropic connection test failed: %s", err)
            return False

    async def get_available_models(self) -> list[str]:
        """Get list of available Anthropic models."""
        return [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
        ]
