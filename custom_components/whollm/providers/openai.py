"""OpenAI API provider for room presence detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from .base import BaseLLMProvider, PresenceGuess, parse_llm_response

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for presence detection.
    
    Uses the OpenAI Chat Completions API (compatible with many services).
    """

    def __init__(self, url: str, model: str, **kwargs) -> None:
        """Initialize the OpenAI provider.
        
        Args:
            url: API endpoint URL (default: OpenAI's API)
            model: Model name (e.g., gpt-4o-mini, gpt-4o)
            api_key: Required API key for authentication
        """
        super().__init__(url or DEFAULT_OPENAI_URL, model or DEFAULT_OPENAI_MODEL, **kwargs)
        
        self.api_key = kwargs.get("api_key")
        if not self.api_key:
            raise ValueError("OpenAI provider requires an api_key")
        
        # Use /chat/completions endpoint
        self.endpoint = f"{self.url.rstrip('/')}/chat/completions"

    async def _call_api(self, messages: list[dict]) -> dict:
        """Call the OpenAI API.
        
        Args:
            messages: List of message dicts with role and content
            
        Returns:
            API response dict
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Low temperature for consistent responses
            "max_tokens": 150,  # Short response needed
        }
        
        session = self._get_session()
        async with session.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                _LOGGER.error("OpenAI API error %d: %s", response.status, error_text)
                raise aiohttp.ClientResponseError(
                    response.request_info, response.history, status=response.status,
                )

            return await response.json()

    async def deduce_presence(
        self,
        hass: HomeAssistant,
        context: dict[str, Any],
        entity_name: str,
        entity_type: str,
        rooms: list[str],
    ) -> PresenceGuess:
        """Query OpenAI to deduce room presence."""
        # Build the prompt
        prompt = self._format_context_for_prompt(context, entity_name, entity_type)
        
        messages = [
            {
                "role": "system",
                "content": """You are a home presence detection system. Based on sensor data, determine which room a person is most likely in.

Respond in this EXACT format:
ROOM: <room_name>
CONFIDENCE: <0.0 to 1.0>
REASON: <brief explanation>

Only use room names from the provided list. If uncertain, use "unknown" with low confidence.""",
            },
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = await self._call_api(messages)
            
            # Extract response content
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            _LOGGER.debug("OpenAI response for %s: %s", entity_name, content[:100])
            
            # Parse the response
            room, confidence, reason = parse_llm_response(content, rooms)
            
            return PresenceGuess(
                room=room,
                confidence=confidence,
                raw_response=content,
                indicators=[reason] if reason else [],
                source="openai",
            )
            
        except Exception as err:
            _LOGGER.warning("OpenAI API error for %s: %s", entity_name, err)
            return self._create_fallback_guess(context, entity_name, entity_type, rooms, str(err))

    async def test_connection(self) -> bool:
        """Test if the OpenAI API is reachable."""
        try:
            # Make a minimal API call to test connectivity
            messages = [{"role": "user", "content": "Say 'ok'"}]
            response = await self._call_api(messages)
            return "choices" in response
        except Exception as err:
            _LOGGER.warning("OpenAI connection test failed: %s", err)
            return False

    async def get_available_models(self) -> list[str]:
        """Get list of available OpenAI models.
        
        Returns commonly used models - actual availability depends on API key.
        """
        # OpenAI doesn't have a simple models endpoint like Ollama
        # Return common chat models
        return [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
