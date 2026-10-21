"""CrewAI API provider for room presence detection.

Integrates with the homelab CrewAI API for intelligent presence detection
using Claude or Ollama models with better context awareness.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

    def __init__(self, url: str = "http://localhost:8502", model: str = "llama3.2", **kwargs) -> None:
        """Initialize the CrewAI provider.

        Args:
            url: CrewAI API URL (default: http://localhost:8502)
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

        _LOGGER.debug("=== CREWAI QUERY for %s (%s) via /presence/detect ===", entity_name, entity_type)

        try:
            async def _do_request():
                session = self._get_session()
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
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status,
                        )
                    return await resp.json()

            result = await self._call_with_retry(_do_request)

            room = result.get("room", "unknown")
            confidence = result.get("confidence", 0.5)
            reason = result.get("reason", "")
            api_indicators = result.get("indicators", [])
            model_used = result.get("model_used", "unknown")

            _LOGGER.debug(
                "CrewAI response for %s: room=%s, confidence=%.0f%%, model=%s",
                entity_name, room, confidence * 100, model_used,
            )

            indicators = self._extract_indicators(context, room, entity_name)
            for ind in api_indicators:
                if ind not in indicators:
                    indicators.append(ind)
            if reason:
                indicators.insert(0, f"Model {model_used}: {reason}")

            return PresenceGuess(
                room=room,
                confidence=confidence,
                raw_response=f"{room} ({confidence * 100:.0f}%): {reason}",
                indicators=indicators,
            )

        except (aiohttp.ClientError, TimeoutError) as err:
            _LOGGER.warning("CrewAI request failed for %s: %s", entity_name, err)
            return self._create_fallback_guess(context, entity_name, entity_type, rooms, str(err))
        except Exception:
            _LOGGER.exception("Unexpected error querying CrewAI for %s", entity_name)
            return self._create_fallback_guess(context, entity_name, entity_type, rooms, "Unexpected error")

    async def test_connection(self) -> bool:
        """Test if CrewAI API is reachable."""
        try:
            session = self._get_session()
            async with session.get(
                f"{self.url}/",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "ok"
                return False
        except (aiohttp.ClientError, TimeoutError) as err:
            _LOGGER.error("Failed to connect to CrewAI: %s", err)
            return False

    async def get_available_models(self) -> list[str]:
        """Get list of available models from CrewAI."""
        try:
            session = self._get_session()
            async with session.get(
                f"{self.url}/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", {})
                    return list(models.keys())
                return ["haiku", "sonnet", "opus"]
        except (aiohttp.ClientError, TimeoutError):
            return ["haiku", "sonnet", "opus"]

    # _extract_indicators, _create_fallback_guess inherited from BaseLLMProvider
