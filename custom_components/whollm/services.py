"""Home Assistant services for WhoLLM.

Provides services for manual control and data access:
- whollm.refresh: Force immediate presence refresh
- whollm.correct_room: Manually correct a person's room
- whollm.get_history: Get movement history
- whollm.clear_history: Clear history data
- whollm.get_patterns: Get detected patterns
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.helpers import config_validation as cv
from homeassistant.util import slugify

from .const import DOMAIN

if TYPE_CHECKING:
    from .coordinator import LLMPresenceCoordinator

_LOGGER = logging.getLogger(__name__)

# Service names
SERVICE_REFRESH = "refresh"
SERVICE_CORRECT_ROOM = "correct_room"
SERVICE_GET_HISTORY = "get_history"
SERVICE_CLEAR_HISTORY = "clear_history"
SERVICE_GET_PATTERNS = "get_patterns"

# Service schemas
REFRESH_SCHEMA = vol.Schema(
    {
        vol.Optional("entity_id"): cv.entity_id,
    }
)

CORRECT_ROOM_SCHEMA = vol.Schema(
    {
        vol.Required("entity_id"): cv.entity_id,
        vol.Required("room"): cv.string,
        vol.Optional("confidence", default=1.0): vol.Coerce(float),
    }
)

GET_HISTORY_SCHEMA = vol.Schema(
    {
        vol.Required("entity_name"): cv.string,
        vol.Optional("limit", default=20): vol.Coerce(int),
    }
)

CLEAR_HISTORY_SCHEMA = vol.Schema(
    {
        vol.Optional("entity_name"): cv.string,
    }
)

GET_PATTERNS_SCHEMA = vol.Schema(
    {
        vol.Required("entity_name"): cv.string,
    }
)


def _get_coordinator(hass: HomeAssistant) -> LLMPresenceCoordinator | None:
    """Return the first WhoLLM coordinator in hass.data, if any."""
    for entry_data in hass.data.get(DOMAIN, {}).values():
        if isinstance(entry_data, dict) and "coordinator" in entry_data:
            return entry_data["coordinator"]
        if hasattr(entry_data, "async_request_refresh"):
            return entry_data
    return None


def _entity_name_for(coordinator: LLMPresenceCoordinator, entity_id: str) -> str:
    """Map a room sensor entity_id (sensor.ben_room) back to its configured name."""
    object_id = entity_id.split(".")[-1].removesuffix("_room")
    for entity in [*coordinator.persons, *coordinator.pets]:
        name = entity.get("name", "")
        if slugify(name) == object_id:
            return name
    return object_id.replace("_", " ").title()


async def async_handle_refresh(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the refresh service call.

    Forces an immediate presence detection update.
    """
    _LOGGER.debug("Refresh service called with data: %s", call.data)

    coordinator = _get_coordinator(hass)
    if coordinator:
        await coordinator.async_request_refresh()
        _LOGGER.info("Presence refresh triggered")
    else:
        _LOGGER.warning("No WhoLLM coordinator found for refresh")


async def async_handle_correct_room(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the correct_room service call.

    Manually corrects a person's room and learns from the correction.
    """
    entity_id = call.data.get("entity_id")
    room = call.data.get("room")
    confidence = call.data.get("confidence", 1.0)

    _LOGGER.debug("Correct room service: %s -> %s (confidence: %s)", entity_id, room, confidence)

    coordinator = _get_coordinator(hass)
    if coordinator is None:
        _LOGGER.warning("No WhoLLM coordinator found for correct_room")
        return

    entity_name = _entity_name_for(coordinator, entity_id)

    # Learn from the correction
    coordinator.habit_predictor.learn_from_event(
        entity_name=entity_name,
        room=room,
        confidence=confidence,
    )

    # Update the state
    current_state = hass.states.get(entity_id)
    if current_state:
        new_attrs = dict(current_state.attributes)
        new_attrs["confidence"] = confidence
        new_attrs["source"] = "manual_correction"
        new_attrs["indicators"] = ["User corrected"]

        hass.states.async_set(entity_id, room, new_attrs)
        _LOGGER.info("Corrected %s room to %s", entity_name, room)
    else:
        _LOGGER.warning("Entity %s not found", entity_id)


async def async_handle_get_history(hass: HomeAssistant, call: ServiceCall) -> dict:
    """Handle the get_history service call.

    Returns recent room transitions for an entity.
    """
    entity_name = call.data.get("entity_name")
    limit = call.data.get("limit", 20)

    coordinator = _get_coordinator(hass)
    if coordinator is None:
        return {"transitions": []}

    history = coordinator.history_tracker.get_recent_transitions(entity_name, limit=limit)
    _LOGGER.debug("Retrieved %d history entries for %s", len(history), entity_name)

    return {"transitions": history}


async def async_handle_clear_history(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the clear_history service call.

    Clears history data for an entity or all entities.
    """
    entity_name = call.data.get("entity_name")

    coordinator = _get_coordinator(hass)
    if coordinator is None:
        return

    coordinator.history_tracker.clear_history(entity_name)

    if entity_name:
        _LOGGER.info("Cleared history for %s", entity_name)
    else:
        _LOGGER.info("Cleared all history")


async def async_handle_get_patterns(hass: HomeAssistant, call: ServiceCall) -> dict:
    """Handle the get_patterns service call.

    Returns detected movement patterns for an entity.
    """
    entity_name = call.data.get("entity_name")

    coordinator = _get_coordinator(hass)
    if coordinator is None:
        return {"patterns": []}

    patterns = coordinator.history_tracker.detect_patterns(entity_name)
    _LOGGER.debug("Retrieved %d patterns for %s", len(patterns), entity_name)

    return {"patterns": patterns}


async def async_setup_services(hass: HomeAssistant) -> None:
    """Set up WhoLLM services."""

    async def refresh_handler(call: ServiceCall) -> None:
        await async_handle_refresh(hass, call)

    async def correct_room_handler(call: ServiceCall) -> None:
        await async_handle_correct_room(hass, call)

    async def get_history_handler(call: ServiceCall) -> dict:
        return await async_handle_get_history(hass, call)

    async def clear_history_handler(call: ServiceCall) -> None:
        await async_handle_clear_history(hass, call)

    async def get_patterns_handler(call: ServiceCall) -> dict:
        return await async_handle_get_patterns(hass, call)

    # Register services
    hass.services.async_register(
        DOMAIN,
        SERVICE_REFRESH,
        refresh_handler,
        schema=REFRESH_SCHEMA,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_CORRECT_ROOM,
        correct_room_handler,
        schema=CORRECT_ROOM_SCHEMA,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_HISTORY,
        get_history_handler,
        schema=GET_HISTORY_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_HISTORY,
        clear_history_handler,
        schema=CLEAR_HISTORY_SCHEMA,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_PATTERNS,
        get_patterns_handler,
        schema=GET_PATTERNS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.info("WhoLLM services registered")


async def async_unload_services(hass: HomeAssistant) -> None:
    """Unload WhoLLM services."""
    hass.services.async_remove(DOMAIN, SERVICE_REFRESH)
    hass.services.async_remove(DOMAIN, SERVICE_CORRECT_ROOM)
    hass.services.async_remove(DOMAIN, SERVICE_GET_HISTORY)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_HISTORY)
    hass.services.async_remove(DOMAIN, SERVICE_GET_PATTERNS)

    _LOGGER.info("WhoLLM services unloaded")
