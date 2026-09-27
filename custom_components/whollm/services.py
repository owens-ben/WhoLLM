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
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN

if TYPE_CHECKING:
    from .coordinator import LLMPresenceCoordinator
    from .habits import HabitPredictor
    from .history import HistoryTracker

_LOGGER = logging.getLogger(__name__)

# Service names
SERVICE_REFRESH = "refresh"
SERVICE_CORRECT_ROOM = "correct_room"
SERVICE_GET_HISTORY = "get_history"
SERVICE_CLEAR_HISTORY = "clear_history"
SERVICE_GET_PATTERNS = "get_patterns"
# Registered in __init__.py alongside the vision components they depend on
SERVICE_IDENTIFY_PERSON = "identify_person"
SERVICE_ENABLE_TRACKING = "enable_tracking"
SERVICE_DISABLE_TRACKING = "disable_tracking"
SERVICE_CLEANUP_STORAGE = "cleanup_storage"

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


def get_habit_predictor(hass: HomeAssistant) -> HabitPredictor | None:
    """Return the habit predictor created during entry setup."""
    return hass.data.get(DOMAIN, {}).get("habit_predictor")


def get_history_tracker(hass: HomeAssistant) -> HistoryTracker | None:
    """Return the history tracker created during entry setup."""
    return hass.data.get(DOMAIN, {}).get("history_tracker")


def _get_coordinators(hass: HomeAssistant) -> list[LLMPresenceCoordinator]:
    """Return every loaded WhoLLM coordinator."""
    return [v for v in hass.data.get(DOMAIN, {}).values() if hasattr(v, "async_request_refresh")]


async def async_handle_refresh(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the refresh service call.

    Forces an immediate presence detection update.
    """
    _LOGGER.debug("Refresh service called with data: %s", call.data)

    coordinators = _get_coordinators(hass)
    if not coordinators:
        _LOGGER.warning("No WhoLLM coordinator found for refresh")
        return

    for coordinator in coordinators:
        await coordinator.async_request_refresh()
    _LOGGER.info("Presence refresh triggered")


async def async_handle_correct_room(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the correct_room service call.

    Manually corrects a person's room and learns from the correction.
    """
    entity_id = call.data.get("entity_id")
    room = call.data.get("room")
    confidence = call.data.get("confidence", 1.0)

    _LOGGER.debug("Correct room service: %s -> %s (confidence: %s)", entity_id, room, confidence)

    # Extract entity name from entity_id (e.g., sensor.alice_room -> Alice)
    entity_name = entity_id.split(".")[-1].replace("_room", "").replace("_", " ").title()

    # Learn from the correction
    predictor = get_habit_predictor(hass)
    if predictor is not None:
        predictor.learn_from_event(
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


async def async_handle_get_history(hass: HomeAssistant, call: ServiceCall) -> list[dict]:
    """Handle the get_history service call.

    Returns recent room transitions for an entity.
    """
    entity_name = call.data.get("entity_name")
    limit = call.data.get("limit", 20)

    tracker = get_history_tracker(hass)
    history = tracker.get_recent_transitions(entity_name, limit=limit) if tracker else []

    _LOGGER.debug("Retrieved %d history entries for %s", len(history), entity_name)

    return history


async def async_handle_clear_history(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the clear_history service call.

    Clears history data for an entity or all entities.
    """
    entity_name = call.data.get("entity_name")

    tracker = get_history_tracker(hass)
    if tracker is None:
        _LOGGER.warning("History tracker not initialized")
        return
    await hass.async_add_executor_job(tracker.clear_history, entity_name)

    if entity_name:
        _LOGGER.info("Cleared history for %s", entity_name)
    else:
        _LOGGER.info("Cleared all history")


async def async_handle_get_patterns(hass: HomeAssistant, call: ServiceCall) -> list[dict]:
    """Handle the get_patterns service call.

    Returns detected movement patterns for an entity.
    """
    entity_name = call.data.get("entity_name")

    tracker = get_history_tracker(hass)
    patterns = tracker.detect_patterns(entity_name) if tracker else []

    _LOGGER.debug("Retrieved %d patterns for %s", len(patterns), entity_name)

    return patterns


async def async_setup_services(hass: HomeAssistant) -> None:
    """Set up WhoLLM services."""

    async def refresh_handler(call: ServiceCall) -> None:
        await async_handle_refresh(hass, call)

    async def correct_room_handler(call: ServiceCall) -> None:
        await async_handle_correct_room(hass, call)

    async def get_history_handler(call: ServiceCall) -> ServiceResponse:
        return {"transitions": await async_handle_get_history(hass, call)}

    async def clear_history_handler(call: ServiceCall) -> None:
        await async_handle_clear_history(hass, call)

    async def get_patterns_handler(call: ServiceCall) -> ServiceResponse:
        return {"patterns": await async_handle_get_patterns(hass, call)}

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
    """Unload all WhoLLM services."""
    for service in (
        SERVICE_REFRESH,
        SERVICE_CORRECT_ROOM,
        SERVICE_GET_HISTORY,
        SERVICE_CLEAR_HISTORY,
        SERVICE_GET_PATTERNS,
        SERVICE_IDENTIFY_PERSON,
        SERVICE_ENABLE_TRACKING,
        SERVICE_DISABLE_TRACKING,
        SERVICE_CLEANUP_STORAGE,
    ):
        hass.services.async_remove(DOMAIN, service)

    _LOGGER.info("WhoLLM services unloaded")
