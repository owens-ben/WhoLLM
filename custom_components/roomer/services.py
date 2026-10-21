"""Home Assistant services for Roomer.

All Roomer services are registered here:
- roomer.refresh: Force immediate presence refresh
- roomer.correct_room: Manually correct a person's room
- roomer.get_history: Get movement history
- roomer.clear_history: Clear history data
- roomer.get_patterns: Get detected patterns
- roomer.identify_person: Identify a person/pet from a camera snapshot
- roomer.enable_tracking / roomer.disable_tracking: Toggle camera auto-tracking
- roomer.cleanup_storage: Clean up old event log data

Per-entry components (habit predictor, history tracker) are owned by the
coordinator stored at hass.data[DOMAIN][entry_id].
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN
from .habits import HabitPredictor
from .history import HistoryTracker

if TYPE_CHECKING:
    from .coordinator import LLMPresenceCoordinator
    from .vision import CameraTrackingController, VisionIdentifier

_LOGGER = logging.getLogger(__name__)


def get_habit_predictor(hass: HomeAssistant) -> HabitPredictor | None:
    """Return the habit predictor owned by the coordinator, if one is set up."""
    for value in hass.data.get(DOMAIN, {}).values():
        predictor = getattr(value, "habit_predictor", None)
        if isinstance(predictor, HabitPredictor):
            return predictor
    return None


def get_history_tracker(hass: HomeAssistant) -> HistoryTracker | None:
    """Return the history tracker owned by the coordinator, if one is set up."""
    for value in hass.data.get(DOMAIN, {}).values():
        tracker = getattr(value, "history_tracker", None)
        if isinstance(tracker, HistoryTracker):
            return tracker
    return None


# Service names
SERVICE_REFRESH = "refresh"
SERVICE_CORRECT_ROOM = "correct_room"
SERVICE_GET_HISTORY = "get_history"
SERVICE_CLEAR_HISTORY = "clear_history"
SERVICE_GET_PATTERNS = "get_patterns"
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

IDENTIFY_PERSON_SCHEMA = vol.Schema(
    {
        vol.Required("camera_entity_id"): cv.entity_id,
        vol.Optional("detection_type", default="person"): vol.In(["person", "animal"]),
    }
)

TRACKING_SCHEMA = vol.Schema(
    {
        vol.Required("camera_name"): cv.string,
    }
)

CLEANUP_STORAGE_SCHEMA = vol.Schema(
    {
        vol.Optional("retention_days"): vol.All(vol.Coerce(int), vol.Range(min=1, max=365)),
        vol.Optional("max_file_size_mb"): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=1000.0)),
    }
)


async def async_handle_refresh(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the refresh service call.

    Forces an immediate presence detection update.
    """
    _LOGGER.debug("Refresh service called with data: %s", call.data)

    # Get coordinator from hass.data
    coordinator: LLMPresenceCoordinator | None = None

    if DOMAIN in hass.data:
        for entry_data in hass.data[DOMAIN].values():
            if isinstance(entry_data, dict) and "coordinator" in entry_data:
                coordinator = entry_data["coordinator"]
                break
            elif hasattr(entry_data, "async_request_refresh"):
                coordinator = entry_data
                break

    if coordinator:
        await coordinator.async_request_refresh()
        _LOGGER.info("Presence refresh triggered")
    else:
        _LOGGER.warning("No Roomer coordinator found for refresh")


async def async_handle_correct_room(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the correct_room service call.

    Manually corrects a person's room and learns from the correction.
    """
    entity_id = call.data.get("entity_id")
    room = call.data.get("room")
    confidence = call.data.get("confidence", 1.0)

    _LOGGER.debug("Correct room service: %s -> %s (confidence: %s)", entity_id, room, confidence)

    # Extract entity name from entity_id (e.g., sensor.ben_room -> Ben)
    entity_name = entity_id.split(".")[-1].replace("_room", "").replace("_", " ").title()

    # Learn from the correction and save it right away, since corrections are rare
    predictor = get_habit_predictor(hass)
    if predictor:
        predictor.learn_from_event(
            entity_name=entity_name,
            room=room,
            confidence=confidence,
        )
        await asyncio.to_thread(predictor.save_learned_patterns)
    else:
        _LOGGER.warning("No habit predictor available; correction not learned")

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
    if tracker is None:
        _LOGGER.warning("No history tracker available")
        return []
    history = tracker.get_recent_transitions(entity_name, limit=limit)

    _LOGGER.debug("Retrieved %d history entries for %s", len(history), entity_name)

    return history


async def async_handle_clear_history(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the clear_history service call.

    Clears history data for an entity or all entities.
    """
    entity_name = call.data.get("entity_name")

    tracker = get_history_tracker(hass)
    if tracker is None:
        _LOGGER.warning("No history tracker available")
        return
    await asyncio.to_thread(tracker.clear_history, entity_name)

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
    if tracker is None:
        _LOGGER.warning("No history tracker available")
        return []
    patterns = tracker.detect_patterns(entity_name)

    _LOGGER.debug("Retrieved %d patterns for %s", len(patterns), entity_name)

    return patterns


async def async_handle_identify_person(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Handle the identify_person service call."""
    camera_entity_id = call.data["camera_entity_id"]
    detection_type = call.data.get("detection_type", "person")

    vision_identifier: VisionIdentifier | None = hass.data.get(DOMAIN, {}).get("vision_identifier")
    if not vision_identifier:
        _LOGGER.error("Vision identifier not initialized")
        return {"success": False, "error": "Vision not initialized"}

    result = await vision_identifier.identify_from_camera(
        hass,
        camera_entity_id,
        detection_type,
    )

    # Fire an event with the result
    hass.bus.async_fire(
        f"{DOMAIN}_vision_identification",
        {
            "camera": camera_entity_id,
            "detection_type": detection_type,
            **result,
        },
    )

    return result


async def async_handle_enable_tracking(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the enable_tracking service call."""
    tracking_controller: CameraTrackingController | None = hass.data.get(DOMAIN, {}).get("tracking_controller")
    if tracking_controller:
        await tracking_controller.enable_tracking(call.data["camera_name"])


async def async_handle_disable_tracking(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the disable_tracking service call."""
    tracking_controller: CameraTrackingController | None = hass.data.get(DOMAIN, {}).get("tracking_controller")
    if tracking_controller:
        await tracking_controller.disable_tracking(call.data["camera_name"])


async def async_handle_cleanup_storage(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Handle the cleanup_storage service call."""
    event_logger = hass.data.get(DOMAIN, {}).get("event_logger")
    if not event_logger:
        _LOGGER.error("Event logger not initialized")
        return {"success": False, "error": "Event logger not initialized"}

    # Override retention settings if provided
    retention_days = call.data.get("retention_days")
    max_file_size_mb = call.data.get("max_file_size_mb")

    if retention_days is not None:
        event_logger.retention_days = retention_days
    if max_file_size_mb is not None:
        event_logger.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    # Perform cleanup
    result = event_logger.cleanup()

    summary = {
        "deleted_by_time": result.get("deleted_by_time", 0),
        "deleted_by_size": result.get("deleted_by_size", 0),
        "total_deleted": result.get("total_deleted", 0),
        "total_kept": result.get("total_kept", 0),
        "final_size_mb": result.get("final_size_mb", 0.0),
    }

    # Fire an event with the cleanup result
    hass.bus.async_fire(f"{DOMAIN}_storage_cleanup", summary)

    return {"success": True, **summary}


async def async_setup_services(hass: HomeAssistant) -> None:
    """Set up Roomer services. Safe to call once per config entry."""

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

    async def identify_person_handler(call: ServiceCall) -> ServiceResponse:
        return await async_handle_identify_person(hass, call)

    async def enable_tracking_handler(call: ServiceCall) -> None:
        await async_handle_enable_tracking(hass, call)

    async def disable_tracking_handler(call: ServiceCall) -> None:
        await async_handle_disable_tracking(hass, call)

    async def cleanup_storage_handler(call: ServiceCall) -> ServiceResponse:
        return await async_handle_cleanup_storage(hass, call)

    services = [
        (SERVICE_REFRESH, refresh_handler, REFRESH_SCHEMA, SupportsResponse.NONE),
        (SERVICE_CORRECT_ROOM, correct_room_handler, CORRECT_ROOM_SCHEMA, SupportsResponse.NONE),
        (SERVICE_GET_HISTORY, get_history_handler, GET_HISTORY_SCHEMA, SupportsResponse.ONLY),
        (SERVICE_CLEAR_HISTORY, clear_history_handler, CLEAR_HISTORY_SCHEMA, SupportsResponse.NONE),
        (SERVICE_GET_PATTERNS, get_patterns_handler, GET_PATTERNS_SCHEMA, SupportsResponse.ONLY),
        (SERVICE_IDENTIFY_PERSON, identify_person_handler, IDENTIFY_PERSON_SCHEMA, SupportsResponse.OPTIONAL),
        (SERVICE_ENABLE_TRACKING, enable_tracking_handler, TRACKING_SCHEMA, SupportsResponse.NONE),
        (SERVICE_DISABLE_TRACKING, disable_tracking_handler, TRACKING_SCHEMA, SupportsResponse.NONE),
        (SERVICE_CLEANUP_STORAGE, cleanup_storage_handler, CLEANUP_STORAGE_SCHEMA, SupportsResponse.OPTIONAL),
    ]

    for service, handler, schema, supports_response in services:
        if hass.services.has_service(DOMAIN, service):
            continue
        hass.services.async_register(
            DOMAIN,
            service,
            handler,
            schema=schema,
            supports_response=supports_response,
        )

    _LOGGER.info("Roomer services registered")


async def async_unload_services(hass: HomeAssistant) -> None:
    """Unload Roomer services."""
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

    _LOGGER.info("Roomer services unloaded")
