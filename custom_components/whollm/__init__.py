"""WhoLLM integration for Home Assistant.

Uses local LLMs (Ollama) to intelligently deduce room presence
based on sensor data (lights, motion, media, device trackers).
Optionally uses vision models to identify specific people/pets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_MAX_FILE_SIZE_MB,
    CONF_RETENTION_DAYS,
    CONF_URL,
    CONF_VISION_MODEL,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_VISION_MODEL,
    DOMAIN,
)
from .coordinator import LLMPresenceCoordinator
from .event_logger import get_event_logger
from .vision import CameraTrackingController, VisionIdentifier

if TYPE_CHECKING:
    from homeassistant.helpers.typing import ConfigType

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BINARY_SENSOR]

# Service schemas
SERVICE_IDENTIFY_PERSON = "identify_person"
SERVICE_ENABLE_TRACKING = "enable_tracking"
SERVICE_DISABLE_TRACKING = "disable_tracking"
SERVICE_REQUEST_VISION_UPDATE = "request_vision_update"
SERVICE_CLEANUP_STORAGE = "cleanup_storage"

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


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up WhoLLM from YAML configuration."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up WhoLLM from a config entry."""
    _LOGGER.info("Setting up WhoLLM integration")

    coordinator = LLMPresenceCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Initialize event logger with config
    retention_days = entry.options.get(CONF_RETENTION_DAYS, entry.data.get(CONF_RETENTION_DAYS, DEFAULT_RETENTION_DAYS))
    max_file_size_mb = entry.options.get(
        CONF_MAX_FILE_SIZE_MB, entry.data.get(CONF_MAX_FILE_SIZE_MB, DEFAULT_MAX_FILE_SIZE_MB)
    )
    event_logger = get_event_logger(retention_days=retention_days, max_file_size_mb=max_file_size_mb)
    hass.data[DOMAIN]["event_logger"] = event_logger

    # Perform initial cleanup on startup
    _LOGGER.info("Performing initial storage cleanup on startup...")
    cleanup_result = event_logger.cleanup()
    _LOGGER.info(
        "Startup cleanup complete: deleted %d events, kept %d, final size: %.1f MB",
        cleanup_result.get("total_deleted", 0),
        cleanup_result.get("total_kept", 0),
        cleanup_result.get("final_size_mb", 0.0),
    )

    # Schedule periodic cleanup (daily)
    from datetime import timedelta

    async def periodic_cleanup(now) -> None:
        """Perform periodic cleanup."""
        _LOGGER.info("Running periodic storage cleanup...")
        logger = hass.data[DOMAIN].get("event_logger")
        if logger:
            result = logger.cleanup()
            _LOGGER.info(
                "Periodic cleanup complete: deleted %d events, kept %d, final size: %.1f MB",
                result.get("total_deleted", 0),
                result.get("total_kept", 0),
                result.get("final_size_mb", 0.0),
            )

    # Schedule daily cleanup (runs every 24 hours)
    async_track_time_interval(
        hass,
        periodic_cleanup,
        timedelta(hours=24),
    )
    _LOGGER.info("Scheduled daily periodic cleanup")

    # Initialize vision components
    ollama_url = entry.data.get(CONF_URL, "http://localhost:11434")
    vision_model = entry.data.get(CONF_VISION_MODEL, DEFAULT_VISION_MODEL)

    # Get known persons/pets from config
    persons = [p.get("name") for p in entry.data.get("persons", [])]
    pets = [p.get("name") for p in entry.data.get("pets", [])]

    vision_identifier = VisionIdentifier(
        ollama_url=ollama_url,
        vision_model=vision_model,
        known_persons=persons if persons else None,
        known_pets=pets if pets else None,
    )
    tracking_controller = CameraTrackingController(hass)

    # Store vision components
    hass.data[DOMAIN]["vision_identifier"] = vision_identifier
    hass.data[DOMAIN]["tracking_controller"] = tracking_controller

    # Register services
    await _async_register_services(hass)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def _async_register_services(hass: HomeAssistant) -> None:
    """Register WhoLLM services."""

    async def handle_identify_person(call: ServiceCall) -> dict[str, Any]:
        """Handle the identify_person service call."""
        camera_entity_id = call.data["camera_entity_id"]
        detection_type = call.data.get("detection_type", "person")

        vision_identifier: VisionIdentifier = hass.data[DOMAIN].get("vision_identifier")
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

    async def handle_enable_tracking(call: ServiceCall) -> None:
        """Handle the enable_tracking service call."""
        camera_name = call.data["camera_name"]

        tracking_controller: CameraTrackingController = hass.data[DOMAIN].get("tracking_controller")
        if tracking_controller:
            await tracking_controller.enable_tracking(camera_name)

    async def handle_disable_tracking(call: ServiceCall) -> None:
        """Handle the disable_tracking service call."""
        camera_name = call.data["camera_name"]

        tracking_controller: CameraTrackingController = hass.data[DOMAIN].get("tracking_controller")
        if tracking_controller:
            await tracking_controller.disable_tracking(camera_name)

    async def handle_request_vision_update(call: ServiceCall) -> None:
        """Handle manual vision update request."""
        _LOGGER.info("Manual vision update requested")
        # This could trigger vision identification on all configured cameras
        # For now, just log it - can be expanded later

    async def handle_cleanup_storage(call: ServiceCall) -> dict[str, Any]:
        """Handle the cleanup_storage service call."""
        event_logger = hass.data[DOMAIN].get("event_logger")
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

        # Fire an event with the cleanup result
        hass.bus.async_fire(
            f"{DOMAIN}_storage_cleanup",
            {
                "deleted_by_time": result.get("deleted_by_time", 0),
                "deleted_by_size": result.get("deleted_by_size", 0),
                "total_deleted": result.get("total_deleted", 0),
                "total_kept": result.get("total_kept", 0),
                "final_size_mb": result.get("final_size_mb", 0.0),
            },
        )

        return {
            "success": True,
            "deleted_by_time": result.get("deleted_by_time", 0),
            "deleted_by_size": result.get("deleted_by_size", 0),
            "total_deleted": result.get("total_deleted", 0),
            "total_kept": result.get("total_kept", 0),
            "final_size_mb": result.get("final_size_mb", 0.0),
        }

    # Register services if not already registered
    if not hass.services.has_service(DOMAIN, SERVICE_IDENTIFY_PERSON):
        hass.services.async_register(
            DOMAIN,
            SERVICE_IDENTIFY_PERSON,
            handle_identify_person,
            schema=IDENTIFY_PERSON_SCHEMA,
        )

        hass.services.async_register(
            DOMAIN,
            SERVICE_ENABLE_TRACKING,
            handle_enable_tracking,
            schema=TRACKING_SCHEMA,
        )

        hass.services.async_register(
            DOMAIN,
            SERVICE_DISABLE_TRACKING,
            handle_disable_tracking,
            schema=TRACKING_SCHEMA,
        )

        hass.services.async_register(
            DOMAIN,
            SERVICE_REQUEST_VISION_UPDATE,
            handle_request_vision_update,
        )

        hass.services.async_register(
            DOMAIN,
            SERVICE_CLEANUP_STORAGE,
            handle_cleanup_storage,
            schema=CLEANUP_STORAGE_SCHEMA,
        )

        _LOGGER.info("Registered WhoLLM services")


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading WhoLLM integration")

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
