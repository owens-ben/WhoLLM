"""WhoLLM integration for Home Assistant.

Uses local LLMs (Ollama) to intelligently deduce room presence
based on sensor data (lights, motion, media, device trackers).
Optionally uses vision models to identify specific people/pets.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_CONFIDENCE_WEIGHTS,
    CONF_MAX_FILE_SIZE_MB,
    CONF_RETENTION_DAYS,
    CONF_URL,
    CONF_VISION_MODEL,
    DEFAULT_CONFIDENCE_WEIGHTS,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_VISION_MODEL,
    DOMAIN,
)
from .coordinator import LLMPresenceCoordinator
from .event_logger import EventLogger
from .habits import ConfidenceCombiner, HabitPredictor
from .history import HistoryTracker
from .services import (
    SERVICE_CLEANUP_STORAGE,
    SERVICE_DISABLE_TRACKING,
    SERVICE_ENABLE_TRACKING,
    SERVICE_IDENTIFY_PERSON,
    SERVICE_REFRESH,
    _get_coordinators,
    async_setup_services,
    async_unload_services,
)
from .vision import CameraTrackingController, VisionIdentifier

if TYPE_CHECKING:
    from homeassistant.helpers.typing import ConfigType

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BINARY_SENSOR]

# Service schemas

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

    # Instantiate shared components (no global singletons)
    retention_days = entry.options.get(CONF_RETENTION_DAYS, entry.data.get(CONF_RETENTION_DAYS, DEFAULT_RETENTION_DAYS))
    max_file_size_mb = entry.options.get(
        CONF_MAX_FILE_SIZE_MB, entry.data.get(CONF_MAX_FILE_SIZE_MB, DEFAULT_MAX_FILE_SIZE_MB)
    )
    # These constructors touch the filesystem, so keep them off the event loop.
    # Store data under HA's config dir so non-/config installs (HA Core) work.
    config_dir = Path(hass.config.config_dir)
    event_logger = await hass.async_add_executor_job(
        lambda: EventLogger(
            log_path=str(config_dir / "whollm_events.jsonl"),
            retention_days=retention_days,
            max_file_size_mb=max_file_size_mb,
        )
    )
    habit_predictor = await hass.async_add_executor_job(HabitPredictor, config_dir)
    history_tracker = await hass.async_add_executor_job(HistoryTracker, config_dir)
    confidence_weights = entry.options.get(
        CONF_CONFIDENCE_WEIGHTS, entry.data.get(CONF_CONFIDENCE_WEIGHTS, DEFAULT_CONFIDENCE_WEIGHTS)
    )
    confidence_combiner = ConfidenceCombiner(confidence_weights)

    coordinator = LLMPresenceCoordinator(
        hass,
        entry,
        event_logger=event_logger,
        habit_predictor=habit_predictor,
        confidence_combiner=confidence_combiner,
        history_tracker=history_tracker,
    )
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator
    hass.data[DOMAIN]["event_logger"] = event_logger
    hass.data[DOMAIN]["habit_predictor"] = habit_predictor
    hass.data[DOMAIN]["history_tracker"] = history_tracker

    # Perform initial cleanup on startup
    _LOGGER.info("Performing initial storage cleanup on startup...")
    cleanup_result = await hass.async_add_executor_job(event_logger.cleanup)
    _LOGGER.info(
        "Startup cleanup complete: deleted %d events, kept %d, final size: %.1f MB",
        cleanup_result.get("total_deleted", 0),
        cleanup_result.get("total_kept", 0),
        cleanup_result.get("final_size_mb", 0.0),
    )

    async def periodic_cleanup(now) -> None:
        """Perform periodic cleanup."""
        _LOGGER.info("Running periodic storage cleanup...")
        logger = hass.data[DOMAIN].get("event_logger")
        if logger:
            result = await hass.async_add_executor_job(logger.cleanup)
            _LOGGER.info(
                "Periodic cleanup complete: deleted %d events, kept %d, final size: %.1f MB",
                result.get("total_deleted", 0),
                result.get("total_kept", 0),
                result.get("final_size_mb", 0.0),
            )

    # Apply options changes (Configure dialog) without a full HA restart
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    # Schedule daily cleanup; cancelled automatically when the entry unloads
    entry.async_on_unload(async_track_time_interval(hass, periodic_cleanup, timedelta(hours=24)))
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
    if not hass.services.has_service(DOMAIN, SERVICE_REFRESH):
        await async_setup_services(hass)

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
        result = await hass.async_add_executor_job(event_logger.cleanup)

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
            supports_response=SupportsResponse.OPTIONAL,
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
            SERVICE_CLEANUP_STORAGE,
            handle_cleanup_storage,
            schema=CLEANUP_STORAGE_SCHEMA,
            supports_response=SupportsResponse.OPTIONAL,
        )

        _LOGGER.info("Registered WhoLLM services")


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading WhoLLM integration")

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        coordinator = hass.data[DOMAIN].get(entry.entry_id)
        if coordinator and hasattr(coordinator, "provider"):
            await coordinator.provider.async_close()
        if coordinator and getattr(coordinator, "history_tracker", None):
            await hass.async_add_executor_job(coordinator.history_tracker.save)
        hass.data[DOMAIN].pop(entry.entry_id, None)

        if not _get_coordinators(hass):
            await async_unload_services(hass)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)
