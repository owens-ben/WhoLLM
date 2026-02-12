"""Event logger for ML training data collection."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

# Default log file location
DEFAULT_LOG_PATH = "/config/whollm_events.jsonl"

# Storage management defaults
from .const import DEFAULT_RETENTION_DAYS, DEFAULT_MAX_FILE_SIZE_MB

DEFAULT_MAX_FILE_SIZE_BYTES = DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024


class EventLogger:
    """Log presence events to JSONL file for future ML training."""

    def __init__(
        self,
        log_path: str = DEFAULT_LOG_PATH,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ):
        """Initialize the event logger.

        Args:
            log_path: Path to the JSONL log file
            retention_days: Number of days to keep events (default: 30)
            max_file_size_mb: Maximum file size in MB before cleanup (default: 100)
        """
        self.log_path = Path(log_path)
        self.retention_days = retention_days
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._ensure_log_file()
        _LOGGER.info(
            "EventLogger initialized, logging to: %s (retention: %d days, max size: %d MB)",
            self.log_path,
            self.retention_days,
            max_file_size_mb,
        )

    def _ensure_log_file(self) -> None:
        """Ensure the log file and directory exist."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.touch()
        except OSError as err:
            _LOGGER.error("Failed to create log file: %s", err)

    async def async_log_presence_event(
        self,
        entity_name: str,
        entity_type: str,
        room: str,
        confidence: float,
        raw_response: str,
        indicators: list[str],
        sensor_context: dict[str, Any],
        detection_method: str = "llm",
    ) -> None:
        """Log a presence detection event (non-blocking)."""
        event = {
            "timestamp": dt_util.now().isoformat(),
            "entity_name": entity_name,
            "entity_type": entity_type,
            "room": room,
            "confidence": confidence,
            "raw_response": raw_response,
            "indicators": indicators,
            "detection_method": detection_method,
            "time_features": self._extract_time_features(),
            "sensor_summary": self._summarize_sensors(sensor_context),
        }

        await self._async_write_event(event)
        _LOGGER.debug("Logged presence event: %s -> %s", entity_name, room)

    async def async_log_vision_event(
        self,
        camera_entity: str,
        identified: str,
        confidence: str,
        description: str,
        detection_type: str,
    ) -> None:
        """Log a vision identification event (non-blocking)."""
        event = {
            "timestamp": dt_util.now().isoformat(),
            "event_type": "vision_identification",
            "camera_entity": camera_entity,
            "identified": identified,
            "confidence": confidence,
            "description": description,
            "detection_type": detection_type,
            "time_features": self._extract_time_features(),
        }

        await self._async_write_event(event)
        _LOGGER.debug("Logged vision event: %s identified %s", camera_entity, identified)

    async def async_log_room_transition(
        self,
        entity_name: str,
        from_room: str,
        to_room: str,
        confidence: float,
    ) -> None:
        """Log when someone moves between rooms (non-blocking)."""
        event = {
            "timestamp": dt_util.now().isoformat(),
            "event_type": "room_transition",
            "entity_name": entity_name,
            "from_room": from_room,
            "to_room": to_room,
            "confidence": confidence,
            "time_features": self._extract_time_features(),
        }

        await self._async_write_event(event)
        _LOGGER.debug("Logged transition: %s %s -> %s", entity_name, from_room, to_room)

    async def _async_write_event(self, event: dict[str, Any]) -> None:
        """Write a single event to the log file without blocking the event loop."""
        line = json.dumps(event) + "\n"
        try:
            await asyncio.to_thread(self._write_line, line)
        except OSError as err:
            _LOGGER.error("Failed to write event: %s", err)

    def _write_line(self, line: str) -> None:
        """Synchronous helper for file write (called via to_thread)."""
        with open(self.log_path, "a") as f:
            f.write(line)

    def _extract_time_features(self) -> dict[str, Any]:
        """Extract time-based features for ML training."""
        now = dt_util.now()
        return {
            "hour": now.hour,
            "minute": now.minute,
            "day_of_week": now.weekday(),  # 0=Monday, 6=Sunday
            "day_name": now.strftime("%A"),
            "is_weekend": now.weekday() >= 5,
            "is_night": now.hour >= 22 or now.hour < 6,
            "is_morning": 6 <= now.hour < 10,
            "is_afternoon": 10 <= now.hour < 17,
            "is_evening": 17 <= now.hour < 22,
        }

    def _summarize_sensors(self, context: dict[str, Any]) -> dict[str, Any]:
        """Create a compact summary of sensor state for ML features."""
        summary = {
            "lights_on": [],
            "motion_detected": [],
            "media_playing": [],
            "computers_on": [],
            "ai_person_detected": [],
            "ai_animal_detected": [],
            "doors_open": [],
            "trackers_home": [],
            "trackers_away": [],
        }

        # Lights that are on
        for entity_id, data in context.get("lights", {}).items():
            if data.get("state") == "on":
                room = entity_id.replace("light.", "").split("_")[0]
                summary["lights_on"].append(room)

        # Motion detected
        for entity_id, data in context.get("motion", {}).items():
            if data.get("state") == "on":
                room = entity_id.replace("binary_sensor.", "").replace("_motion", "").split("_")[0]
                summary["motion_detected"].append(room)

        # Media playing
        for entity_id, data in context.get("media", {}).items():
            if data.get("state") in ["playing", "paused"]:
                device = entity_id.replace("media_player.", "")
                summary["media_playing"].append(device)

        # Computers on
        for entity_id, data in context.get("computers", {}).items():
            if data.get("state") in ["on", "home"]:
                name = entity_id.split(".")[-1]
                summary["computers_on"].append(name)

        # AI detection
        for entity_id, data in context.get("ai_detection", {}).items():
            if data.get("state") == "on":
                camera = data.get("camera", entity_id)
                if data.get("detection_type") == "person":
                    summary["ai_person_detected"].append(camera)
                else:
                    summary["ai_animal_detected"].append(camera)

        # Doors open
        for entity_id, data in context.get("doors", {}).items():
            if data.get("state") == "on":
                door = entity_id.replace("binary_sensor.", "")
                summary["doors_open"].append(door)

        # Device trackers
        for entity_id, data in context.get("device_trackers", {}).items():
            name = entity_id.replace("device_tracker.", "").replace("person.", "")
            if data.get("state") == "home":
                summary["trackers_home"].append(name)
            elif data.get("state") in ["not_home", "away"]:
                summary["trackers_away"].append(name)

        return summary

    def get_event_count(self) -> int:
        """Get total number of logged events."""
        try:
            with open(self.log_path) as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def get_recent_events(self, count: int = 100) -> list[dict]:
        """Get the most recent events."""
        try:
            events = []
            with open(self.log_path) as f:
                for line in f:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return events[-count:]
        except OSError:
            return []

    def get_file_size(self) -> int:
        """Get current file size in bytes."""
        try:
            return self.log_path.stat().st_size if self.log_path.exists() else 0
        except OSError:
            return 0

    def get_file_size_mb(self) -> float:
        """Get current file size in MB."""
        return self.get_file_size() / (1024 * 1024)

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dict with file size, event count, and retention info
        """
        file_size = self.get_file_size()
        file_size_mb = file_size / (1024 * 1024)
        event_count = self.get_event_count()

        return {
            "file_path": str(self.log_path),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size_mb, 2),
            "event_count": event_count,
            "retention_days": self.retention_days,
            "max_file_size_mb": self.max_file_size_bytes / (1024 * 1024),
            "usage_percent": round(
                (file_size / self.max_file_size_bytes * 100) if self.max_file_size_bytes > 0 else 0, 1
            ),
        }

    def cleanup_old_events(self, days: int | None = None) -> dict[str, Any]:
        """Remove events older than retention period.

        Args:
            days: Override retention days (uses instance default if None)

        Returns:
            Dict with cleanup statistics
        """
        if not self.log_path.exists():
            return {"deleted": 0, "kept": 0, "error": None}

        retention_days = days if days is not None else self.retention_days
        cutoff_date = dt_util.now() - timedelta(days=retention_days)
        cutoff_iso = cutoff_date.isoformat()

        deleted_count = 0
        kept_count = 0
        error = None

        try:
            # Read all events
            events = []
            with open(self.log_path) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        event_timestamp = event.get("timestamp", "")
                        if event_timestamp >= cutoff_iso:
                            events.append(line.rstrip())
                            kept_count += 1
                        else:
                            deleted_count += 1
                    except (json.JSONDecodeError, KeyError):
                        # Skip malformed lines
                        deleted_count += 1
                        continue

            # Write back only kept events
            if deleted_count > 0:
                with open(self.log_path, "w") as f:
                    for event_line in events:
                        f.write(event_line + "\n")

                _LOGGER.info(
                    "Cleaned up %d old events, kept %d events (retention: %d days)",
                    deleted_count,
                    kept_count,
                    retention_days,
                )
            else:
                _LOGGER.debug("No old events to clean up")

        except OSError as err:
            error = str(err)
            _LOGGER.error("Failed to cleanup old events: %s", err)

        return {
            "deleted": deleted_count,
            "kept": kept_count,
            "error": error,
        }

    def cleanup_by_size(self, target_size_mb: float | None = None) -> dict[str, Any]:
        """Remove oldest events if file exceeds size limit.

        Args:
            target_size_mb: Target file size in MB (uses max_file_size_bytes if None)

        Returns:
            Dict with cleanup statistics
        """
        if not self.log_path.exists():
            return {"deleted": 0, "kept": 0, "error": None, "final_size_mb": 0.0}

        current_size = self.get_file_size()
        target_size_bytes = (target_size_mb * 1024 * 1024) if target_size_mb is not None else self.max_file_size_bytes

        if current_size <= target_size_bytes:
            return {
                "deleted": 0,
                "kept": 0,
                "error": None,
                "final_size_mb": current_size / (1024 * 1024),
            }

        deleted_count = 0
        kept_count = 0
        error = None

        try:
            # Read all events first (oldest to newest in file)
            all_events = []
            with open(self.log_path) as f:
                for line in f:
                    try:
                        line_size = len(line.encode("utf-8"))
                        all_events.append((line.rstrip(), line_size))
                    except (ValueError, UnicodeDecodeError):
                        deleted_count += 1
                        continue

            # Keep newest events that fit within target size
            # Work backwards from the end (newest events)
            events_to_keep = []
            total_size = 0

            for event_line, line_size in reversed(all_events):
                if total_size + line_size <= target_size_bytes:
                    events_to_keep.insert(0, event_line)  # Insert at beginning to maintain order
                    total_size += line_size
                    kept_count += 1
                else:
                    deleted_count += 1

            # Write back kept events (oldest to newest)
            if deleted_count > 0:
                with open(self.log_path, "w") as f:
                    for event_line in events_to_keep:
                        f.write(event_line + "\n")

                final_size = self.get_file_size()
                _LOGGER.info(
                    "Cleaned up %d events by size, kept %d events (target: %.1f MB, final: %.1f MB)",
                    deleted_count,
                    kept_count,
                    target_size_bytes / (1024 * 1024),
                    final_size / (1024 * 1024),
                )
            else:
                _LOGGER.debug("No events to clean up by size")

        except OSError as err:
            error = str(err)
            _LOGGER.error("Failed to cleanup events by size: %s", err)

        final_size = self.get_file_size()
        return {
            "deleted": deleted_count,
            "kept": kept_count,
            "error": error,
            "final_size_mb": final_size / (1024 * 1024),
        }

    def cleanup(self) -> dict[str, Any]:
        """Perform both time-based and size-based cleanup.

        Returns:
            Dict with combined cleanup statistics
        """
        _LOGGER.info("Starting storage cleanup...")

        # First, cleanup by retention days
        time_result = self.cleanup_old_events()

        # Then, cleanup by size if still too large
        size_result = self.cleanup_by_size()

        # Combine results
        total_deleted = time_result.get("deleted", 0) + size_result.get("deleted", 0)
        total_kept = time_result.get("kept", 0) + size_result.get("kept", 0)

        result = {
            "deleted_by_time": time_result.get("deleted", 0),
            "deleted_by_size": size_result.get("deleted", 0),
            "total_deleted": total_deleted,
            "total_kept": total_kept,
            "final_size_mb": size_result.get("final_size_mb", self.get_file_size_mb()),
            "error": time_result.get("error") or size_result.get("error"),
        }

        _LOGGER.info(
            "Storage cleanup complete: deleted %d events (time: %d, size: %d), kept %d, final size: %.1f MB",
            total_deleted,
            time_result.get("deleted", 0),
            size_result.get("deleted", 0),
            total_kept,
            result["final_size_mb"],
        )

        return result
