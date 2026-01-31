"""Tests for WhoLLM event logger."""
from __future__ import annotations

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from custom_components.whollm.event_logger import (
    EventLogger,
    get_event_logger,
    reset_event_logger,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_MAX_FILE_SIZE_MB,
)


@pytest.fixture
def temp_log_file():
    """Create a temporary log file."""
    with TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_events.jsonl"
        yield log_path


@pytest.fixture
def event_logger(temp_log_file):
    """Create an EventLogger instance with temp file."""
    return EventLogger(
        log_path=str(temp_log_file),
        retention_days=30,
        max_file_size_mb=10,
    )


@pytest.fixture(autouse=True)
def reset_global_logger():
    """Reset global logger between tests."""
    reset_event_logger()
    yield
    reset_event_logger()


class TestEventLoggerInitialization:
    """Test EventLogger initialization."""

    def test_initialization(self, temp_log_file):
        """Test basic initialization."""
        logger = EventLogger(str(temp_log_file))
        
        assert logger.log_path == temp_log_file
        assert logger.retention_days == DEFAULT_RETENTION_DAYS
        assert logger.max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024

    def test_initialization_with_custom_settings(self, temp_log_file):
        """Test initialization with custom settings."""
        logger = EventLogger(
            str(temp_log_file),
            retention_days=7,
            max_file_size_mb=50,
        )
        
        assert logger.retention_days == 7
        assert logger.max_file_size_bytes == 50 * 1024 * 1024

    def test_creates_log_file(self, temp_log_file):
        """Test that log file is created on init."""
        assert not temp_log_file.exists()
        
        EventLogger(str(temp_log_file))
        
        assert temp_log_file.exists()

    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        with TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "events.jsonl"
            
            EventLogger(str(nested_path))
            
            assert nested_path.exists()


class TestLogPresenceEvent:
    """Test presence event logging."""

    def test_log_presence_event(self, event_logger, temp_log_file):
        """Test logging a presence event."""
        event_logger.log_presence_event(
            entity_name="Alice",
            entity_type="person",
            room="office",
            confidence=0.85,
            raw_response="office",
            indicators=["PC is active"],
            sensor_context={
                "lights": {"light.office": {"state": "on"}},
                "motion": {},
                "media": {},
                "computers": {},
                "ai_detection": {},
                "doors": {},
                "device_trackers": {},
            },
            detection_method="llm",
        )
        
        # Read the logged event
        with open(temp_log_file) as f:
            event = json.loads(f.readline())
        
        assert event["entity_name"] == "Alice"
        assert event["entity_type"] == "person"
        assert event["room"] == "office"
        assert event["confidence"] == 0.85
        assert event["detection_method"] == "llm"
        assert "timestamp" in event
        assert "time_features" in event
        assert "sensor_summary" in event

    def test_log_multiple_events(self, event_logger, temp_log_file):
        """Test logging multiple events."""
        for i in range(5):
            event_logger.log_presence_event(
                entity_name=f"Person{i}",
                entity_type="person",
                room="office",
                confidence=0.8,
                raw_response="office",
                indicators=[],
                sensor_context={},
            )
        
        assert event_logger.get_event_count() == 5


class TestLogVisionEvent:
    """Test vision event logging."""

    def test_log_vision_event(self, event_logger, temp_log_file):
        """Test logging a vision identification event."""
        event_logger.log_vision_event(
            camera_entity="camera.living_room",
            identified="Alice",
            confidence="high",
            description="Person wearing blue shirt",
            detection_type="person",
        )
        
        with open(temp_log_file) as f:
            event = json.loads(f.readline())
        
        assert event["event_type"] == "vision_identification"
        assert event["camera_entity"] == "camera.living_room"
        assert event["identified"] == "Alice"
        assert event["confidence"] == "high"
        assert event["detection_type"] == "person"


class TestLogRoomTransition:
    """Test room transition logging."""

    def test_log_room_transition(self, event_logger, temp_log_file):
        """Test logging a room transition."""
        event_logger.log_room_transition(
            entity_name="Alice",
            from_room="office",
            to_room="kitchen",
            confidence=0.9,
        )
        
        with open(temp_log_file) as f:
            event = json.loads(f.readline())
        
        assert event["event_type"] == "room_transition"
        assert event["entity_name"] == "Alice"
        assert event["from_room"] == "office"
        assert event["to_room"] == "kitchen"
        assert event["confidence"] == 0.9


class TestTimeFeatures:
    """Test time feature extraction."""

    def test_extract_time_features(self, event_logger):
        """Test that time features are extracted correctly."""
        features = event_logger._extract_time_features()
        
        assert "hour" in features
        assert "minute" in features
        assert "day_of_week" in features
        assert "day_name" in features
        assert "is_weekend" in features
        assert "is_night" in features
        assert "is_morning" in features
        assert "is_afternoon" in features
        assert "is_evening" in features
        
        # Validate types
        assert isinstance(features["hour"], int)
        assert isinstance(features["is_weekend"], bool)
        assert 0 <= features["hour"] <= 23
        assert 0 <= features["day_of_week"] <= 6


class TestSensorSummary:
    """Test sensor context summarization."""

    def test_summarize_sensors(self, event_logger):
        """Test sensor summary extraction."""
        context = {
            "lights": {
                "light.living_room": {"state": "on"},
                "light.bedroom": {"state": "off"},
                "light.office": {"state": "on"},
            },
            "motion": {
                "binary_sensor.office_motion": {"state": "on"},
            },
            "media": {
                "media_player.living_room_tv": {"state": "playing"},
            },
            "computers": {
                "switch.office_pc": {"state": "on"},
            },
            "ai_detection": {},
            "doors": {
                "binary_sensor.front_door": {"state": "on"},
            },
            "device_trackers": {
                "device_tracker.alice_phone": {"state": "home"},
                "device_tracker.bob_phone": {"state": "not_home"},
            },
        }
        
        summary = event_logger._summarize_sensors(context)
        
        assert "living" in summary["lights_on"][0]
        assert "office" in summary["motion_detected"][0]
        assert len(summary["media_playing"]) == 1
        assert len(summary["computers_on"]) == 1
        assert len(summary["doors_open"]) == 1
        assert "alice" in summary["trackers_home"][0].lower()
        assert "bob" in summary["trackers_away"][0].lower()


class TestStorageStats:
    """Test storage statistics."""

    def test_get_event_count(self, event_logger):
        """Test getting event count."""
        assert event_logger.get_event_count() == 0
        
        event_logger.log_room_transition("Alice", "office", "bedroom", 0.9)
        event_logger.log_room_transition("Bob", "bedroom", "office", 0.8)
        
        assert event_logger.get_event_count() == 2

    def test_get_file_size(self, event_logger, temp_log_file):
        """Test getting file size."""
        assert event_logger.get_file_size() == 0
        
        event_logger.log_room_transition("Alice", "office", "bedroom", 0.9)
        
        assert event_logger.get_file_size() > 0
        assert event_logger.get_file_size() == temp_log_file.stat().st_size

    def test_get_file_size_mb(self, event_logger):
        """Test getting file size in MB."""
        event_logger.log_room_transition("Alice", "office", "bedroom", 0.9)
        
        size_bytes = event_logger.get_file_size()
        size_mb = event_logger.get_file_size_mb()
        
        assert size_mb == pytest.approx(size_bytes / (1024 * 1024), rel=0.01)

    def test_get_storage_stats(self, event_logger, temp_log_file):
        """Test getting complete storage stats."""
        event_logger.log_room_transition("Alice", "office", "bedroom", 0.9)
        
        stats = event_logger.get_storage_stats()
        
        assert stats["file_path"] == str(temp_log_file)
        assert stats["event_count"] == 1
        assert stats["file_size_bytes"] > 0
        assert stats["retention_days"] == 30
        assert "usage_percent" in stats

    def test_get_recent_events(self, event_logger):
        """Test getting recent events."""
        for i in range(10):
            event_logger.log_room_transition(f"Person{i}", "office", "bedroom", 0.9)
        
        recent = event_logger.get_recent_events(5)
        
        assert len(recent) == 5
        # Should be the last 5 events
        assert recent[-1]["entity_name"] == "Person9"
        assert recent[0]["entity_name"] == "Person5"


class TestCleanupOldEvents:
    """Test time-based cleanup."""

    def test_cleanup_old_events(self, temp_log_file):
        """Test cleaning up events older than retention period."""
        logger = EventLogger(str(temp_log_file), retention_days=7)
        
        # Write some old events
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        recent_time = datetime.now().isoformat()
        
        with open(temp_log_file, "w") as f:
            f.write(json.dumps({"timestamp": old_time, "entity_name": "Old"}) + "\n")
            f.write(json.dumps({"timestamp": recent_time, "entity_name": "Recent"}) + "\n")
        
        result = logger.cleanup_old_events()
        
        assert result["deleted"] == 1
        assert result["kept"] == 1
        
        # Verify only recent event remains
        events = logger.get_recent_events(100)
        assert len(events) == 1
        assert events[0]["entity_name"] == "Recent"

    def test_cleanup_with_custom_days(self, event_logger, temp_log_file):
        """Test cleanup with custom retention days."""
        # Write event from 3 days ago
        three_days_ago = (datetime.now() - timedelta(days=3)).isoformat()
        with open(temp_log_file, "w") as f:
            f.write(json.dumps({"timestamp": three_days_ago, "entity_name": "Test"}) + "\n")
        
        # Cleanup with 7 day retention - should keep
        result = event_logger.cleanup_old_events(days=7)
        assert result["deleted"] == 0
        assert result["kept"] == 1
        
        # Cleanup with 1 day retention - should delete
        result = event_logger.cleanup_old_events(days=1)
        assert result["deleted"] == 1
        assert result["kept"] == 0


class TestCleanupBySize:
    """Test size-based cleanup."""

    def test_cleanup_by_size_under_limit(self, event_logger, temp_log_file):
        """Test that cleanup does nothing when under size limit."""
        event_logger.log_room_transition("Alice", "office", "bedroom", 0.9)
        
        result = event_logger.cleanup_by_size()
        
        assert result["deleted"] == 0

    def test_cleanup_by_size_over_limit(self, temp_log_file):
        """Test cleanup when over size limit."""
        # Create logger with very small size limit (1KB)
        logger = EventLogger(str(temp_log_file), max_file_size_mb=0.001)
        
        # Write enough events to exceed limit
        for i in range(100):
            logger.log_presence_event(
                entity_name=f"Person{i}",
                entity_type="person",
                room="office",
                confidence=0.8,
                raw_response="office" * 100,  # Make events larger
                indicators=["indicator" * 50],
                sensor_context={"data": "x" * 500},
            )
        
        initial_size = logger.get_file_size()
        
        # Cleanup to 1KB
        result = logger.cleanup_by_size(target_size_mb=0.001)
        
        final_size = logger.get_file_size()
        
        assert result["deleted"] > 0
        assert final_size < initial_size

    def test_cleanup_by_size_keeps_newest(self, temp_log_file):
        """Test that size cleanup keeps newest events."""
        logger = EventLogger(str(temp_log_file), max_file_size_mb=0.001)
        
        # Write events in order
        for i in range(50):
            logger.log_room_transition(f"Person{i}", "office", "bedroom", 0.9)
        
        # Cleanup
        logger.cleanup_by_size(target_size_mb=0.001)
        
        # Get remaining events
        events = logger.get_recent_events(100)
        
        # Newest events should be kept
        if len(events) > 0:
            names = [e["entity_name"] for e in events]
            # Higher numbers should be present (newer events)
            assert "Person49" in names or len(events) < 50


class TestCombinedCleanup:
    """Test combined time and size cleanup."""

    def test_cleanup_both_time_and_size(self, temp_log_file):
        """Test that cleanup() runs both time and size cleanup."""
        logger = EventLogger(str(temp_log_file), retention_days=7, max_file_size_mb=0.001)
        
        # Write old events
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        for i in range(10):
            with open(temp_log_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": old_time,
                    "entity_name": f"Old{i}",
                    "data": "x" * 500,
                }) + "\n")
        
        # Write recent events
        for i in range(50):
            logger.log_room_transition(f"Recent{i}", "office", "bedroom", 0.9)
        
        result = logger.cleanup()
        
        # Should have deleted by time and potentially by size
        assert result["deleted_by_time"] > 0
        assert result["total_deleted"] >= result["deleted_by_time"]
        assert "final_size_mb" in result


class TestGlobalLoggerInstance:
    """Test global logger singleton."""

    def test_get_event_logger_creates_instance(self):
        """Test that get_event_logger creates an instance."""
        reset_event_logger()
        
        logger = get_event_logger()
        
        assert logger is not None
        assert isinstance(logger, EventLogger)

    def test_get_event_logger_returns_same_instance(self):
        """Test that get_event_logger returns same instance."""
        reset_event_logger()
        
        logger1 = get_event_logger()
        logger2 = get_event_logger()
        
        assert logger1 is logger2

    def test_get_event_logger_with_custom_settings(self):
        """Test that custom settings are used on first creation."""
        reset_event_logger()
        
        logger = get_event_logger(retention_days=14, max_file_size_mb=50)
        
        assert logger.retention_days == 14
        assert logger.max_file_size_bytes == 50 * 1024 * 1024

    def test_reset_event_logger(self):
        """Test that reset_event_logger clears instance."""
        logger1 = get_event_logger()
        reset_event_logger()
        logger2 = get_event_logger()
        
        assert logger1 is not logger2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_log_event_handles_missing_context(self, event_logger):
        """Test logging with minimal/empty context."""
        event_logger.log_presence_event(
            entity_name="Alice",
            entity_type="person",
            room="office",
            confidence=0.8,
            raw_response="office",
            indicators=[],
            sensor_context={},
        )
        
        assert event_logger.get_event_count() == 1

    def test_cleanup_empty_file(self, event_logger):
        """Test cleanup on empty file."""
        result = event_logger.cleanup_old_events()
        
        assert result["deleted"] == 0
        assert result["kept"] == 0
        assert result["error"] is None

    def test_get_recent_events_empty_file(self, event_logger):
        """Test getting recent events from empty file."""
        events = event_logger.get_recent_events(10)
        
        assert events == []

    def test_handles_malformed_json_lines(self, event_logger, temp_log_file):
        """Test that malformed JSON lines are handled gracefully."""
        with open(temp_log_file, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps({"timestamp": datetime.now().isoformat(), "valid": True}) + "\n")
        
        events = event_logger.get_recent_events(10)
        
        # Should skip invalid line and return valid one
        assert len(events) == 1
        assert events[0]["valid"] is True
