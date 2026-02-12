"""
Tests for history and movement patterns (TDD).

These tests verify history tracking and pattern detection.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile


class TestHistoryTracker:
    """Test the HistoryTracker class."""

    def test_tracker_initialization(self):
        """Tracker should initialize without errors."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            assert tracker is not None

    def test_record_transition(self):
        """Should record room transitions."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            tracker.record_transition(
                entity_name="Ben",
                from_room="bedroom",
                to_room="kitchen",
                confidence=0.8,
            )
            
            history = tracker.get_recent_transitions("Ben")
            assert len(history) == 1
            assert history[0]["from_room"] == "bedroom"
            assert history[0]["to_room"] == "kitchen"

    def test_no_duplicate_transitions(self):
        """Same room transitions should not be recorded."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            tracker.record_transition(
                entity_name="Ben",
                from_room="office",
                to_room="office",  # Same room
                confidence=0.8,
            )
            
            history = tracker.get_recent_transitions("Ben")
            assert len(history) == 0

    def test_get_recent_transitions_limit(self):
        """Should respect limit parameter."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            # Record 10 transitions
            rooms = ["bedroom", "bathroom", "kitchen", "office", "living_room"]
            for i in range(10):
                tracker.record_transition(
                    entity_name="Ben",
                    from_room=rooms[i % len(rooms)],
                    to_room=rooms[(i + 1) % len(rooms)],
                    confidence=0.7,
                )
            
            history = tracker.get_recent_transitions("Ben", limit=5)
            assert len(history) == 5

    def test_entity_filter(self):
        """Should filter by entity name."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            tracker.record_transition("Ben", "bedroom", "kitchen", 0.8)
            tracker.record_transition("Alex", "bedroom", "office", 0.8)
            tracker.record_transition("Ben", "kitchen", "office", 0.8)
            
            ben_history = tracker.get_recent_transitions("Ben")
            alex_history = tracker.get_recent_transitions("Alex")
            
            assert len(ben_history) == 2
            assert len(alex_history) == 1


class TestRoomStatistics:
    """Test room statistics calculations."""

    def test_get_room_statistics(self):
        """Should calculate room statistics."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            # Record several transitions
            tracker.record_transition("Ben", "bedroom", "kitchen", 0.8)
            tracker.record_transition("Ben", "kitchen", "office", 0.8)
            tracker.record_transition("Ben", "office", "living_room", 0.8)
            tracker.record_transition("Ben", "living_room", "bedroom", 0.8)
            
            stats = tracker.get_room_statistics("Ben", days=1)
            
            # Should have entries for all rooms visited
            assert "kitchen" in stats
            assert "office" in stats
            assert "living_room" in stats


class TestPatternDetection:
    """Test movement pattern detection."""

    def test_detect_patterns_requires_data(self):
        """Pattern detection requires minimum history."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            # Only a few transitions
            tracker.record_transition("Ben", "bedroom", "kitchen", 0.8)
            
            patterns = tracker.detect_patterns("Ben")
            assert len(patterns) == 0  # Not enough data

    def test_clear_history(self):
        """Should clear history."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            tracker.record_transition("Ben", "bedroom", "kitchen", 0.8)
            tracker.record_transition("Alex", "bedroom", "office", 0.8)
            
            # Clear Ben's history
            tracker.clear_history("Ben")
            
            assert len(tracker.get_recent_transitions("Ben")) == 0
            assert len(tracker.get_recent_transitions("Alex")) == 1


class TestHistoryPersistence:
    """Test history persistence to disk."""

    def test_save_and_load(self):
        """History should persist across restarts."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tracker and add data
            tracker1 = HistoryTracker(config_path=Path(tmpdir))
            tracker1.record_transition("Ben", "bedroom", "kitchen", 0.8)
            tracker1._save_history()
            
            # Create new tracker (simulating restart)
            tracker2 = HistoryTracker(config_path=Path(tmpdir))
            
            history = tracker2.get_recent_transitions("Ben")
            assert len(history) == 1
            assert history[0]["to_room"] == "kitchen"


class TestDailyTimeline:
    """Test daily timeline feature."""

    def test_get_daily_timeline_empty(self):
        """Should return empty list for no data."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            timeline = tracker.get_daily_timeline("Ben")
            assert timeline == []

    def test_get_daily_timeline_with_data(self):
        """Should return timeline entries."""
        from custom_components.whollm.history import HistoryTracker
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoryTracker(config_path=Path(tmpdir))
            
            # Add some transitions for today
            tracker.record_transition("Ben", "bedroom", "kitchen", 0.8)
            tracker.record_transition("Ben", "kitchen", "office", 0.8)
            
            timeline = tracker.get_daily_timeline("Ben")
            assert len(timeline) == 2


