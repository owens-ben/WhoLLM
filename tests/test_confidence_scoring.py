"""
Tests for improved confidence scoring algorithm (TDD).

These tests verify that confidence scoring properly weights different signals
and combines them intelligently.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta


class TestConfidenceWeights:
    """Test default confidence weights."""

    def test_light_confidence_minimum(self):
        """Light-only indicators should have at least 35% base confidence."""
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS, ENTITY_HINT_LIGHT
        
        # Light confidence should be at least 0.35
        assert DEFAULT_CONFIDENCE_WEIGHTS[ENTITY_HINT_LIGHT] >= 0.35

    def test_motion_higher_than_light(self):
        """Motion sensors should have higher confidence than lights."""
        from custom_components.whollm.const import (
            DEFAULT_CONFIDENCE_WEIGHTS,
            ENTITY_HINT_LIGHT,
            ENTITY_HINT_MOTION,
        )
        
        assert DEFAULT_CONFIDENCE_WEIGHTS[ENTITY_HINT_MOTION] > DEFAULT_CONFIDENCE_WEIGHTS[ENTITY_HINT_LIGHT]

    def test_computer_very_high_confidence(self):
        """Computer/PC indicators should have very high confidence."""
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS, ENTITY_HINT_COMPUTER
        
        assert DEFAULT_CONFIDENCE_WEIGHTS[ENTITY_HINT_COMPUTER] >= 0.85

    def test_camera_highest_confidence(self):
        """Camera AI detection should have highest confidence."""
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS, ENTITY_HINT_CAMERA
        
        assert DEFAULT_CONFIDENCE_WEIGHTS[ENTITY_HINT_CAMERA] >= 0.90


class TestMultipleIndicators:
    """Test confidence combination with multiple indicators."""

    def test_multiple_indicators_increase_confidence(self):
        """Multiple indicators in same room should increase confidence."""
        from custom_components.whollm.habits import ConfidenceCombiner
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS
        
        combiner = ConfidenceCombiner(DEFAULT_CONFIDENCE_WEIGHTS)
        
        # Single light indicator
        single_indicators = [
            {"entity_id": "light.living_room", "hint_type": "light", "room": "living_room", "state": "on"}
        ]
        
        # Multiple indicators
        multiple_indicators = [
            {"entity_id": "light.living_room", "hint_type": "light", "room": "living_room", "state": "on"},
            {"entity_id": "binary_sensor.living_room_motion", "hint_type": "motion", "room": "living_room", "state": "on"},
        ]
        
        room1, conf1, _ = combiner.combine(
            llm_room="living_room",
            llm_confidence=0.5,
            habit_room=None,
            habit_confidence=0,
            sensor_indicators=single_indicators,
            entity_name="Test",
            room_entities={},
        )
        
        room2, conf2, _ = combiner.combine(
            llm_room="living_room",
            llm_confidence=0.5,
            habit_room=None,
            habit_confidence=0,
            sensor_indicators=multiple_indicators,
            entity_name="Test",
            room_entities={},
        )
        
        # Multiple indicators should yield higher confidence
        assert conf2 > conf1

    def test_conflicting_indicators_resolved(self):
        """Conflicting indicators should be resolved by weight."""
        from custom_components.whollm.habits import ConfidenceCombiner
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS
        
        combiner = ConfidenceCombiner(DEFAULT_CONFIDENCE_WEIGHTS)
        
        # Light in living room, but motion in kitchen
        # Use unknown LLM room so it doesn't bias the result
        indicators = [
            {"entity_id": "light.living_room", "hint_type": "light", "room": "living_room", "state": "on"},
            {"entity_id": "binary_sensor.kitchen_motion", "hint_type": "motion", "room": "kitchen", "state": "on"},
        ]
        
        room, confidence, _ = combiner.combine(
            llm_room="unknown",  # Don't bias with LLM
            llm_confidence=0.0,
            habit_room=None,
            habit_confidence=0,
            sensor_indicators=indicators,
            entity_name="Test",
            room_entities={},
        )
        
        # Motion (0.60) should win over light (0.35)
        assert room == "kitchen"


class TestTimeDecay:
    """Test time-based confidence decay."""

    def test_recent_motion_high_confidence(self):
        """Recent motion (<1 min) should have high confidence."""
        from custom_components.whollm.habits import calculate_time_decay
        
        # Motion 30 seconds ago
        decay = calculate_time_decay(seconds_ago=30, sensor_type="motion")
        assert decay >= 0.9

    def test_old_motion_lower_confidence(self):
        """Old motion (>5 min) should have lower confidence."""
        from custom_components.whollm.habits import calculate_time_decay
        
        # Motion 10 minutes ago
        decay = calculate_time_decay(seconds_ago=600, sensor_type="motion")
        assert decay < 0.5

    def test_light_no_decay(self):
        """Lights don't decay - if on, they're on."""
        from custom_components.whollm.habits import calculate_time_decay
        
        # Light on for 30 minutes
        decay = calculate_time_decay(seconds_ago=1800, sensor_type="light")
        assert decay >= 0.9


class TestConfidenceCombiner:
    """Test the ConfidenceCombiner class."""

    def test_combiner_exists(self):
        """ConfidenceCombiner should exist and be importable."""
        from custom_components.whollm.habits import ConfidenceCombiner
        
        assert ConfidenceCombiner is not None

    def test_combiner_returns_tuple(self):
        """Combiner should return (room, confidence, explanation)."""
        from custom_components.whollm.habits import ConfidenceCombiner
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS
        
        combiner = ConfidenceCombiner(DEFAULT_CONFIDENCE_WEIGHTS)
        
        result = combiner.combine(
            llm_room="office",
            llm_confidence=0.7,
            habit_room=None,
            habit_confidence=0,
            sensor_indicators=[],
            entity_name="Test",
            room_entities={},
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        room, confidence, explanation = result
        assert isinstance(room, str)
        assert isinstance(confidence, float)

    def test_llm_fallback_when_no_sensors(self):
        """Without sensor data, should use LLM prediction."""
        from custom_components.whollm.habits import ConfidenceCombiner
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS
        
        combiner = ConfidenceCombiner(DEFAULT_CONFIDENCE_WEIGHTS)
        
        room, confidence, _ = combiner.combine(
            llm_room="bedroom",
            llm_confidence=0.6,
            habit_room=None,
            habit_confidence=0,
            sensor_indicators=[],
            entity_name="Test",
            room_entities={},
        )
        
        assert room == "bedroom"


class TestConfidenceBounds:
    """Test confidence value bounds."""

    def test_confidence_max_100(self):
        """Confidence should never exceed 1.0 (100%)."""
        from custom_components.whollm.habits import ConfidenceCombiner
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS
        
        combiner = ConfidenceCombiner(DEFAULT_CONFIDENCE_WEIGHTS)
        
        # Lots of strong indicators
        indicators = [
            {"entity_id": "cam.person", "hint_type": "camera", "room": "office", "state": "on"},
            {"entity_id": "switch.pc", "hint_type": "computer", "room": "office", "state": "on"},
            {"entity_id": "motion.office", "hint_type": "motion", "room": "office", "state": "on"},
            {"entity_id": "light.office", "hint_type": "light", "room": "office", "state": "on"},
        ]
        
        room, confidence, _ = combiner.combine(
            llm_room="office",
            llm_confidence=0.9,
            habit_room="office",
            habit_confidence=0.8,
            sensor_indicators=indicators,
            entity_name="Test",
            room_entities={},
        )
        
        assert confidence <= 1.0

    def test_confidence_min_0(self):
        """Confidence should never be negative."""
        from custom_components.whollm.habits import ConfidenceCombiner
        from custom_components.whollm.const import DEFAULT_CONFIDENCE_WEIGHTS
        
        combiner = ConfidenceCombiner(DEFAULT_CONFIDENCE_WEIGHTS)
        
        room, confidence, _ = combiner.combine(
            llm_room="unknown",
            llm_confidence=0.0,
            habit_room=None,
            habit_confidence=0,
            sensor_indicators=[],
            entity_name="Test",
            room_entities={},
        )
        
        assert confidence >= 0.0
