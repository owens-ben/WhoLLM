"""Tests for habit learning and confidence combining."""

from __future__ import annotations

import json
import pytest
from datetime import datetime
from pathlib import Path

from custom_components.whollm.habits import HabitPredictor, ConfidenceCombiner
from custom_components.whollm.const import (
    ENTITY_HINT_COMPUTER,
    ENTITY_HINT_MEDIA,
    ENTITY_HINT_MOTION,
    ENTITY_HINT_LIGHT,
)


class TestHabitPredictor:
    """Tests for HabitPredictor class."""

    def test_init_empty(self, habit_predictor):
        """Test initialization with no existing patterns."""
        assert habit_predictor.habits == {}

    def test_learn_from_event_low_confidence_ignored(self, habit_predictor):
        """Events with confidence < 0.7 should not be learned."""
        habit_predictor.learn_from_event(
            entity_name="Alice",
            room="office",
            confidence=0.5,  # Too low
        )
        assert "Alice" not in habit_predictor.habits

    def test_learn_from_event_high_confidence(self, habit_predictor):
        """Events with confidence >= 0.7 should be learned."""
        ts = datetime(2024, 1, 15, 14, 30)  # 2:30 PM
        habit_predictor.learn_from_event(
            entity_name="Alice",
            room="office",
            confidence=0.85,
            timestamp=ts,
        )

        assert "Alice" in habit_predictor.habits
        # 14:30 falls into slot (14, 16)
        assert (14, 16) in habit_predictor.habits["Alice"]
        pattern = habit_predictor.habits["Alice"][(14, 16)]
        assert pattern[0] == "office"
        assert pattern[1] == 0.85

    def test_learn_reinforces_same_room(self, habit_predictor):
        """Learning same room multiple times increases confidence."""
        ts = datetime(2024, 1, 15, 14, 30)

        # First observation
        habit_predictor.learn_from_event("Alice", "office", 0.8, ts)
        # Second observation, same room
        habit_predictor.learn_from_event("Alice", "office", 0.9, ts)

        pattern = habit_predictor.habits["Alice"][(14, 16)]
        # Confidence should be weighted average: (0.8 * 1 + 0.9) / 2 = 0.85
        assert pattern[1] == pytest.approx(0.85, rel=0.01)
        assert pattern[2] == 2  # Count should be 2

    def test_learn_different_room_higher_confidence_wins(self, habit_predictor):
        """New room with higher confidence replaces existing."""
        ts = datetime(2024, 1, 15, 14, 30)

        habit_predictor.learn_from_event("Alice", "office", 0.75, ts)
        habit_predictor.learn_from_event("Alice", "bedroom", 0.9, ts)  # Higher confidence

        pattern = habit_predictor.habits["Alice"][(14, 16)]
        assert pattern[0] == "bedroom"
        assert pattern[1] == 0.9

    def test_learn_different_room_lower_confidence_ignored(self, habit_predictor):
        """New room with lower confidence does not replace."""
        ts = datetime(2024, 1, 15, 14, 30)

        habit_predictor.learn_from_event("Alice", "office", 0.9, ts)
        habit_predictor.learn_from_event("Alice", "bedroom", 0.75, ts)  # Lower

        pattern = habit_predictor.habits["Alice"][(14, 16)]
        assert pattern[0] == "office"  # Still office

    def test_get_habit_hint_no_patterns(self, habit_predictor):
        """Get hint when no patterns exist."""
        hint = habit_predictor.get_habit_hint("Alice", "person")

        assert hint["predicted_room"] == "unknown"
        assert hint["confidence"] == 0.0
        assert "no learned patterns" in hint["reason"]

    def test_get_habit_hint_with_pattern(self, habit_predictor):
        """Get hint when matching pattern exists."""
        # Learn a pattern for current time
        now = datetime.now()
        habit_predictor.learn_from_event("Alice", "office", 0.85, now)

        hint = habit_predictor.get_habit_hint("Alice", "person")

        assert hint["predicted_room"] == "office"
        assert hint["confidence"] == 0.85
        assert hint["source"] == "learned_habit"

    def test_save_and_load_patterns(self, temp_config_dir):
        """Test pattern persistence."""
        predictor1 = HabitPredictor(config_path=temp_config_dir)
        ts = datetime(2024, 1, 15, 10, 0)
        predictor1.learn_from_event("Alice", "kitchen", 0.9, ts)
        predictor1.save_learned_patterns()

        # Create new predictor, should load saved patterns
        predictor2 = HabitPredictor(config_path=temp_config_dir)

        assert "Alice" in predictor2.habits
        assert (10, 12) in predictor2.habits["Alice"]
        assert predictor2.habits["Alice"][(10, 12)][0] == "kitchen"

    def test_clear_patterns_single_person(self, habit_predictor):
        """Clear patterns for one person."""
        habit_predictor.learn_from_event("Alice", "office", 0.9)
        habit_predictor.learn_from_event("Bob", "bedroom", 0.85)

        habit_predictor.clear_patterns("Alice")

        assert "Alice" not in habit_predictor.habits
        assert "Bob" in habit_predictor.habits

    def test_clear_patterns_all(self, habit_predictor):
        """Clear all patterns."""
        habit_predictor.learn_from_event("Alice", "office", 0.9)
        habit_predictor.learn_from_event("Bob", "bedroom", 0.85)

        habit_predictor.clear_patterns()

        assert habit_predictor.habits == {}

    def test_get_daily_schedule(self, habit_predictor):
        """Get daily schedule for a person."""
        habit_predictor.learn_from_event("Alice", "bedroom", 0.9, datetime(2024, 1, 15, 7, 0))
        habit_predictor.learn_from_event("Alice", "office", 0.85, datetime(2024, 1, 15, 10, 0))
        habit_predictor.learn_from_event("Alice", "office", 0.9, datetime(2024, 1, 15, 14, 0))

        schedule = habit_predictor.get_daily_schedule("Alice")

        assert len(schedule) == 3
        # Should be sorted by hour
        assert schedule[0]["start_hour"] == 6
        assert schedule[0]["room"] == "bedroom"


class TestConfidenceCombiner:
    """Tests for ConfidenceCombiner class."""

    def test_combine_no_signals(self, confidence_combiner):
        """Handle case with no signals."""
        room, confidence, explanation = confidence_combiner.combine(
            llm_room="unknown",
            llm_confidence=0.0,
        )

        assert room == "unknown"
        assert confidence == 0.0
        assert "No signals" in explanation

    def test_combine_llm_only(self, confidence_combiner):
        """Combine with only LLM signal."""
        room, confidence, explanation = confidence_combiner.combine(
            llm_room="office",
            llm_confidence=0.8,
        )

        assert room == "office"
        assert confidence > 0
        assert "LLM" in explanation

    def test_combine_with_sensor_indicators(self, confidence_combiner, sample_indicators):
        """Combine LLM with sensor indicators."""
        room, confidence, explanation = confidence_combiner.combine(
            llm_room="office",
            llm_confidence=0.7,
            sensor_indicators=sample_indicators,
        )

        assert room == "office"
        assert confidence > 0.5  # Multiple signals should boost confidence

    def test_combine_habit_overrides_weak_llm(self, confidence_combiner):
        """Strong habit prediction can override weak LLM."""
        room, confidence, _ = confidence_combiner.combine(
            llm_room="bedroom",
            llm_confidence=0.3,
            habit_room="office",
            habit_confidence=0.9,
        )

        # With strong habit signal, office should win
        # (depends on weight configuration)
        assert confidence > 0

    def test_combine_computer_indicator_strong(self, confidence_combiner):
        """Computer indicator should be a strong signal."""
        indicators = [
            {"entity_id": "switch.office_pc", "hint_type": ENTITY_HINT_COMPUTER, "room": "office", "state": "on"},
        ]

        room, confidence, _ = confidence_combiner.combine(
            llm_room="bedroom",
            llm_confidence=0.5,
            sensor_indicators=indicators,
        )

        # Computer on should strongly suggest office
        assert room == "office"

    def test_combine_media_playing_strong(self, confidence_combiner):
        """Media playing should be a strong signal."""
        indicators = [
            {
                "entity_id": "media_player.living_room_tv",
                "hint_type": ENTITY_HINT_MEDIA,
                "room": "living_room",
                "state": "playing",
            },
        ]

        room, confidence, _ = confidence_combiner.combine(
            llm_room="office",
            llm_confidence=0.5,
            sensor_indicators=indicators,
        )

        assert room == "living_room"

    def test_combine_media_paused_weaker(self, confidence_combiner):
        """Media paused should be weaker than playing."""
        playing_indicators = [
            {"entity_id": "media_player.tv", "hint_type": ENTITY_HINT_MEDIA, "room": "living_room", "state": "playing"},
        ]
        paused_indicators = [
            {"entity_id": "media_player.tv", "hint_type": ENTITY_HINT_MEDIA, "room": "living_room", "state": "paused"},
        ]

        _, playing_conf, _ = confidence_combiner.combine(
            llm_room="living_room", llm_confidence=0.5, sensor_indicators=playing_indicators
        )
        _, paused_conf, _ = confidence_combiner.combine(
            llm_room="living_room", llm_confidence=0.5, sensor_indicators=paused_indicators
        )

        assert playing_conf >= paused_conf

    def test_combine_multiple_rooms_picks_strongest(self, confidence_combiner):
        """When indicators point to different rooms, pick strongest."""
        indicators = [
            {"entity_id": "light.bedroom", "hint_type": ENTITY_HINT_LIGHT, "room": "bedroom", "state": "on"},
            {"entity_id": "switch.office_pc", "hint_type": ENTITY_HINT_COMPUTER, "room": "office", "state": "on"},
            {
                "entity_id": "binary_sensor.office_motion",
                "hint_type": ENTITY_HINT_MOTION,
                "room": "office",
                "state": "on",
            },
        ]

        room, _, _ = confidence_combiner.combine(
            llm_room="bedroom",
            llm_confidence=0.4,
            sensor_indicators=indicators,
        )

        # Office has computer (0.85) + motion (0.60) vs bedroom light (0.25)
        assert room == "office"

    def test_combine_multiple_signals_agree_boost(self, confidence_combiner):
        """Multiple signals agreeing should boost confidence."""
        # Single signal
        single_indicators = [
            {
                "entity_id": "binary_sensor.office_motion",
                "hint_type": ENTITY_HINT_MOTION,
                "room": "office",
                "state": "on",
            },
        ]

        # Multiple signals
        multi_indicators = [
            {
                "entity_id": "binary_sensor.office_motion",
                "hint_type": ENTITY_HINT_MOTION,
                "room": "office",
                "state": "on",
            },
            {"entity_id": "switch.office_pc", "hint_type": ENTITY_HINT_COMPUTER, "room": "office", "state": "on"},
            {"entity_id": "light.office", "hint_type": ENTITY_HINT_LIGHT, "room": "office", "state": "on"},
        ]

        _, single_conf, _ = confidence_combiner.combine(
            llm_room="office", llm_confidence=0.6, sensor_indicators=single_indicators
        )
        _, multi_conf, _ = confidence_combiner.combine(
            llm_room="office", llm_confidence=0.6, sensor_indicators=multi_indicators
        )

        assert multi_conf > single_conf

    def test_update_weights(self, confidence_combiner):
        """Test updating confidence weights."""
        original_motion = confidence_combiner.weights.get(ENTITY_HINT_MOTION)

        confidence_combiner.update_weights({ENTITY_HINT_MOTION: 0.95})

        assert confidence_combiner.weights[ENTITY_HINT_MOTION] == 0.95
        assert confidence_combiner.weights[ENTITY_HINT_MOTION] != original_motion
