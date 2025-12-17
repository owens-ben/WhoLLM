"""Habit hints and pattern-based presence prediction."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Hardcoded habit patterns (will be replaced by learned patterns)
# Format: {person: {(start_hour, end_hour): (room, confidence)}}
DEFAULT_HABITS = {
    "Alice": {
        # Night (sleeping)
        (0, 6): ("bedroom", 0.9, "sleeping"),
        (22, 24): ("bedroom", 0.8, "going to bed or sleeping"),
        # Morning routine
        (6, 7): ("bedroom", 0.6, "waking up"),
        (7, 8): ("bathroom", 0.5, "morning routine"),
        (8, 9): ("kitchen", 0.6, "breakfast"),
        # Work day
        (9, 12): ("office", 0.8, "working"),
        (12, 13): ("kitchen", 0.5, "lunch"),
        (13, 17): ("office", 0.8, "working"),
        # Evening
        (17, 19): ("kitchen", 0.5, "dinner prep"),
        (19, 22): ("living_room", 0.6, "relaxing"),
    },
    "Bob": {
        # Night (sleeping)
        (0, 7): ("bedroom", 0.9, "sleeping"),
        (23, 24): ("bedroom", 0.8, "going to bed"),
        # Morning
        (7, 9): ("kitchen", 0.5, "morning routine"),
        # Day
        (9, 17): ("living_room", 0.4, "home during day"),
        # Evening
        (17, 19): ("kitchen", 0.5, "dinner"),
        (19, 23): ("living_room", 0.6, "watching TV"),
    },
    "Max": {  # Pet
        # Pets follow owners and have favorite spots
        (0, 7): ("bedroom", 0.7, "sleeping with owners"),
        (7, 9): ("kitchen", 0.5, "breakfast time"),
        (9, 17): ("living_room", 0.5, "napping on couch"),
        (17, 19): ("kitchen", 0.4, "dinner time"),
        (19, 22): ("living_room", 0.6, "with family"),
        (22, 24): ("bedroom", 0.6, "going to bed with owners"),
    },
}

# Day-specific overrides (e.g., weekends are different)
WEEKEND_OVERRIDES = {
    "Alice": {
        (9, 12): ("living_room", 0.5, "weekend morning"),
        (12, 17): ("living_room", 0.4, "weekend afternoon"),
    },
    "Bob": {
        (9, 12): ("living_room", 0.5, "weekend morning"),
    },
}


class HabitPredictor:
    """Predict presence based on typical habits and patterns."""
    
    def __init__(self, habits: dict | None = None):
        """Initialize with habit patterns."""
        self.habits = habits or DEFAULT_HABITS
        self.weekend_overrides = WEEKEND_OVERRIDES
        self._learned_patterns_path = Path("/config/llm_presence_learned_habits.json")
        self._load_learned_patterns()
    
    def _load_learned_patterns(self) -> None:
        """Load learned patterns from file if available."""
        try:
            if self._learned_patterns_path.exists():
                with open(self._learned_patterns_path) as f:
                    learned = json.load(f)
                # Merge learned patterns with defaults (learned takes priority)
                for person, patterns in learned.items():
                    if person not in self.habits:
                        self.habits[person] = {}
                    self.habits[person].update(patterns)
                _LOGGER.info("Loaded learned habit patterns")
        except Exception as err:
            _LOGGER.warning("Could not load learned patterns: %s", err)
    
    def get_habit_hint(
        self,
        entity_name: str,
        entity_type: str = "person",
    ) -> dict[str, Any]:
        """Get habit-based prediction for current time.
        
        Returns:
            Dict with predicted_room, confidence, reason, and hint_text
        """
        now = datetime.now()
        hour = now.hour
        is_weekend = now.weekday() >= 5
        
        # Get habits for this entity
        person_habits = self.habits.get(entity_name, {})
        
        # Check weekend overrides first
        if is_weekend and entity_name in self.weekend_overrides:
            overrides = self.weekend_overrides[entity_name]
            for (start, end), (room, conf, reason) in overrides.items():
                if start <= hour < end:
                    return {
                        "predicted_room": room,
                        "confidence": conf,
                        "reason": reason,
                        "hint_text": f"Based on weekend patterns, {entity_name} is typically in {room} ({reason}) at this time.",
                        "source": "weekend_habit",
                    }
        
        # Check regular habits
        for time_range, pattern in person_habits.items():
            if isinstance(time_range, tuple) and len(time_range) == 2:
                start, end = time_range
                if start <= hour < end:
                    if isinstance(pattern, tuple) and len(pattern) >= 2:
                        room = pattern[0]
                        conf = pattern[1]
                        reason = pattern[2] if len(pattern) > 2 else ""
                        return {
                            "predicted_room": room,
                            "confidence": conf,
                            "reason": reason,
                            "hint_text": f"Based on typical patterns, {entity_name} is usually in {room} ({reason}) at this time.",
                            "source": "habit",
                        }
        
        # No habit found for this time
        return {
            "predicted_room": "unknown",
            "confidence": 0.0,
            "reason": "no pattern",
            "hint_text": f"No typical pattern found for {entity_name} at this time.",
            "source": "none",
        }
    
    def get_habit_context_for_prompt(self, entity_name: str, entity_type: str = "person") -> str:
        """Get habit hint formatted for LLM prompt."""
        hint = self.get_habit_hint(entity_name, entity_type)
        
        if hint["confidence"] > 0:
            return f"""
HABIT PATTERN HINT (use as context, not absolute truth):
{hint['hint_text']}
Confidence: {int(hint['confidence'] * 100)}%
Note: This is a typical pattern - actual sensor data should override this if contradictory!
"""
        return ""
    
    def get_daily_schedule(self, entity_name: str) -> list[dict]:
        """Get the full daily schedule for a person."""
        schedule = []
        person_habits = self.habits.get(entity_name, {})
        
        for time_range, pattern in sorted(person_habits.items()):
            if isinstance(time_range, tuple) and len(time_range) == 2:
                start, end = time_range
                if isinstance(pattern, tuple) and len(pattern) >= 2:
                    schedule.append({
                        "start_hour": start,
                        "end_hour": end,
                        "room": pattern[0],
                        "confidence": pattern[1],
                        "activity": pattern[2] if len(pattern) > 2 else "",
                    })
        
        return schedule


class ConfidenceCombiner:
    """Combine confidence scores from multiple sources."""
    
    # Weight factors for different detection methods
    WEIGHTS = {
        "camera_ai": 0.95,      # Camera AI detection is very reliable
        "face_recognition": 0.90,  # Face recognition is reliable
        "vision_llm": 0.70,     # LLM vision is decent
        "llm_reasoning": 0.60,  # LLM text reasoning
        "habit": 0.40,          # Habit-based prediction
        "motion": 0.50,         # Motion sensor
        "media": 0.80,          # Media playing (strong indicator)
        "light": 0.30,          # Light on (weak indicator)
        "computer": 0.75,       # Computer on (good indicator)
    }
    
    @classmethod
    def combine(
        cls,
        llm_room: str,
        llm_confidence: float,
        habit_room: str | None = None,
        habit_confidence: float = 0.0,
        sensor_indicators: list[str] | None = None,
        camera_ai_room: str | None = None,
    ) -> tuple[str, float, str]:
        """Combine multiple signals into final prediction.
        
        Returns:
            Tuple of (room, confidence, explanation)
        """
        candidates = {}
        explanations = []
        
        # Camera AI is highest priority
        if camera_ai_room and camera_ai_room != "unknown":
            candidates[camera_ai_room] = candidates.get(camera_ai_room, 0) + cls.WEIGHTS["camera_ai"]
            explanations.append(f"Camera AI detected in {camera_ai_room}")
        
        # LLM reasoning
        if llm_room and llm_room != "unknown":
            weighted = llm_confidence * cls.WEIGHTS["llm_reasoning"]
            candidates[llm_room] = candidates.get(llm_room, 0) + weighted
            explanations.append(f"LLM reasoning: {llm_room} ({int(llm_confidence*100)}%)")
        
        # Habit prediction
        if habit_room and habit_room != "unknown" and habit_confidence > 0:
            weighted = habit_confidence * cls.WEIGHTS["habit"]
            candidates[habit_room] = candidates.get(habit_room, 0) + weighted
            explanations.append(f"Habit pattern: {habit_room} ({int(habit_confidence*100)}%)")
        
        # Sensor indicators boost
        if sensor_indicators:
            for indicator in sensor_indicators:
                indicator_lower = indicator.lower()
                # Extract room from indicator if possible
                for room in ["office", "bedroom", "living_room", "kitchen", "bathroom", "entry"]:
                    if room.replace("_", " ") in indicator_lower or room in indicator_lower:
                        if "motion" in indicator_lower:
                            candidates[room] = candidates.get(room, 0) + cls.WEIGHTS["motion"]
                        elif "light" in indicator_lower:
                            candidates[room] = candidates.get(room, 0) + cls.WEIGHTS["light"]
                        elif "tv" in indicator_lower or "media" in indicator_lower:
                            candidates[room] = candidates.get(room, 0) + cls.WEIGHTS["media"]
                        elif "pc" in indicator_lower or "computer" in indicator_lower:
                            candidates[room] = candidates.get(room, 0) + cls.WEIGHTS["computer"]
        
        if not candidates:
            return "unknown", 0.0, "No signals available"
        
        # Find the room with highest combined score
        best_room = max(candidates, key=candidates.get)
        best_score = candidates[best_room]
        
        # Normalize confidence to 0-1 range
        max_possible = sum(cls.WEIGHTS.values())
        final_confidence = min(1.0, best_score / max_possible * 2)  # Scale up a bit
        
        # Boost confidence if multiple sources agree
        agreeing_sources = sum(1 for room, score in candidates.items() if room == best_room and score > 0.1)
        if agreeing_sources >= 2:
            final_confidence = min(1.0, final_confidence + 0.1)
            explanations.append("Multiple sources agree")
        
        explanation = " | ".join(explanations)
        
        return best_room, final_confidence, explanation


# Global instances
_habit_predictor: HabitPredictor | None = None
_confidence_combiner = ConfidenceCombiner()


def get_habit_predictor() -> HabitPredictor:
    """Get or create the global habit predictor."""
    global _habit_predictor
    if _habit_predictor is None:
        _habit_predictor = HabitPredictor()
    return _habit_predictor


def get_confidence_combiner() -> ConfidenceCombiner:
    """Get the confidence combiner."""
    return _confidence_combiner



