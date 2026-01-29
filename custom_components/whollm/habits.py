"""Habit hints and pattern-based presence prediction."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

# =============================================================================
# EXAMPLE HABIT PATTERNS - CUSTOMIZE FOR YOUR HOUSEHOLD
# =============================================================================
# These are example patterns using placeholder names. The integration will use
# these as defaults if a person matches the name, OR you can customize by:
#   1. Editing this file directly with your household's patterns
#   2. Creating a learned_habits.json file in your HA config directory
#   3. Using the integration's learning features over time
#
# Format: {person_name: {(start_hour, end_hour): (room, confidence, activity)}}
# =============================================================================

DEFAULT_HABITS = {
    # Example: Person who works from home in an office
    "Alice": {
        # Night patterns
        (0, 4): ("office", 0.6, "late night work/gaming"),
        (4, 7): ("bedroom", 0.9, "sleeping"),
        (22, 24): ("office", 0.7, "evening computing"),
        # Morning routine
        (7, 8): ("bedroom", 0.6, "waking up"),
        (8, 9): ("kitchen", 0.5, "breakfast"),
        # Work day (remote worker example)
        (9, 12): ("office", 0.85, "working"),
        (12, 13): ("kitchen", 0.5, "lunch"),
        (13, 17): ("office", 0.85, "working"),
        # Evening
        (17, 19): ("living_room", 0.5, "relaxing or dinner"),
        (19, 22): ("office", 0.7, "evening activities"),
    },
    # Example: Person who is often in living areas
    "Bob": {
        # Night patterns
        (0, 7): ("bedroom", 0.9, "sleeping"),
        (23, 24): ("bedroom", 0.85, "getting ready for bed"),
        # Morning
        (7, 9): ("kitchen", 0.5, "morning routine"),
        # Day patterns
        (9, 17): ("living_room", 0.5, "daytime activities"),
        # Evening
        (17, 19): ("living_room", 0.6, "dinner/TV time"),
        (19, 23): ("living_room", 0.75, "watching TV"),
    },
    # Example: Pet patterns (dog/cat)
    "Max": {
        # Pets typically follow owners and have favorite spots
        (0, 7): ("bedroom", 0.7, "sleeping with owners"),
        (7, 9): ("kitchen", 0.5, "feeding time"),
        # During day - napping or following household members
        (9, 17): ("living_room", 0.6, "napping or with family"),
        (17, 19): ("kitchen", 0.4, "dinner time"),
        # Evening - with family
        (19, 22): ("living_room", 0.5, "with family"),
        (22, 24): ("bedroom", 0.5, "settling down for night"),
    },
}

# Day-specific overrides (weekends often have different patterns)
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
        self._learned_patterns_path = Path("/config/whollm_learned_habits.json")
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
    
    # Weight factors for different detection methods - IMPROVED WEIGHTS
    WEIGHTS = {
        "camera_ai": 0.95,      # Camera AI detection is very reliable
        "face_recognition": 0.90,  # Face recognition is reliable
        "vision_llm": 0.70,     # LLM vision is decent
        "llm_reasoning": 0.50,  # LLM text reasoning (reduced - was over-trusted)
        "habit": 0.35,          # Habit-based prediction (reduced)
        "motion": 0.60,         # Motion sensor (increased)
        "media_playing": 0.90,  # TV/media PLAYING is very strong (increased!)
        "media_paused": 0.70,   # Media paused - still strong
        "light": 0.25,          # Light on - weak indicator alone
        "computer_active": 0.85,  # Computer actively in use - very strong for office
        "behavioral": 0.80,     # Cross-person behavioral logic
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
        media_context: dict | None = None,
        entity_name: str | None = None,
        pc_is_on: bool = False,
    ) -> tuple[str, float, str]:
        """Combine multiple signals into final prediction.
        
        Args:
            llm_room: Room predicted by LLM
            llm_confidence: LLM's confidence
            habit_room: Room from habit patterns
            habit_confidence: Habit pattern confidence
            sensor_indicators: List of indicator strings
            camera_ai_room: Room from camera AI detection
            media_context: Dict of media player states {room: state}
            entity_name: Name of person (for behavioral logic)
            pc_is_on: Whether a PC is actively on
            
        Returns:
            Tuple of (room, confidence, explanation)
        """
        candidates = {}
        explanations = []
        
        # Camera AI is highest priority
        if camera_ai_room and camera_ai_room != "unknown":
            candidates[camera_ai_room] = candidates.get(camera_ai_room, 0) + cls.WEIGHTS["camera_ai"]
            explanations.append(f"Camera AI: {camera_ai_room}")
        
        # MEDIA PLAYING - Very strong signal! (NEW IMPROVED)
        if media_context:
            for room, state in media_context.items():
                if state == "playing":
                    candidates[room] = candidates.get(room, 0) + cls.WEIGHTS["media_playing"]
                    explanations.append(f"Media PLAYING in {room}")
                elif state == "paused":
                    candidates[room] = candidates.get(room, 0) + cls.WEIGHTS["media_paused"]
                    explanations.append(f"Media paused in {room}")
        
        # BEHAVIORAL LOGIC - Cross-person inference based on PC activity
        # This assumes the PC owner (first person configured) is in office when PC is on
        # Customize this logic for your household
        if entity_name and pc_is_on:
            # PC is active - strong indicator someone is in office
            # The logic below is an example - customize for your setup
            candidates["office"] = candidates.get("office", 0) + cls.WEIGHTS["computer_active"]
            explanations.append("PC active -> office")
            
            # If media is playing elsewhere, other household members are likely there
            if media_context and media_context.get("living_room") == "playing":
                # Boost living room for other people when PC user is in office
                candidates["living_room"] = candidates.get("living_room", 0) + cls.WEIGHTS["behavioral"] * 0.5
                explanations.append("Media playing in living_room")
        
        # LLM reasoning (slightly reduced weight)
        if llm_room and llm_room != "unknown":
            weighted = llm_confidence * cls.WEIGHTS["llm_reasoning"]
            candidates[llm_room] = candidates.get(llm_room, 0) + weighted
            explanations.append(f"LLM: {llm_room}")
        
        # Habit prediction (reduced weight)
        if habit_room and habit_room != "unknown" and habit_confidence > 0:
            weighted = habit_confidence * cls.WEIGHTS["habit"]
            candidates[habit_room] = candidates.get(habit_room, 0) + weighted
        
        # Sensor indicators boost
        if sensor_indicators:
            for indicator in sensor_indicators:
                indicator_lower = indicator.lower()
                # Extract room from indicator if possible
                for room in ["office", "bedroom", "living_room", "kitchen", "bathroom", "entry"]:
                    room_match = room.replace("_", " ") in indicator_lower or room in indicator_lower
                    if room_match:
                        if "motion" in indicator_lower:
                            candidates[room] = candidates.get(room, 0) + cls.WEIGHTS["motion"]
                        elif "light" in indicator_lower:
                            candidates[room] = candidates.get(room, 0) + cls.WEIGHTS["light"]
        
        if not candidates:
            return "unknown", 0.0, "No signals available"
        
        # Find the room with highest combined score
        best_room = max(candidates, key=candidates.get)
        best_score = candidates[best_room]
        
        # Calculate confidence based on signal strength
        # Strong signals should give HIGH confidence
        max_possible = cls.WEIGHTS["camera_ai"] + cls.WEIGHTS["media_playing"] + cls.WEIGHTS["computer_active"]
        final_confidence = min(1.0, best_score / max_possible)
        
        # Minimum confidence floors based on signal types
        if camera_ai_room and camera_ai_room == best_room:
            final_confidence = max(0.85, final_confidence)
        elif media_context and media_context.get(best_room) == "playing":
            final_confidence = max(0.75, final_confidence)
        elif pc_is_on and best_room == "office":
            final_confidence = max(0.80, final_confidence)
        
        # Boost confidence if multiple strong sources agree
        strong_signals = sum(1 for room, score in candidates.items() if room == best_room and score >= 0.5)
        if strong_signals >= 2:
            final_confidence = min(1.0, final_confidence + 0.1)
            explanations.append("Multiple strong signals agree")
        
        explanation = " | ".join(explanations[:4])  # Limit explanation length
        
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



