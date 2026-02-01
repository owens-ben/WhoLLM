"""History and movement pattern tracking for WhoLLM.

Tracks room transitions and provides movement history/patterns.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass
class RoomTransition:
    """Record of a room transition."""
    
    entity_name: str
    from_room: str
    to_room: str
    timestamp: str
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MovementPattern:
    """A detected movement pattern."""
    
    entity_name: str
    pattern_type: str  # "morning_routine", "work_hours", "evening_routine", etc.
    rooms: list[str]
    typical_time: str
    frequency: str  # "daily", "weekday", "weekend"
    confidence: float


class HistoryTracker:
    """Track room transition history and detect patterns."""
    
    def __init__(self, config_path: Path | None = None, max_history: int = 1000):
        """Initialize history tracker.
        
        Args:
            config_path: Path to store history data
            max_history: Maximum number of transitions to keep
        """
        self._config_path = config_path or Path("/config")
        self._history_path = self._config_path / "whollm_history.json"
        self._max_history = max_history
        
        # In-memory history (deque for efficient append/pop)
        self._transitions: deque[RoomTransition] = deque(maxlen=max_history)
        
        # Per-entity recent history
        self._entity_history: dict[str, deque[RoomTransition]] = {}
        
        # Load existing history
        self._load_history()
    
    def _load_history(self) -> None:
        """Load history from file."""
        try:
            if self._history_path.exists():
                with open(self._history_path) as f:
                    data = json.load(f)
                    for item in data.get("transitions", []):
                        transition = RoomTransition(**item)
                        self._transitions.append(transition)
                        self._add_to_entity_history(transition)
                _LOGGER.info("Loaded %d history entries", len(self._transitions))
        except Exception as err:
            _LOGGER.debug("Could not load history: %s", err)
    
    def _save_history(self) -> None:
        """Save history to file."""
        try:
            data = {
                "transitions": [t.to_dict() for t in self._transitions],
                "saved_at": datetime.now().isoformat(),
            }
            with open(self._history_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as err:
            _LOGGER.warning("Could not save history: %s", err)
    
    def _add_to_entity_history(self, transition: RoomTransition) -> None:
        """Add transition to entity-specific history."""
        entity = transition.entity_name
        if entity not in self._entity_history:
            self._entity_history[entity] = deque(maxlen=100)
        self._entity_history[entity].append(transition)
    
    def record_transition(
        self,
        entity_name: str,
        from_room: str,
        to_room: str,
        confidence: float,
    ) -> None:
        """Record a room transition.
        
        Args:
            entity_name: Name of person/pet
            from_room: Previous room
            to_room: New room
            confidence: Confidence of the transition
        """
        if from_room == to_room:
            return  # No actual transition
        
        transition = RoomTransition(
            entity_name=entity_name,
            from_room=from_room,
            to_room=to_room,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
        )
        
        self._transitions.append(transition)
        self._add_to_entity_history(transition)
        
        # Periodic save (every 10 transitions)
        if len(self._transitions) % 10 == 0:
            self._save_history()
    
    def get_recent_transitions(
        self,
        entity_name: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get recent room transitions.
        
        Args:
            entity_name: Filter by entity (optional)
            limit: Maximum number of transitions to return
            
        Returns:
            List of transition dicts
        """
        if entity_name:
            history = self._entity_history.get(entity_name, deque())
            return [t.to_dict() for t in list(history)[-limit:]]
        
        return [t.to_dict() for t in list(self._transitions)[-limit:]]
    
    def get_daily_timeline(
        self,
        entity_name: str,
        date: datetime | None = None,
    ) -> list[dict]:
        """Get room timeline for a specific day.
        
        Args:
            entity_name: Name of person/pet
            date: Date to get timeline for (default: today)
            
        Returns:
            List of {time, room, duration} entries
        """
        if date is None:
            date = datetime.now()
        
        target_date = date.strftime("%Y-%m-%d")
        
        # Filter transitions for this entity and date
        history = self._entity_history.get(entity_name, deque())
        day_transitions = [
            t for t in history 
            if t.timestamp.startswith(target_date)
        ]
        
        if not day_transitions:
            return []
        
        timeline = []
        for i, trans in enumerate(day_transitions):
            entry = {
                "time": trans.timestamp.split("T")[1][:5],  # HH:MM
                "room": trans.to_room,
            }
            
            # Calculate duration (until next transition)
            if i + 1 < len(day_transitions):
                next_time = datetime.fromisoformat(day_transitions[i + 1].timestamp)
                this_time = datetime.fromisoformat(trans.timestamp)
                duration = (next_time - this_time).total_seconds() / 60
                entry["duration_minutes"] = round(duration)
            
            timeline.append(entry)
        
        return timeline
    
    def get_room_statistics(
        self,
        entity_name: str,
        days: int = 7,
    ) -> dict[str, Any]:
        """Get room occupancy statistics.
        
        Args:
            entity_name: Name of person/pet
            days: Number of days to analyze
            
        Returns:
            Dict with room stats {room: {count, avg_duration_minutes}}
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        history = self._entity_history.get(entity_name, deque())
        recent = [t for t in history if t.timestamp >= cutoff_str]
        
        if not recent:
            return {}
        
        stats: dict[str, dict] = {}
        
        for i, trans in enumerate(recent):
            room = trans.to_room
            if room not in stats:
                stats[room] = {"count": 0, "total_minutes": 0}
            
            stats[room]["count"] += 1
            
            # Calculate duration
            if i + 1 < len(recent):
                next_time = datetime.fromisoformat(recent[i + 1].timestamp)
                this_time = datetime.fromisoformat(trans.timestamp)
                duration = (next_time - this_time).total_seconds() / 60
                stats[room]["total_minutes"] += duration
        
        # Calculate averages
        for room_stats in stats.values():
            if room_stats["count"] > 0:
                room_stats["avg_duration_minutes"] = round(
                    room_stats["total_minutes"] / room_stats["count"]
                )
            del room_stats["total_minutes"]  # Remove intermediate value
        
        return stats
    
    def detect_patterns(self, entity_name: str) -> list[dict]:
        """Detect movement patterns for an entity.
        
        Args:
            entity_name: Name of person/pet
            
        Returns:
            List of detected patterns
        """
        history = self._entity_history.get(entity_name, deque())
        if len(history) < 10:
            return []  # Not enough data
        
        patterns = []
        
        # Analyze common morning transitions (6-10 AM)
        morning_transitions = [
            t for t in history
            if "06:" <= t.timestamp.split("T")[1][:3] <= "10:"
        ]
        
        if len(morning_transitions) >= 3:
            # Find most common sequence
            morning_rooms = [t.to_room for t in morning_transitions]
            if morning_rooms:
                most_common = max(set(morning_rooms), key=morning_rooms.count)
                patterns.append({
                    "type": "morning_routine",
                    "description": f"Usually in {most_common} in the morning",
                    "confidence": morning_rooms.count(most_common) / len(morning_rooms),
                })
        
        # Analyze evening transitions (6-10 PM)
        evening_transitions = [
            t for t in history
            if "18:" <= t.timestamp.split("T")[1][:3] <= "22:"
        ]
        
        if len(evening_transitions) >= 3:
            evening_rooms = [t.to_room for t in evening_transitions]
            if evening_rooms:
                most_common = max(set(evening_rooms), key=evening_rooms.count)
                patterns.append({
                    "type": "evening_routine",
                    "description": f"Usually in {most_common} in the evening",
                    "confidence": evening_rooms.count(most_common) / len(evening_rooms),
                })
        
        return patterns
    
    def clear_history(self, entity_name: str | None = None) -> None:
        """Clear history data.
        
        Args:
            entity_name: Clear only for this entity (optional, clears all if None)
        """
        if entity_name:
            self._entity_history.pop(entity_name, None)
            # Remove from main history
            self._transitions = deque(
                (t for t in self._transitions if t.entity_name != entity_name),
                maxlen=self._max_history,
            )
        else:
            self._transitions.clear()
            self._entity_history.clear()
        
        self._save_history()


# Global instance
_history_tracker: HistoryTracker | None = None


def get_history_tracker(config_path: Path | None = None) -> HistoryTracker:
    """Get or create the global history tracker."""
    global _history_tracker
    if _history_tracker is None:
        _history_tracker = HistoryTracker(config_path)
    return _history_tracker
