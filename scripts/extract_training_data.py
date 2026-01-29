#!/usr/bin/env python3
"""
Extract training data from Home Assistant logbook entries.

This script processes HA logbook data and converts it into training data
for the ML presence prediction model.
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Any

# Room definitions
ROOMS = ['living_room', 'bedroom', 'office', 'kitchen', 'bathroom', 'entry']

# Room keyword mapping
ROOM_KEYWORDS = {
    'living_room': ['living_room', 'living room', 'livingroom', 'lounge', 'living'],
    'bedroom': ['bedroom', 'bed_room', 'master_bedroom'],
    'office': ['office', 'study', 'desk'],
    'kitchen': ['kitchen', 'cook'],
    'bathroom': ['bathroom', 'bath', 'shower', 'toilet'],
    'entry': ['entry', 'front_door', 'hallway', 'foyer', 'front'],
}

# People to track - customize for your household
# These are detected from training data or can be set here
PEOPLE = []  # Will be auto-detected from data, or set manually like: ['alice', 'bob', 'max']


def extract_room_from_entity(entity_id: str) -> str | None:
    """Extract room name from entity_id."""
    entity_lower = entity_id.lower()
    for room, keywords in ROOM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in entity_lower:
                return room
    return None


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    ts_str = ts_str.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return datetime.fromisoformat(ts_str.split('+')[0])


def extract_person_presence(entity_id: str, state: str) -> tuple[str, str, bool] | None:
    """Extract person presence from entity.
    
    Returns: (person_name, room, is_present) or None
    """
    entity_lower = entity_id.lower()
    
    # Check for llm_presence binary sensors like binary_sensor.alice_in_office
    if entity_lower.startswith('binary_sensor.'):
        name = entity_lower.replace('binary_sensor.', '')
        
        for person in PEOPLE:
            if name.startswith(f'{person}_in_') or name.startswith(f'{person}_'):
                # Extract room
                for room in ROOMS:
                    if room in name:
                        is_present = state.lower() == 'on'
                        return (person, room, is_present)
    
    return None


def process_logbook_entries(entries: list[dict]) -> list[dict]:
    """Process logbook entries and create training samples."""
    # Sort entries by time
    entries = sorted(entries, key=lambda x: x.get('when', ''))
    
    # Track current state of all entities
    entity_states: dict[str, dict] = {}
    
    # Track person locations over time
    person_locations: dict[str, dict[str, Any]] = {p: {'room': 'unknown', 'confidence': 0} for p in PEOPLE}
    
    training_samples = []
    
    # Group into time windows
    window_minutes = 5  # 5-minute windows for better aggregation
    current_window_start = None
    window_entries = []
    
    for entry in entries:
        when_str = entry.get('when', entry.get('state', ''))
        if not when_str or not isinstance(when_str, str):
            continue
            
        try:
            when = parse_timestamp(when_str)
        except (ValueError, TypeError):
            continue
        
        entity_id = entry.get('entity_id', '')
        state = entry.get('state', '')
        name = entry.get('name', '')
        
        # Update entity state
        entity_states[entity_id] = {
            'state': state,
            'name': name,
            'last_changed': when.isoformat(),
        }
        
        # Check for person presence updates
        presence_info = extract_person_presence(entity_id, state)
        if presence_info:
            person, room, is_present = presence_info
            if is_present:
                person_locations[person] = {
                    'room': room,
                    'timestamp': when.isoformat(),
                }
        
        # Window management
        if current_window_start is None:
            current_window_start = when
            
        if when - current_window_start > timedelta(minutes=window_minutes):
            # Create sample from this window
            if window_entries:
                sample = create_training_sample(
                    window_start=current_window_start,
                    entity_states=entity_states.copy(),
                    window_entries=window_entries,
                    person_locations=person_locations.copy(),
                )
                if sample and sample.get('has_presence_data'):
                    training_samples.append(sample)
            
            current_window_start = when
            window_entries = []
        
        window_entries.append(entry)
    
    # Process final window
    if window_entries and current_window_start:
        sample = create_training_sample(
            window_start=current_window_start,
            entity_states=entity_states.copy(),
            window_entries=window_entries,
            person_locations=person_locations.copy(),
        )
        if sample and sample.get('has_presence_data'):
            training_samples.append(sample)
    
    return training_samples


def create_training_sample(
    window_start: datetime,
    entity_states: dict[str, dict],
    window_entries: list[dict],
    person_locations: dict[str, dict],
) -> dict | None:
    """Create a training sample from a time window."""
    
    # Time features
    features = {
        'timestamp': window_start.isoformat(),
        'hour': window_start.hour,
        'minute': window_start.minute,
        'day_of_week': window_start.weekday(),
        'day_name': window_start.strftime('%A'),
        'is_weekend': window_start.weekday() >= 5,
        'is_night': window_start.hour >= 22 or window_start.hour < 6,
        'is_morning': 6 <= window_start.hour < 10,
        'is_afternoon': 10 <= window_start.hour < 17,
        'is_evening': 17 <= window_start.hour < 22,
    }
    
    # Sensor state summary
    sensor_summary = {
        'lights_on': [],
        'lights_off': [],
        'motion_detected': [],
        'media_playing': [],
        'media_idle': [],
        'doors_open': [],
        'persons_home': [],
        'persons_away': [],
        'room_temperatures': {},
        'room_humidity': {},
    }
    
    # Room activity indicators
    room_indicators = {room: [] for room in ROOMS}
    
    for entity_id, state_info in entity_states.items():
        state = state_info.get('state', '')
        entity_lower = entity_id.lower()
        room = extract_room_from_entity(entity_id)
        
        # Lights
        if entity_lower.startswith('light.'):
            if state == 'on':
                sensor_summary['lights_on'].append(entity_id)
                if room:
                    room_indicators[room].append(f'light_on:{entity_id}')
            else:
                sensor_summary['lights_off'].append(entity_id)
        
        # Motion sensors
        elif 'motion' in entity_lower or 'presence' in entity_lower or 'occupancy' in entity_lower:
            if state == 'on':
                sensor_summary['motion_detected'].append(entity_id)
                if room:
                    room_indicators[room].append(f'motion:{entity_id}')
        
        # Media players
        elif entity_lower.startswith('media_player.'):
            if state in ['playing', 'paused']:
                sensor_summary['media_playing'].append(entity_id)
                if room:
                    room_indicators[room].append(f'media:{entity_id}')
            else:
                sensor_summary['media_idle'].append(entity_id)
        
        # Device trackers / person entities
        elif entity_lower.startswith(('device_tracker.', 'person.')):
            name = entity_id.split('.')[-1]
            if state == 'home':
                sensor_summary['persons_home'].append(name)
            elif state in ['not_home', 'away']:
                sensor_summary['persons_away'].append(name)
        
        # Temperature sensors
        elif 'temperature' in entity_lower and entity_lower.startswith('sensor.'):
            try:
                temp = float(state)
                if room:
                    sensor_summary['room_temperatures'][room] = temp
            except (ValueError, TypeError):
                pass
        
        # Humidity sensors
        elif 'humidity' in entity_lower and entity_lower.startswith('sensor.'):
            try:
                humidity = float(state)
                if room:
                    sensor_summary['room_humidity'][room] = humidity
            except (ValueError, TypeError):
                pass
        
        # Doors/windows
        elif 'door' in entity_lower or 'window' in entity_lower:
            if state == 'on':
                sensor_summary['doors_open'].append(entity_id)
                if room:
                    room_indicators[room].append(f'door_open:{entity_id}')
    
    features['sensor_summary'] = sensor_summary
    features['room_indicators'] = {r: inds for r, inds in room_indicators.items() if inds}
    
    # Ground truth labels - person locations
    features['person_locations'] = person_locations
    
    # Check if we have useful presence data
    has_presence = any(loc.get('room', 'unknown') != 'unknown' for loc in person_locations.values())
    has_sensors = bool(sensor_summary['lights_on'] or sensor_summary['motion_detected'] or 
                      sensor_summary['media_playing'] or sensor_summary['persons_home'])
    
    features['has_presence_data'] = has_presence or has_sensors
    features['event_count'] = len(window_entries)
    
    return features


def main():
    parser = argparse.ArgumentParser(description='Extract ML training data from HA logbook')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file')
    parser.add_argument('--append', '-a', action='store_true', help='Append to output')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return 1
    
    with open(input_path) as f:
        data = json.load(f)
    
    entries = data.get('entries', []) if isinstance(data, dict) else data
    
    print(f"Processing {len(entries)} logbook entries...")
    
    samples = process_logbook_entries(entries)
    print(f"Created {len(samples)} training samples with presence data")
    
    output_path = Path(args.output)
    mode = 'a' if args.append else 'w'
    
    with open(output_path, mode) as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Written to {output_path}")
    
    # Summary
    if samples:
        # Count person appearances by room
        room_counts = defaultdict(lambda: defaultdict(int))
        for s in samples:
            for person, loc in s.get('person_locations', {}).items():
                room = loc.get('room', 'unknown')
                if room != 'unknown':
                    room_counts[person][room] += 1
        
        print("\nPerson location counts:")
        for person, rooms in room_counts.items():
            print(f"  {person}: {dict(rooms)}")
        
        print(f"\nTime range: {samples[0]['timestamp']} to {samples[-1]['timestamp']}")
        
        # Time distribution
        hours = defaultdict(int)
        for s in samples:
            hours[s['hour']] += 1
        print(f"Hours covered: {sorted(hours.keys())}")
    
    return 0


if __name__ == '__main__':
    exit(main())
