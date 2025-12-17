#!/usr/bin/env python3
"""
Prepare ML training data from LLM presence events.

This converts the raw events into ML-ready training format with:
- Feature extraction
- Label encoding
- Train/validation split
- Data quality metrics
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import random

# Feature definitions
ROOMS = ['living_room', 'bedroom', 'office', 'kitchen', 'bathroom', 'entry', 'unknown']
PEOPLE = ['Alice', 'Bob', 'Max']
ROOM_TO_IDX = {r: i for i, r in enumerate(ROOMS)}
PERSON_TO_IDX = {p: i for i, p in enumerate(PEOPLE)}


def extract_features(event: dict) -> dict:
    """Extract ML features from an event."""
    tf = event.get('time_features', {})
    ss = event.get('sensor_summary', {})
    
    features = {
        # Time features (normalized)
        'hour': tf.get('hour', 12) / 24.0,
        'minute': tf.get('minute', 0) / 60.0,
        'day_of_week': tf.get('day_of_week', 0) / 6.0,
        'is_weekend': 1.0 if tf.get('is_weekend', False) else 0.0,
        'is_night': 1.0 if tf.get('is_night', False) else 0.0,
        'is_morning': 1.0 if tf.get('is_morning', False) else 0.0,
        'is_afternoon': 1.0 if tf.get('is_afternoon', False) else 0.0,
        'is_evening': 1.0 if tf.get('is_evening', False) else 0.0,
        
        # Sensor counts (normalized)
        'lights_on_count': min(len(ss.get('lights_on', [])), 10) / 10.0,
        'motion_detected_count': min(len(ss.get('motion_detected', [])), 5) / 5.0,
        'media_playing_count': min(len(ss.get('media_playing', [])), 5) / 5.0,
        'trackers_home_count': min(len(ss.get('trackers_home', [])), 10) / 10.0,
        'trackers_away_count': min(len(ss.get('trackers_away', [])), 10) / 10.0,
        'doors_open_count': min(len(ss.get('doors_open', [])), 5) / 5.0,
        'ai_person_detected': 1.0 if ss.get('ai_person_detected', []) else 0.0,
        'ai_animal_detected': 1.0 if ss.get('ai_animal_detected', []) else 0.0,
        
        # Room-specific light indicators
        'light_living_room': 1.0 if any('living' in l.lower() for l in ss.get('lights_on', [])) else 0.0,
        'light_bedroom': 1.0 if any('bedroom' in l.lower() for l in ss.get('lights_on', [])) else 0.0,
        'light_office': 1.0 if any('office' in l.lower() for l in ss.get('lights_on', [])) else 0.0,
        'light_kitchen': 1.0 if any('kitchen' in l.lower() for l in ss.get('lights_on', [])) else 0.0,
        'light_bathroom': 1.0 if any('bath' in l.lower() for l in ss.get('lights_on', [])) else 0.0,
        
        # Room-specific motion indicators  
        'motion_living_room': 1.0 if any('living' in m.lower() for m in ss.get('motion_detected', [])) else 0.0,
        'motion_office': 1.0 if any('office' in m.lower() for m in ss.get('motion_detected', [])) else 0.0,
        
        # Room-specific media indicators
        'media_living_room': 1.0 if any('living' in m.lower() for m in ss.get('media_playing', [])) else 0.0,
        'media_bedroom': 1.0 if any('bedroom' in m.lower() for m in ss.get('media_playing', [])) else 0.0,
        'media_office': 1.0 if any('office' in m.lower() for m in ss.get('media_playing', [])) else 0.0,
        
        # LLM confidence
        'llm_confidence': event.get('confidence', 0.5),
        
        # Indicators count
        'indicator_count': min(len(event.get('indicators', [])), 10) / 10.0,
    }
    
    # Person encoding (one-hot)
    person = event.get('entity_name', 'Alice')
    for p in PEOPLE:
        features[f'is_{p.lower()}'] = 1.0 if person == p else 0.0
    
    # Entity type
    features['is_pet'] = 1.0 if event.get('entity_type') == 'pet' else 0.0
    
    return features


def prepare_sample(event: dict) -> dict | None:
    """Prepare a single training sample."""
    room = event.get('room', 'unknown')
    if room not in ROOM_TO_IDX:
        room = 'unknown'
    
    features = extract_features(event)
    
    return {
        'features': features,
        'label': room,
        'label_idx': ROOM_TO_IDX[room],
        'person': event.get('entity_name', 'unknown'),
        'timestamp': event.get('timestamp', ''),
        'confidence': event.get('confidence', 0),
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare ML training data')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Min confidence filter')
    
    args = parser.parse_args()
    
    # Load events
    events = []
    with open(args.input) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(events)} events")
    
    # Filter by confidence
    if args.min_confidence > 0:
        events = [e for e in events if e.get('confidence', 0) >= args.min_confidence]
        print(f"After confidence filter (>= {args.min_confidence}): {len(events)} events")
    
    # Prepare samples
    samples = []
    for event in events:
        sample = prepare_sample(event)
        if sample:
            samples.append(sample)
    
    print(f"Prepared {len(samples)} samples")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    
    val_size = int(len(samples) * args.val_split)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]
    
    print(f"Train: {len(train_samples)}, Validation: {len(val_samples)}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train data
    train_path = output_dir / 'train.jsonl'
    with open(train_path, 'w') as f:
        for s in train_samples:
            f.write(json.dumps(s) + '\n')
    
    # Save validation data
    val_path = output_dir / 'val.jsonl'
    with open(val_path, 'w') as f:
        for s in val_samples:
            f.write(json.dumps(s) + '\n')
    
    # Save feature names
    feature_names = list(train_samples[0]['features'].keys()) if train_samples else []
    meta = {
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'rooms': ROOMS,
        'people': PEOPLE,
        'train_size': len(train_samples),
        'val_size': len(val_samples),
        'created_at': datetime.now().isoformat(),
    }
    
    meta_path = output_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nSaved to {output_dir}:")
    print(f"  - train.jsonl ({len(train_samples)} samples)")
    print(f"  - val.jsonl ({len(val_samples)} samples)")
    print(f"  - metadata.json")
    
    # Statistics
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Labels (rooms): {ROOMS}")
    
    # Label distribution
    train_labels = defaultdict(int)
    for s in train_samples:
        train_labels[s['label']] += 1
    
    print("\nLabel distribution (train):")
    for label, count in sorted(train_labels.items(), key=lambda x: -x[1]):
        pct = count / len(train_samples) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Per-person distribution
    print("\nPer-person label distribution (train):")
    for person in PEOPLE:
        person_samples = [s for s in train_samples if s['person'] == person]
        person_labels = defaultdict(int)
        for s in person_samples:
            person_labels[s['label']] += 1
        
        print(f"  {person} ({len(person_samples)} samples):")
        for label, count in sorted(person_labels.items(), key=lambda x: -x[1])[:5]:
            pct = count / len(person_samples) * 100 if person_samples else 0
            print(f"    {label}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
