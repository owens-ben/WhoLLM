#!/usr/bin/env python3
"""
Train per-person ML models for room presence prediction.

Creates separate models for each configured person/pet with:
- Per-person feature importance
- Uncertainty estimation for active learning
- Model persistence with metadata

Usage:
    python train_per_person_models.py [--people "Alice,Bob,Max"]
"""

import json
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Paths - relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
MODEL_DIR = PROJECT_DIR / 'models'

# Constants
ROOMS = ['living_room', 'bedroom', 'office', 'kitchen', 'bathroom', 'entry', 'unknown']
# People will be detected from training data or can be passed as argument
PEOPLE = None  # Set dynamically from data
ROOM_TO_IDX = {r: i for i, r in enumerate(ROOMS)}

# Uncertainty threshold for notifications
UNCERTAINTY_THRESHOLD = 0.4  # If max probability < this, ask for help


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
        'computers_on_count': min(len(ss.get('computers_on', [])), 3) / 3.0,
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
        'motion_bedroom': 1.0 if any('bedroom' in m.lower() for m in ss.get('motion_detected', [])) else 0.0,
        'motion_office': 1.0 if any('office' in m.lower() for m in ss.get('motion_detected', [])) else 0.0,
        
        # Room-specific media indicators
        'media_living_room': 1.0 if any('living' in m.lower() or 'tv' in m.lower() for m in ss.get('media_playing', [])) else 0.0,
        'media_bedroom': 1.0 if any('bedroom' in m.lower() for m in ss.get('media_playing', [])) else 0.0,
        'media_office': 1.0 if any('office' in m.lower() for m in ss.get('media_playing', [])) else 0.0,
        
        # Computer/PC indicators (strong for office workers)
        'pc_on': 1.0 if ss.get('computers_on', []) else 0.0,
        
        # LLM confidence as a feature
        'llm_confidence': event.get('confidence', 0.5),
        
        # Indicators count
        'indicator_count': min(len(event.get('indicators', [])), 10) / 10.0,
    }
    
    return features


def load_events(filepath: Path) -> list:
    """Load events from JSONL file."""
    events = []
    with open(filepath) as f:
        for line in f:
            try:
                e = json.loads(line)
                # Skip transition events
                if 'room' in e and 'entity_name' in e:
                    events.append(e)
            except json.JSONDecodeError:
                continue
    return events


def prepare_person_data(events: list, person: str) -> tuple:
    """Prepare training data for a specific person."""
    person_events = [e for e in events if e.get('entity_name') == person]
    
    if not person_events:
        return None, None, None
    
    X = []
    y = []
    
    for event in person_events:
        features = extract_features(event)
        room = event.get('room', 'unknown')
        if room not in ROOM_TO_IDX:
            room = 'unknown'
        
        X.append(list(features.values()))
        y.append(room)
    
    feature_names = list(extract_features(person_events[0]).keys())
    return np.array(X), y, feature_names


def train_person_model(X: np.ndarray, y: list, person: str) -> dict:
    """Train a calibrated model for one person with uncertainty estimation."""
    
    # Split data
    n_samples = len(y)
    split_idx = int(n_samples * 0.8)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = [y[i] for i in train_idx]
    y_val = [y[i] for i in val_idx]
    
    print(f"\n{'='*50}")
    print(f"Training model for: {person}")
    print(f"{'='*50}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Label distribution
    train_dist = defaultdict(int)
    for label in y_train:
        train_dist[label] += 1
    print(f"Label distribution: {dict(train_dist)}")
    
    # Try different models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=8, 
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1,
            random_state=42
        ),
    }
    
    best_model = None
    best_acc = 0
    best_name = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        print(f"  {name}: train={train_acc:.3f}, val={val_acc:.3f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_name = name
    
    # Calibrate the best model for better probability estimates
    print(f"\nBest model: {best_name} (val_acc={best_acc:.3f})")
    print("Calibrating probabilities...")
    
    # Use prefit calibration (sklearn 1.4+ syntax)
    from sklearn import __version__ as sklearn_version
    if int(sklearn_version.split('.')[1]) >= 4:
        # New API: CalibratedClassifierCV with estimator already fitted
        from sklearn.calibration import CalibratedClassifierCV
        calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=2)
        calibrated.fit(X_train, y_train)  # Will use cv for calibration
    else:
        calibrated = CalibratedClassifierCV(best_model, cv='prefit', method='isotonic')
        calibrated.fit(X_val, y_val)
    
    # Final evaluation
    y_pred = calibrated.predict(X_val)
    y_proba = calibrated.predict_proba(X_val)
    
    print(f"\nClassification Report for {person}:")
    print(classification_report(y_val, y_pred, zero_division=0))
    
    # Uncertainty analysis
    max_probs = np.max(y_proba, axis=1)
    uncertain_count = np.sum(max_probs < UNCERTAINTY_THRESHOLD)
    print(f"\nUncertainty Analysis:")
    print(f"  Samples with max_prob < {UNCERTAINTY_THRESHOLD}: {uncertain_count}/{len(y_val)} ({uncertain_count/len(y_val)*100:.1f}%)")
    print(f"  Mean max probability: {np.mean(max_probs):.3f}")
    print(f"  Min max probability: {np.min(max_probs):.3f}")
    
    return {
        'model': calibrated,
        'base_model_name': best_name,
        'val_accuracy': best_acc,
        'classes': calibrated.classes_.tolist(),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'label_distribution': dict(train_dist),
    }


def get_feature_importance(model, feature_names: list, person: str) -> list:
    """Extract feature importance from the model."""
    # Access the base estimator
    base = model.calibrated_classifiers_[0].estimator
    
    if hasattr(base, 'feature_importances_'):
        importances = base.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:10]
        
        print(f"\nTop 10 features for {person}:")
        result = []
        for i, idx in enumerate(sorted_idx):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
            result.append((feature_names[idx], float(importances[idx])))
        return result
    
    return []


def main():
    print("="*60)
    print("Per-Person ML Model Training for Room Presence")
    print("="*60)
    
    # Load all events
    events_path = DATA_DIR / 'ha_events.jsonl'
    print(f"\nLoading events from: {events_path}")
    events = load_events(events_path)
    print(f"Loaded {len(events)} events")
    
    # Create models directory
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Train per-person models
    all_models = {}
    
    for person in PEOPLE:
        X, y, feature_names = prepare_person_data(events, person)
        
        if X is None or len(X) < 50:
            print(f"\nSkipping {person}: insufficient data ({len(X) if X is not None else 0} samples)")
            continue
        
        model_info = train_person_model(X, y, person)
        
        if model_info:
            # Get feature importance
            top_features = get_feature_importance(model_info['model'], feature_names, person)
            model_info['top_features'] = top_features
            model_info['feature_names'] = feature_names
            
            all_models[person] = model_info
    
    # Save all models to single file
    output = {
        'models': {name: info for name, info in all_models.items()},
        'feature_names': feature_names,
        'rooms': ROOMS,
        'people': PEOPLE,
        'uncertainty_threshold': UNCERTAINTY_THRESHOLD,
        'trained_at': datetime.now().isoformat(),
        'total_events': len(events),
    }
    
    model_path = MODEL_DIR / 'per_person_models.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\n{'='*60}")
    print(f"Models saved to: {model_path}")
    print(f"{'='*60}")
    
    # Summary
    print("\nSummary:")
    for person, info in all_models.items():
        print(f"  {person}: {info['base_model_name']}, val_acc={info['val_accuracy']:.3f}, "
              f"train={info['train_size']}, val={info['val_size']}")
    
    print(f"\nUncertainty threshold: {UNCERTAINTY_THRESHOLD}")
    print("When model confidence < threshold, HA notification will be sent for feedback.")
    
    return output


if __name__ == '__main__':
    np.random.seed(42)
    main()
