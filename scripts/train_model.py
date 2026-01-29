#!/usr/bin/env python3
"""Train ML model for room presence prediction."""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Paths - relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'ml_ready'
MODEL_DIR = PROJECT_DIR / 'models'

def load_data():
    """Load training and validation data."""
    train = [json.loads(l) for l in open(DATA_DIR / 'train.jsonl')]
    val = [json.loads(l) for l in open(DATA_DIR / 'val.jsonl')]
    meta = json.load(open(DATA_DIR / 'metadata.json'))
    
    X_train = np.array([list(t['features'].values()) for t in train])
    y_train = [t['label'] for t in train]
    X_val = np.array([list(t['features'].values()) for t in val])
    y_val = [t['label'] for t in val]
    
    return X_train, y_train, X_val, y_val, meta, train, val

def train_and_evaluate():
    """Train multiple models and compare."""
    print("Loading data...")
    X_train, y_train, X_val, y_val, meta, train_raw, val_raw = load_data()
    
    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Features: {meta['num_features']}")
    print()
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    }
    
    results = {}
    best_model = None
    best_acc = 0
    
    for name, model in models.items():
        print(f"=== {name} ===")
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Val Accuracy:   {val_acc:.3f}")
        
        results[name] = {'train_acc': train_acc, 'val_acc': val_acc, 'model': model}
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = (name, model)
        print()
    
    # Detailed report for best model
    print(f"=== Best Model: {best_model[0]} (Val Acc: {best_acc:.3f}) ===")
    print()
    
    y_pred = best_model[1].predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    
    # Feature importance (for tree models)
    if hasattr(best_model[1], 'feature_importances_'):
        print("\nTop 10 Important Features:")
        importances = best_model[1].feature_importances_
        feature_names = meta['feature_names']
        sorted_idx = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(sorted_idx):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    # Per-person accuracy
    print("\nPer-Person Validation Accuracy:")
    for person in meta['people']:
        person_idx = [i for i, v in enumerate(val_raw) if v['person'] == person]
        if person_idx:
            X_person = X_val[person_idx]
            y_person = [y_val[i] for i in person_idx]
            person_acc = accuracy_score(y_person, best_model[1].predict(X_person))
            print(f"  {person}: {person_acc:.3f} ({len(person_idx)} samples)")
    
    # Save best model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / 'presence_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model[1],
            'model_name': best_model[0],
            'feature_names': meta['feature_names'],
            'rooms': meta['rooms'],
            'val_accuracy': best_acc,
        }, f)
    print(f"\nModel saved to: {model_path}")
    
    return best_model[1], meta

if __name__ == '__main__':
    train_and_evaluate()
