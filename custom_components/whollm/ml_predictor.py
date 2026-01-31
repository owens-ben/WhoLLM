"""ML-based room presence prediction with uncertainty notifications."""
from __future__ import annotations

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Model path - looks in HA config directory
MODEL_PATH = Path('/config/custom_components/whollm/models/per_person_models.pkl')
# No fallback path - models must be trained for your household

# Notification settings
NOTIFICATION_COOLDOWN = timedelta(minutes=15)  # Don't spam notifications
UNCERTAINTY_THRESHOLD = 0.4  # Ask for help if max probability < this


class MLPredictor:
    """ML-based room presence predictor with uncertainty detection."""
    
    def __init__(self, hass: HomeAssistant | None = None):
        """Initialize the ML predictor."""
        self.hass = hass
        self.models: dict = {}
        self.feature_names: list = []
        self.rooms: list = []
        self.uncertainty_threshold = UNCERTAINTY_THRESHOLD
        self._last_notification: dict[str, datetime] = {}
        self._pending_feedback: dict[str, dict] = {}
        self._loaded = False
        
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load trained models from disk."""
        # Try to load from config path
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                
                self.models = {
                    name: info['model'] 
                    for name, info in data.get('models', {}).items()
                }
                self.feature_names = data.get('feature_names', [])
                self.rooms = data.get('rooms', [])
                self.uncertainty_threshold = data.get('uncertainty_threshold', UNCERTAINTY_THRESHOLD)
                self._model_metadata = data
                self._loaded = True
                
                _LOGGER.info(
                    "Loaded ML models for %s from %s (trained: %s)",
                    list(self.models.keys()),
                    MODEL_PATH,
                    data.get('trained_at', 'unknown')
                )
                return True
                
            except Exception as err:
                _LOGGER.error("Failed to load models from %s: %s", MODEL_PATH, err)
        
        _LOGGER.warning("No ML models found, prediction disabled")
        return False
    
    def extract_features(self, context: dict, entity_name: str) -> list[float]:
        """Extract features from sensor context matching training format."""
        tf = context.get('time_context', {})
        
        # Parse time from context
        current_time = tf.get('current_time', '12:00')
        try:
            hour, minute = map(int, current_time.split(':'))
        except:
            hour, minute = 12, 0
        
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        day_of_week = day_map.get(tf.get('day_of_week', 'Monday'), 0)
        
        # Count sensors
        lights = context.get('lights', {})
        motion = context.get('motion', {})
        media = context.get('media', {})
        trackers = context.get('device_trackers', {})
        doors = context.get('doors', {})
        computers = context.get('computers', {})
        ai_detection = context.get('ai_detection', {})
        
        lights_on = [k for k, v in lights.items() if v.get('state') == 'on']
        motion_detected = [k for k, v in motion.items() if v.get('state') == 'on']
        media_playing = [k for k, v in media.items() if v.get('state') == 'playing']
        trackers_home = [k for k, v in trackers.items() if v.get('state') == 'home']
        trackers_away = [k for k, v in trackers.items() if v.get('state') not in ['home', 'unknown']]
        doors_open = [k for k, v in doors.items() if v.get('state') == 'on']
        computers_on = [k for k, v in computers.items() if v.get('state') in ['on', 'home']]
        ai_person = any(v.get('state') == 'on' and v.get('detection_type') == 'person' 
                        for v in ai_detection.values())
        ai_animal = any(v.get('state') == 'on' and v.get('detection_type') == 'animal' 
                        for v in ai_detection.values())
        
        features = {
            'hour': hour / 24.0,
            'minute': minute / 60.0,
            'day_of_week': day_of_week / 6.0,
            'is_weekend': 1.0 if day_of_week >= 5 else 0.0,
            'is_night': 1.0 if tf.get('is_night', False) else 0.0,
            'is_morning': 1.0 if tf.get('is_morning', False) else 0.0,
            'is_afternoon': 1.0 if 12 <= hour < 18 else 0.0,
            'is_evening': 1.0 if tf.get('is_evening', False) else 0.0,
            
            'lights_on_count': min(len(lights_on), 10) / 10.0,
            'motion_detected_count': min(len(motion_detected), 5) / 5.0,
            'media_playing_count': min(len(media_playing), 5) / 5.0,
            'trackers_home_count': min(len(trackers_home), 10) / 10.0,
            'trackers_away_count': min(len(trackers_away), 10) / 10.0,
            'doors_open_count': min(len(doors_open), 5) / 5.0,
            'computers_on_count': min(len(computers_on), 3) / 3.0,
            'ai_person_detected': 1.0 if ai_person else 0.0,
            'ai_animal_detected': 1.0 if ai_animal else 0.0,
            
            # Room-specific lights
            'light_living_room': 1.0 if any('living' in l.lower() for l in lights_on) else 0.0,
            'light_bedroom': 1.0 if any('bedroom' in l.lower() for l in lights_on) else 0.0,
            'light_office': 1.0 if any('office' in l.lower() for l in lights_on) else 0.0,
            'light_kitchen': 1.0 if any('kitchen' in l.lower() for l in lights_on) else 0.0,
            'light_bathroom': 1.0 if any('bath' in l.lower() for l in lights_on) else 0.0,
            
            # Room-specific motion
            'motion_living_room': 1.0 if any('living' in m.lower() for m in motion_detected) else 0.0,
            'motion_bedroom': 1.0 if any('bedroom' in m.lower() for m in motion_detected) else 0.0,
            'motion_office': 1.0 if any('office' in m.lower() for m in motion_detected) else 0.0,
            
            # Room-specific media
            'media_living_room': 1.0 if any('living' in m.lower() or 'tv' in m.lower() for m in media_playing) else 0.0,
            'media_bedroom': 1.0 if any('bedroom' in m.lower() for m in media_playing) else 0.0,
            'media_office': 1.0 if any('office' in m.lower() for m in media_playing) else 0.0,
            
            # PC on
            'pc_on': 1.0 if computers_on else 0.0,
            
            # LLM confidence (will be filled in by caller if available)
            'llm_confidence': 0.5,
            
            # Indicator count (will be filled in by caller if available)
            'indicator_count': 0.0,
        }
        
        # Return features in correct order
        return [features.get(name, 0.0) for name in self.feature_names]
    
    def predict(
        self,
        entity_name: str,
        context: dict,
        llm_confidence: float = 0.5,
        indicators: list[str] | None = None,
    ) -> dict[str, Any]:
        """Predict room for entity with uncertainty estimation.
        
        Returns:
            Dict with:
            - room: predicted room
            - confidence: model confidence (max probability)
            - probabilities: dict of room -> probability
            - uncertain: True if confidence < threshold
            - method: 'ml' or 'fallback'
        """
        if not self._loaded or entity_name not in self.models:
            return {
                'room': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'uncertain': True,
                'method': 'fallback',
            }
        
        try:
            # Extract features
            features = self.extract_features(context, entity_name)
            
            # Update LLM confidence and indicator count
            llm_conf_idx = self.feature_names.index('llm_confidence') if 'llm_confidence' in self.feature_names else -1
            indicator_idx = self.feature_names.index('indicator_count') if 'indicator_count' in self.feature_names else -1
            
            if llm_conf_idx >= 0:
                features[llm_conf_idx] = llm_confidence
            if indicator_idx >= 0:
                features[indicator_idx] = min(len(indicators or []), 10) / 10.0
            
            # Get model for this person
            model = self.models[entity_name]
            
            # Predict with probabilities
            import numpy as np
            X = np.array([features])
            proba = model.predict_proba(X)[0]
            classes = model.classes_
            
            # Build probability dict
            probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
            
            # Get best prediction
            best_idx = np.argmax(proba)
            room = classes[best_idx]
            confidence = float(proba[best_idx])
            
            uncertain = confidence < self.uncertainty_threshold
            
            return {
                'room': room,
                'confidence': confidence,
                'probabilities': probabilities,
                'uncertain': uncertain,
                'method': 'ml',
            }
            
        except Exception as err:
            _LOGGER.error("ML prediction failed for %s: %s", entity_name, err)
            return {
                'room': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'uncertain': True,
                'method': 'error',
            }
    
    async def notify_uncertainty(
        self,
        entity_name: str,
        predicted_room: str,
        confidence: float,
        probabilities: dict[str, float],
        context: dict,
    ) -> bool:
        """Send HA notification when prediction is uncertain.
        
        Returns True if notification was sent.
        """
        if self.hass is None:
            _LOGGER.warning("Cannot send notification: no hass instance")
            return False
        
        # Check cooldown
        now = datetime.now()
        last = self._last_notification.get(entity_name)
        if last and (now - last) < NOTIFICATION_COOLDOWN:
            _LOGGER.debug("Skipping notification for %s (cooldown)", entity_name)
            return False
        
        # Build notification message
        top_rooms = sorted(probabilities.items(), key=lambda x: -x[1])[:3]
        options_text = ", ".join([f"{r} ({p*100:.0f}%)" for r, p in top_rooms])
        
        message = (
            f"Uncertain about {entity_name}'s location.\n\n"
            f"Best guess: {predicted_room} ({confidence*100:.0f}%)\n"
            f"Options: {options_text}\n\n"
            f"Tap to correct (this helps train the model)."
        )
        
        # Store pending feedback request
        self._pending_feedback[entity_name] = {
            'timestamp': now.isoformat(),
            'predicted_room': predicted_room,
            'confidence': confidence,
            'probabilities': probabilities,
            'context_snapshot': {
                'time': context.get('time_context', {}).get('current_time'),
                'lights_on': len([k for k, v in context.get('lights', {}).items() if v.get('state') == 'on']),
                'media_playing': len([k for k, v in context.get('media', {}).items() if v.get('state') == 'playing']),
            }
        }
        
        try:
            # Use persistent notification (works for all setups)
            # For mobile notifications, set up automations that listen for persistent_notification events
            await self.hass.services.async_call(
                'persistent_notification',
                'create',
                {
                    'title': f'Where is {entity_name}?',
                    'message': message,
                    'notification_id': f'whollm_{entity_name.lower()}',
                },
            )
            self._last_notification[entity_name] = now
            _LOGGER.info("Sent uncertainty notification for %s", entity_name)
            return True
        except Exception as err:
            _LOGGER.error("Notification failed: %s", err)
            return False
    
    def record_feedback(
        self,
        entity_name: str,
        correct_room: str,
        timestamp: str | None = None,
    ) -> bool:
        """Record user feedback for future training.
        
        This appends to a feedback file that will be used in the next training run.
        """
        feedback_path = Path('/config/whollm_feedback.jsonl')
        
        pending = self._pending_feedback.get(entity_name, {})
        
        feedback = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'entity_name': entity_name,
            'correct_room': correct_room,
            'predicted_room': pending.get('predicted_room', 'unknown'),
            'predicted_confidence': pending.get('confidence', 0),
            'context_snapshot': pending.get('context_snapshot', {}),
            'source': 'user_feedback',
        }
        
        try:
            import json
            with open(feedback_path, 'a') as f:
                f.write(json.dumps(feedback) + '\n')
            
            _LOGGER.info("Recorded feedback: %s is in %s", entity_name, correct_room)
            
            # Clear pending
            if entity_name in self._pending_feedback:
                del self._pending_feedback[entity_name]
            
            return True
        except Exception as err:
            _LOGGER.error("Failed to record feedback: %s", err)
            return False
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        if not self._loaded:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'trained_at': self._model_metadata.get('trained_at', 'unknown'),
            'total_events': self._model_metadata.get('total_events', 0),
            'uncertainty_threshold': self.uncertainty_threshold,
            'models': {
                name: {
                    'val_accuracy': info.get('val_accuracy', 0),
                    'train_size': info.get('train_size', 0),
                    'top_features': info.get('top_features', [])[:5],
                }
                for name, info in self._model_metadata.get('models', {}).items()
            }
        }


# Global instance
_ml_predictor: MLPredictor | None = None


def get_ml_predictor(hass: HomeAssistant | None = None) -> MLPredictor:
    """Get or create the global ML predictor."""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLPredictor(hass)
    elif hass is not None and _ml_predictor.hass is None:
        _ml_predictor.hass = hass
    return _ml_predictor
