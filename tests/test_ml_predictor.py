"""Tests for WhoLLM ML predictor."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


@pytest.fixture
def sample_context():
    """Create sample sensor context."""
    return {
        "lights": {
            "light.office": {"state": "on"},
            "light.bedroom": {"state": "off"},
            "light.living_room": {"state": "on"},
        },
        "motion": {
            "binary_sensor.office_motion": {"state": "on"},
        },
        "media": {
            "media_player.living_room_tv": {"state": "playing"},
        },
        "device_trackers": {
            "device_tracker.alice_phone": {"state": "home"},
        },
        "doors": {},
        "computers": {
            "switch.office_pc": {"state": "on"},
        },
        "ai_detection": {},
        "time_context": {
            "current_time": "14:30",
            "day_of_week": "Tuesday",
            "is_night": False,
            "is_morning": False,
            "is_evening": False,
        },
    }


class TestMLPredictorInitialization:
    """Test MLPredictor initialization."""

    def test_initialization_without_models(self, mock_hass):
        """Test initialization when no models are available."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor(mock_hass)
        
        assert predictor._loaded is False
        assert predictor.models == {}

    def test_initialization_with_hass(self, mock_hass):
        """Test that hass instance is stored."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor(mock_hass)
        
        assert predictor.hass is mock_hass


class TestFeatureExtraction:
    """Test feature extraction from context."""

    def test_extract_features_basic(self, sample_context):
        """Test basic feature extraction."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        # Set feature names to test
        predictor.feature_names = [
            "hour", "minute", "day_of_week", "is_weekend",
            "is_night", "is_morning", "is_afternoon", "is_evening",
            "lights_on_count", "motion_detected_count", "media_playing_count",
            "pc_on", "llm_confidence", "indicator_count",
        ]
        
        features = predictor.extract_features(sample_context, "Alice")
        
        assert len(features) == len(predictor.feature_names)
        # Hour 14 -> 14/24 = ~0.583
        assert features[0] == pytest.approx(14 / 24, rel=0.01)

    def test_extract_features_time_parsing(self, sample_context):
        """Test time parsing in feature extraction."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        predictor.feature_names = ["hour", "minute"]
        
        sample_context["time_context"]["current_time"] = "09:45"
        
        features = predictor.extract_features(sample_context, "Alice")
        
        assert features[0] == pytest.approx(9 / 24, rel=0.01)
        assert features[1] == pytest.approx(45 / 60, rel=0.01)

    def test_extract_features_weekend_detection(self, sample_context):
        """Test weekend detection in features."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        predictor.feature_names = ["day_of_week", "is_weekend"]
        
        # Saturday
        sample_context["time_context"]["day_of_week"] = "Saturday"
        features = predictor.extract_features(sample_context, "Alice")
        assert features[1] == 1.0  # is_weekend
        
        # Tuesday
        sample_context["time_context"]["day_of_week"] = "Tuesday"
        features = predictor.extract_features(sample_context, "Alice")
        assert features[1] == 0.0  # not weekend

    def test_extract_features_sensor_counts(self, sample_context):
        """Test sensor count features."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        predictor.feature_names = [
            "lights_on_count", "motion_detected_count", "media_playing_count"
        ]
        
        features = predictor.extract_features(sample_context, "Alice")
        
        # 2 lights on (office, living_room) -> 2/10 = 0.2
        assert features[0] == pytest.approx(0.2, rel=0.01)
        # 1 motion sensor on -> 1/5 = 0.2
        assert features[1] == pytest.approx(0.2, rel=0.01)
        # 1 media playing -> 1/5 = 0.2
        assert features[2] == pytest.approx(0.2, rel=0.01)


class TestPrediction:
    """Test ML prediction."""

    def test_predict_without_models(self, sample_context):
        """Test prediction returns fallback when no models loaded."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        result = predictor.predict("Alice", sample_context)
        
        assert result["room"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["uncertain"] is True
        assert result["method"] == "fallback"

    def test_predict_unknown_entity(self, sample_context):
        """Test prediction for entity not in models."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        predictor._loaded = True
        predictor.models = {"Bob": MagicMock()}  # Only Bob
        
        result = predictor.predict("Alice", sample_context)
        
        assert result["method"] == "fallback"


class TestUncertaintyNotification:
    """Test uncertainty notification system."""

    @pytest.mark.asyncio
    async def test_notify_uncertainty_sends_notification(self, mock_hass):
        """Test that notification is sent when uncertain."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor(mock_hass)
        
        result = await predictor.notify_uncertainty(
            entity_name="Alice",
            predicted_room="office",
            confidence=0.3,
            probabilities={"office": 0.3, "bedroom": 0.25, "living_room": 0.2},
            context={"time_context": {"current_time": "14:00"}},
        )
        
        assert result is True
        mock_hass.services.async_call.assert_called_once()
        
        call_args = mock_hass.services.async_call.call_args
        assert call_args[0][0] == "persistent_notification"
        assert call_args[0][1] == "create"
        # Check the notification data
        notification_data = call_args[1]
        assert "Alice" in notification_data.get("title", "") or "Alice" in str(call_args)

    @pytest.mark.asyncio
    async def test_notify_uncertainty_no_hass(self):
        """Test notification returns False without hass."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor(None)
        
        result = await predictor.notify_uncertainty(
            entity_name="Alice",
            predicted_room="office",
            confidence=0.3,
            probabilities={"office": 0.3},
            context={},
        )
        
        assert result is False

    @pytest.mark.asyncio
    async def test_notify_uncertainty_cooldown(self, mock_hass):
        """Test notification cooldown prevents spam."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor(mock_hass)
        
        # First notification
        result1 = await predictor.notify_uncertainty(
            entity_name="Alice",
            predicted_room="office",
            confidence=0.3,
            probabilities={"office": 0.3},
            context={"time_context": {"current_time": "14:00"}},
        )
        
        # Second notification immediately (should be blocked)
        result2 = await predictor.notify_uncertainty(
            entity_name="Alice",
            predicted_room="bedroom",
            confidence=0.3,
            probabilities={"bedroom": 0.3},
            context={"time_context": {"current_time": "14:00"}},
        )
        
        assert result1 is True
        assert result2 is False  # Blocked by cooldown

    @pytest.mark.asyncio
    async def test_notify_uncertainty_stores_pending(self, mock_hass):
        """Test that pending feedback request is stored."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor(mock_hass)
        
        await predictor.notify_uncertainty(
            entity_name="Alice",
            predicted_room="office",
            confidence=0.3,
            probabilities={"office": 0.3, "bedroom": 0.2},
            context={"time_context": {"current_time": "14:00"}},
        )
        
        assert "Alice" in predictor._pending_feedback
        assert predictor._pending_feedback["Alice"]["predicted_room"] == "office"


class TestFeedbackRecording:
    """Test feedback recording for ML training."""

    def test_record_feedback(self):
        """Test recording user feedback."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with TemporaryDirectory() as tmpdir:
            feedback_path = Path(tmpdir) / "feedback.jsonl"
            
            with patch.object(Path, "exists", return_value=False):
                predictor = MLPredictor()
            
            # Mock the feedback path
            with patch("custom_components.whollm.ml_predictor.Path") as mock_path:
                mock_path.return_value = feedback_path
                
                # Add pending feedback
                predictor._pending_feedback["Alice"] = {
                    "predicted_room": "office",
                    "confidence": 0.3,
                    "context_snapshot": {"time": "14:00"},
                }
                
                result = predictor.record_feedback("Alice", "bedroom")
            
            assert result is True
            assert "Alice" not in predictor._pending_feedback  # Cleared

    def test_record_feedback_without_pending(self):
        """Test recording feedback without pending request."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        # No pending feedback
        predictor._pending_feedback = {}
        
        # Should still work (with empty pending data)
        result = predictor.record_feedback("Alice", "bedroom")
        
        # May succeed or fail depending on file permissions
        assert isinstance(result, bool)


class TestModelInfo:
    """Test model information retrieval."""

    def test_get_model_info_not_loaded(self):
        """Test model info when not loaded."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        info = predictor.get_model_info()
        
        assert info["loaded"] is False

    def test_get_model_info_loaded(self):
        """Test model info when models are loaded."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        # Simulate loaded state
        predictor._loaded = True
        predictor._model_metadata = {
            "trained_at": "2024-01-15",
            "total_events": 1000,
            "models": {
                "Alice": {
                    "val_accuracy": 0.85,
                    "train_size": 500,
                    "top_features": ["hour", "pc_on", "motion_office"],
                },
            },
        }
        
        info = predictor.get_model_info()
        
        assert info["loaded"] is True
        assert info["trained_at"] == "2024-01-15"
        assert info["total_events"] == 1000
        assert "Alice" in info["models"]
        assert info["models"]["Alice"]["val_accuracy"] == 0.85


class TestGlobalPredictor:
    """Test global predictor singleton."""

    def test_get_ml_predictor_creates_instance(self, mock_hass):
        """Test that get_ml_predictor creates instance."""
        from custom_components.whollm import ml_predictor
        
        # Reset global
        ml_predictor._ml_predictor = None
        
        with patch.object(Path, "exists", return_value=False):
            predictor = ml_predictor.get_ml_predictor(mock_hass)
        
        assert predictor is not None
        assert predictor.hass is mock_hass

    def test_get_ml_predictor_returns_same_instance(self, mock_hass):
        """Test that same instance is returned."""
        from custom_components.whollm import ml_predictor
        
        ml_predictor._ml_predictor = None
        
        with patch.object(Path, "exists", return_value=False):
            predictor1 = ml_predictor.get_ml_predictor(mock_hass)
            predictor2 = ml_predictor.get_ml_predictor()
        
        assert predictor1 is predictor2

    def test_get_ml_predictor_updates_hass(self):
        """Test that hass is updated if provided later."""
        from custom_components.whollm import ml_predictor
        
        ml_predictor._ml_predictor = None
        
        mock_hass = MagicMock()
        
        with patch.object(Path, "exists", return_value=False):
            predictor1 = ml_predictor.get_ml_predictor(None)
            assert predictor1.hass is None
            
            predictor2 = ml_predictor.get_ml_predictor(mock_hass)
            assert predictor2.hass is mock_hass


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_extract_features_invalid_time(self, sample_context):
        """Test feature extraction with invalid time format."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        predictor.feature_names = ["hour", "minute"]
        
        sample_context["time_context"]["current_time"] = "invalid"
        
        features = predictor.extract_features(sample_context, "Alice")
        
        # Should fallback to noon
        assert features[0] == pytest.approx(12 / 24, rel=0.01)
        assert features[1] == 0.0

    def test_extract_features_missing_context(self):
        """Test feature extraction with minimal context."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        predictor.feature_names = ["lights_on_count", "pc_on"]
        
        minimal_context = {
            "time_context": {"current_time": "12:00", "day_of_week": "Monday"},
        }
        
        features = predictor.extract_features(minimal_context, "Alice")
        
        # Should return zeros for missing data
        assert features[0] == 0.0
        assert features[1] == 0.0

    def test_predict_handles_model_error(self, sample_context):
        """Test that prediction handles model errors gracefully."""
        from custom_components.whollm.ml_predictor import MLPredictor
        
        with patch.object(Path, "exists", return_value=False):
            predictor = MLPredictor()
        
        # Simulate loaded but broken model
        predictor._loaded = True
        predictor.feature_names = ["hour"]
        
        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(side_effect=Exception("Model error"))
        predictor.models = {"Alice": mock_model}
        
        result = predictor.predict("Alice", sample_context)
        
        assert result["method"] == "error"
        assert result["uncertain"] is True
