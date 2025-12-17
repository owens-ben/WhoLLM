# Machine Learning Enhancement Proposal for LLM Room Presence

## Executive Summary

This document outlines strategies to enhance the LLM Room Presence system with machine learning capabilities that learn from user habits over time, process queued images for timeline building, and integrate with proven open-source face recognition systems.

## Current State

The current system uses:
- **Ollama LLM** for text-based presence reasoning
- **Ollama Vision (LLaVA/Moondream)** for person identification from camera snapshots
- **Home Assistant sensors** for context (lights, motion, media, etc.)

### Limitations
1. Vision identification is slow (~30-60s per image on CPU)
2. No learning from past identifications
3. No habit tracking or pattern recognition
4. No face embedding storage for faster recognition
5. Each identification starts from scratch

---

## Proposed Enhancements

### Phase 1: Face Recognition with Embeddings (Immediate)

**Goal**: Replace slow LLM vision with fast face recognition using embeddings.

#### Option A: Double-Take + CompreFace (Recommended)
- **Double-Take** (1.4k ⭐): Unified UI for face recognition training
  - GitHub: https://github.com/jakowenko/double-take
  - Integrates with Home Assistant via MQTT
  - Supports multiple recognition backends
  
- **CompreFace** (7k ⭐): Self-hosted face recognition API
  - GitHub: https://github.com/exadel-inc/CompreFace
  - REST API for face detection, recognition, verification
  - Stores face embeddings for fast matching
  - **Speed**: ~100ms per image (vs 30-60s with LLM vision)

```yaml
# docker-compose.yml addition
services:
  compreface-core:
    image: exadel/compreface-core:latest
    
  compreface-api:
    image: exadel/compreface-api:latest
    ports:
      - "8000:8000"
    
  compreface-admin:
    image: exadel/compreface-admin:latest
    ports:
      - "8001:8001"
```

#### Option B: InsightFace (More Technical)
- **InsightFace** (27k ⭐): State-of-the-art face analysis
  - GitHub: https://github.com/deepinsight/insightface
  - Python library for embedding generation
  - Can run on CPU or GPU
  - Requires custom integration

#### Integration Architecture

```
Camera → Frigate/HA → Person Detected Event
                           ↓
                    CompreFace API
                           ↓
                    Face Embedding Match
                           ↓
                    "Alice" (confidence: 0.95)
                           ↓
                    Update LLM Presence Sensor
```

### Phase 2: Image Queue & Timeline Database

**Goal**: Store all images and build a historical timeline of presence.

#### Database Schema (SQLite/PostgreSQL)

```sql
CREATE TABLE presence_events (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    person_name VARCHAR(100),
    room VARCHAR(50),
    confidence FLOAT,
    detection_method VARCHAR(50),  -- 'face', 'llm', 'motion', 'sensor'
    camera_entity VARCHAR(100),
    image_path VARCHAR(500),
    embedding BLOB,  -- Face embedding for future matching
    sensor_context JSON,  -- Full sensor state at time of detection
    llm_reasoning TEXT
);

CREATE TABLE habit_patterns (
    id INTEGER PRIMARY KEY,
    person_name VARCHAR(100),
    day_of_week INTEGER,  -- 0-6
    hour INTEGER,  -- 0-23
    typical_room VARCHAR(50),
    confidence FLOAT,
    sample_count INTEGER,
    last_updated DATETIME
);

CREATE TABLE image_queue (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    camera_entity VARCHAR(100),
    image_path VARCHAR(500),
    detection_type VARCHAR(50),  -- 'person', 'animal'
    processed BOOLEAN DEFAULT FALSE,
    priority INTEGER DEFAULT 5
);
```

#### Image Queue Processor

```python
class ImageQueueProcessor:
    """Background processor for queued images."""
    
    def __init__(self, db_path: str, compreface_url: str):
        self.db = sqlite3.connect(db_path)
        self.compreface = CompreFaceClient(compreface_url)
        
    async def process_queue(self):
        """Process queued images during low-activity periods."""
        while True:
            # Get unprocessed images
            images = self.db.execute(
                "SELECT * FROM image_queue WHERE processed = FALSE "
                "ORDER BY priority DESC, timestamp ASC LIMIT 10"
            ).fetchall()
            
            for image in images:
                result = await self.process_image(image)
                self.store_result(result)
                self.mark_processed(image['id'])
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def process_image(self, image: dict) -> dict:
        """Process a single image with face recognition."""
        # Try fast face recognition first
        faces = await self.compreface.detect_faces(image['image_path'])
        
        if faces:
            for face in faces:
                match = await self.compreface.recognize(face)
                if match['confidence'] > 0.7:
                    return {
                        'person': match['name'],
                        'confidence': match['confidence'],
                        'method': 'face_recognition',
                        'embedding': face['embedding']
                    }
        
        # Fall back to LLM vision for unrecognized faces
        return await self.llm_identify(image['image_path'])
```

### Phase 3: Habit Learning & Prediction

**Goal**: Learn daily patterns and predict presence without sensors.

#### Pattern Recognition Algorithm

```python
class HabitLearner:
    """Learn and predict presence patterns from historical data."""
    
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        
    def update_patterns(self, person: str, room: str, timestamp: datetime):
        """Update habit patterns with new observation."""
        day = timestamp.weekday()
        hour = timestamp.hour
        
        # Upsert pattern
        self.db.execute("""
            INSERT INTO habit_patterns 
            (person_name, day_of_week, hour, typical_room, confidence, sample_count, last_updated)
            VALUES (?, ?, ?, ?, 1.0, 1, ?)
            ON CONFLICT (person_name, day_of_week, hour) DO UPDATE SET
                typical_room = CASE 
                    WHEN typical_room = excluded.typical_room 
                    THEN typical_room 
                    ELSE excluded.typical_room 
                END,
                confidence = (confidence * sample_count + 1.0) / (sample_count + 1),
                sample_count = sample_count + 1,
                last_updated = excluded.last_updated
        """, (person, day, hour, room, timestamp))
        
    def predict_location(self, person: str, timestamp: datetime = None) -> dict:
        """Predict where a person is likely to be."""
        if timestamp is None:
            timestamp = datetime.now()
            
        day = timestamp.weekday()
        hour = timestamp.hour
        
        result = self.db.execute("""
            SELECT typical_room, confidence, sample_count
            FROM habit_patterns
            WHERE person_name = ? AND day_of_week = ? AND hour = ?
        """, (person, day, hour)).fetchone()
        
        if result:
            return {
                'predicted_room': result[0],
                'confidence': result[1],
                'based_on_samples': result[2]
            }
        return {'predicted_room': 'unknown', 'confidence': 0.0}
    
    def get_daily_timeline(self, person: str, day: int = None) -> list:
        """Get typical daily timeline for a person."""
        if day is None:
            day = datetime.now().weekday()
            
        return self.db.execute("""
            SELECT hour, typical_room, confidence
            FROM habit_patterns
            WHERE person_name = ? AND day_of_week = ?
            ORDER BY hour
        """, (person, day)).fetchall()
```

#### Machine Learning Models

For more sophisticated prediction, use scikit-learn or XGBoost:

```python
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

class MLPredictor:
    """ML-based presence prediction."""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        
    def train(self, events_df: pd.DataFrame):
        """Train model on historical presence events."""
        # Feature engineering
        features = pd.DataFrame({
            'hour': events_df['timestamp'].dt.hour,
            'day_of_week': events_df['timestamp'].dt.dayofweek,
            'minute': events_df['timestamp'].dt.minute,
            'is_weekend': events_df['timestamp'].dt.dayofweek >= 5,
            # Add sensor features
            'living_room_light_on': events_df['sensor_context'].apply(
                lambda x: x.get('lights', {}).get('living_room', 'off') == 'on'
            ),
            'bedroom_tv_on': events_df['sensor_context'].apply(
                lambda x: x.get('media', {}).get('bedroom_tv', 'off') == 'playing'
            ),
            # Previous room (for transition prediction)
            'prev_room': events_df['room'].shift(1).fillna('unknown')
        })
        
        self.model.fit(features, events_df['room'])
        
    def predict(self, current_context: dict) -> dict:
        """Predict current room based on context."""
        features = self.extract_features(current_context)
        probabilities = self.model.predict_proba([features])[0]
        
        return {
            room: prob 
            for room, prob in zip(self.model.classes_, probabilities)
        }
```

### Phase 4: Continuous Learning Pipeline

**Goal**: System improves automatically over time.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                     │
├─────────────────────────────────────────────────────────────┤
│  Cameras → Images → Queue → Face Recognition → Events DB    │
│  Sensors → State Changes → Context Snapshots → Events DB    │
│  User Corrections → Feedback → Training Data                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Processing Layer                          │
├─────────────────────────────────────────────────────────────┤
│  Background Queue Processor (images during idle time)        │
│  Nightly Habit Pattern Update (aggregate daily data)         │
│  Weekly Model Retraining (improve predictions)               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Layer                          │
├─────────────────────────────────────────────────────────────┤
│  Real-time: Fast face recognition + sensor fusion            │
│  Fallback: LLM reasoning with habit context                  │
│  Confidence: Weighted ensemble of all methods                │
└─────────────────────────────────────────────────────────────┘
```

#### Feedback Loop

```python
class FeedbackCollector:
    """Collect user corrections to improve the model."""
    
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        
    def record_correction(
        self, 
        event_id: int, 
        correct_person: str, 
        correct_room: str
    ):
        """Record when user corrects an identification."""
        self.db.execute("""
            INSERT INTO corrections 
            (event_id, correct_person, correct_room, timestamp)
            VALUES (?, ?, ?, ?)
        """, (event_id, correct_person, correct_room, datetime.now()))
        
        # Update the original event for retraining
        self.db.execute("""
            UPDATE presence_events 
            SET person_name = ?, room = ?, corrected = TRUE
            WHERE id = ?
        """, (correct_person, correct_room, event_id))
        
    def get_training_data(self) -> pd.DataFrame:
        """Get corrected data for model retraining."""
        return pd.read_sql("""
            SELECT * FROM presence_events 
            WHERE corrected = TRUE OR confidence > 0.9
        """, self.db)
```

---

## Implementation Roadmap

### Week 1: Face Recognition Integration
1. Deploy CompreFace Docker containers
2. Create API client for LLM Presence
3. Add face enrollment UI/service
4. Replace LLM vision with CompreFace for known faces

### Week 2: Database & Queue System
1. Set up SQLite/PostgreSQL database
2. Create image queue table and processor
3. Store all presence events with context
4. Build background processing service

### Week 3: Habit Learning
1. Implement pattern extraction from events
2. Create habit prediction service
3. Add habit context to LLM prompts
4. Build daily timeline visualization

### Week 4: ML Model Training
1. Collect 1+ week of presence data
2. Train GradientBoosting model
3. Create ensemble prediction system
4. Add confidence scoring

### Ongoing: Continuous Improvement
- Nightly pattern updates
- Weekly model retraining
- User feedback integration
- Performance monitoring

---

## Storage Requirements

### Image Storage (Data Hog Mode)
- **Per image**: ~50KB (compressed JPEG)
- **Per day**: ~500 images = 25MB
- **Per month**: ~750MB
- **Per year**: ~9GB

### Database Storage
- **Events**: ~1KB per event
- **Per day**: ~500 events = 500KB
- **Per year**: ~180MB

### Total Annual Storage: ~10GB

### Retention Policies
```yaml
retention:
  images:
    full_resolution: 7 days
    thumbnails: 90 days
    face_crops: forever  # For retraining
  events:
    full_context: 30 days
    summary: forever
  patterns:
    hourly: forever
    raw_data: 90 days
```

---

## Projects to Borrow From

### 1. Double-Take
- **What**: Face recognition UI for Home Assistant
- **Borrow**: MQTT integration pattern, training workflow
- **Link**: https://github.com/jakowenko/double-take

### 2. CompreFace
- **What**: Self-hosted face recognition API
- **Borrow**: Face embedding storage, recognition API
- **Link**: https://github.com/exadel-inc/CompreFace

### 3. Frigate
- **What**: NVR with object detection
- **Borrow**: Event processing, camera integration
- **Link**: https://github.com/blakeblackshear/frigate

### 4. HASS Machine Learning
- **What**: Community project for ML in HA
- **Borrow**: Data collection patterns, sensor integration
- **Link**: https://community.home-assistant.io/t/master-thesis-machine-learning-with-home-assistant/233053

### 5. EL-HARP Framework
- **What**: Edge-based activity prediction
- **Borrow**: Gradient boosting approach, feature engineering
- **Link**: https://www.mdpi.com/1424-8220/25/19/6082

### 6. CHARM Model
- **What**: Complex human activity recognition
- **Borrow**: Hierarchical activity classification
- **Link**: https://arxiv.org/abs/2207.07806

---

## Quick Wins (Can Implement Today)

### 1. Add Habit Context to LLM Prompts
```python
def get_habit_context(person: str) -> str:
    """Add habit-based hints to LLM prompt."""
    now = datetime.now()
    
    # Simple rule-based habits (replace with learned patterns later)
    habits = {
        'Alice': {
            (0, 6): 'bedroom (sleeping)',
            (6, 9): 'kitchen or bathroom (morning routine)',
            (9, 17): 'office (working)',
            (17, 22): 'living_room or kitchen (evening)',
            (22, 24): 'bedroom (going to bed)',
        },
        'Bob': {
            # Similar patterns
        }
    }
    
    hour = now.hour
    person_habits = habits.get(person, {})
    for (start, end), location in person_habits.items():
        if start <= hour < end:
            return f"Based on typical patterns, {person} is usually in {location} at this time."
    
    return ""
```

### 2. Store Events for Future Training
```python
async def log_presence_event(
    person: str, 
    room: str, 
    confidence: float,
    context: dict
):
    """Log every presence detection for future ML training."""
    event = {
        'timestamp': datetime.now().isoformat(),
        'person': person,
        'room': room,
        'confidence': confidence,
        'context': context
    }
    
    # Append to JSONL file (simple, no database needed)
    with open('/config/presence_events.jsonl', 'a') as f:
        f.write(json.dumps(event) + '\n')
```

### 3. Add Confidence Weighting
```python
def weighted_presence(
    llm_guess: str,
    llm_confidence: float,
    habit_prediction: str,
    habit_confidence: float,
    sensor_indicators: list
) -> tuple[str, float]:
    """Combine multiple signals with confidence weighting."""
    
    # Weight factors
    weights = {
        'llm': 0.4,
        'habit': 0.3,
        'sensors': 0.3
    }
    
    # If LLM and habits agree, boost confidence
    if llm_guess == habit_prediction:
        return llm_guess, min(1.0, (llm_confidence + habit_confidence) / 2 + 0.2)
    
    # Otherwise, go with higher confidence
    if llm_confidence > habit_confidence:
        return llm_guess, llm_confidence * weights['llm']
    else:
        return habit_prediction, habit_confidence * weights['habit']
```

---

## Conclusion

By integrating face recognition (CompreFace), building a historical database, and implementing habit learning, the LLM Room Presence system can:

1. **Speed up identification**: 100ms vs 30-60s
2. **Learn over time**: Patterns emerge from data
3. **Predict presence**: Know where people are without sensors
4. **Self-improve**: Feedback loop for continuous learning
5. **Build timeline**: Complete history of household activity

The key is to start collecting data now (even without ML) so you have training data when you implement the learning algorithms.



