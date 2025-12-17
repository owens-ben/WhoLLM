#!/bin/bash
# Sync training data from Home Assistant and prepare for ML

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
HA_CONFIG="/path/to/homelab/infrastructure/docker/homeassistant/config"

echo "=== LLM Presence Training Data Sync ==="
echo "Date: $(date)"
echo ""

# Create data directories if they don't exist
mkdir -p "$DATA_DIR/raw" "$DATA_DIR/ml_ready"

# Copy events file from Home Assistant
echo "1. Copying events from Home Assistant..."
if [ -f "$HA_CONFIG/llm_presence_events.jsonl" ]; then
    sudo cp "$HA_CONFIG/llm_presence_events.jsonl" "$DATA_DIR/ha_events.jsonl"
    sudo chown $USER:$USER "$DATA_DIR/ha_events.jsonl"
    EVENT_COUNT=$(wc -l < "$DATA_DIR/ha_events.jsonl")
    echo "   Copied $EVENT_COUNT events"
else
    echo "   WARNING: No events file found at $HA_CONFIG/llm_presence_events.jsonl"
    exit 1
fi

# Get time range
echo ""
echo "2. Analyzing data..."
FIRST_TS=$(head -1 "$DATA_DIR/ha_events.jsonl" | python3 -c "import json,sys; print(json.load(sys.stdin).get('timestamp','unknown'))")
LAST_TS=$(tail -1 "$DATA_DIR/ha_events.jsonl" | python3 -c "import json,sys; print(json.load(sys.stdin).get('timestamp','unknown'))")
echo "   Time range: $FIRST_TS to $LAST_TS"

# Prepare ML training data
echo ""
echo "3. Preparing ML training data..."
python3 "$SCRIPT_DIR/prepare_ml_training_data.py" \
    --input "$DATA_DIR/ha_events.jsonl" \
    --output "$DATA_DIR/ml_ready" \
    --val-split 0.2

echo ""
echo "=== Sync Complete ==="
echo "Training data ready at: $DATA_DIR/ml_ready/"
ls -la "$DATA_DIR/ml_ready/"
