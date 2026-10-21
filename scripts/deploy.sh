#!/bin/bash
# Copy the integration and blueprints to a Home Assistant host over SSH,
# then optionally restart it.
#
# Usage: scripts/deploy.sh [ENV_FILE]   (default: local/.env)
#
# Reads HA_SSH, HA_CONFIG_DIR and HA_RESTART_CMD; see local/.env.example.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-$PROJECT_DIR/local/.env}"

if [ ! -f "$ENV_FILE" ]; then
    echo "No env file at $ENV_FILE (copy local/.env.example to local/.env)" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${HA_SSH:?HA_SSH is not set in $ENV_FILE}"
HA_CONFIG_DIR="${HA_CONFIG_DIR:-/config}"
DEST="$HA_SSH:$HA_CONFIG_DIR/custom_components/whollm/"

echo "Deploying to $DEST"
# rsync creates the last path component only; fresh configs lack custom_components/
ssh "$HA_SSH" "mkdir -p '$HA_CONFIG_DIR/custom_components'"
# models/ is excluded so --delete never removes models trained in place
rsync -av --delete \
    --exclude '__pycache__/' --exclude 'models/' --exclude '.gitkeep' \
    "$PROJECT_DIR/custom_components/whollm/" "$DEST"

echo "Deploying blueprints to $HA_SSH:$HA_CONFIG_DIR/blueprints/automation/whollm/"
ssh "$HA_SSH" "mkdir -p '$HA_CONFIG_DIR/blueprints/automation/whollm'"
rsync -av "$PROJECT_DIR/blueprints/"*.yaml "$HA_SSH:$HA_CONFIG_DIR/blueprints/automation/whollm/"

if [ -n "${HA_RESTART_CMD:-}" ]; then
    echo "Restarting Home Assistant: $HA_RESTART_CMD"
    # shellcheck disable=SC2029
    ssh "$HA_SSH" "$HA_RESTART_CMD"
else
    echo "HA_RESTART_CMD not set; restart Home Assistant to load the changes."
fi
