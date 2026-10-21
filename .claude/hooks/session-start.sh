#!/bin/bash
# Install test and lint dependencies for Claude Code cloud sessions.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

command -v uv >/dev/null 2>&1 || python3 -m pip install --quiet uv

# uv resolves a homeassistant release compatible with this Python; plain
# pip backtracks for a long time. Reusing .venv keeps reruns fast.
[ -x .venv/bin/python ] || uv venv --quiet .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install --quiet -r requirements_test.txt "ruff==0.16.9"

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export VIRTUAL_ENV=\"$PWD/.venv\"" >> "$CLAUDE_ENV_FILE"
  echo "export PATH=\"$PWD/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
