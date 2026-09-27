#!/usr/bin/env bash
# Install WhoLLM into a Home Assistant config directory.
#
# Local (HA Container / Core / Supervised, run on the HA host):
#   ./scripts/install.sh /path/to/ha/config
#
# Remote over SSH (HA OS with the "Advanced SSH & Web Terminal" add-on):
#   ./scripts/install.sh --ssh root@homeassistant.local
#
# Options:
#   --ssh USER@HOST     Copy to /config on a remote host over SSH
#   --ssh-port PORT     SSH port (default 22)
#   --card              Also install the Lovelace card to www/whollm-card.js
#   --blueprints        Also install automation blueprints
#   --ollama URL        Check that Ollama is reachable and has a model pulled
#   --dry-run           Show what would be copied, change nothing
#
# Re-running is safe: the previous install is moved to <config>/.whollm_backup
# first (outside custom_components/, so HA never loads it as a second copy).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_DIR/custom_components/whollm"

SSH_TARGET=""
SSH_PORT=22
HA_CONFIG=""
WITH_CARD=0
WITH_BLUEPRINTS=0
OLLAMA_URL=""
DRY_RUN=0

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ssh) SSH_TARGET="$2"; shift 2 ;;
        --ssh-port) SSH_PORT="$2"; shift 2 ;;
        --card) WITH_CARD=1; shift ;;
        --blueprints) WITH_BLUEPRINTS=1; shift ;;
        --ollama) OLLAMA_URL="${2%/}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage 0 ;;
        -*) echo "Unknown option: $1" >&2; usage 1 ;;
        *) HA_CONFIG="$1"; shift ;;
    esac
done

if [[ -n "$SSH_TARGET" ]]; then
    HA_CONFIG="${HA_CONFIG:-/config}"
elif [[ -z "$HA_CONFIG" ]]; then
    echo "Error: pass your HA config directory, or --ssh USER@HOST" >&2
    usage 1
fi

step() { printf '\n==> %s\n' "$*"; }
ok() { printf '    ok: %s\n' "$*"; }
fail() { printf '    FAIL: %s\n' "$*" >&2; exit 1; }

# Run a shell snippet on the target (locally or over SSH)
on_target() {
    if [[ -n "$SSH_TARGET" ]]; then
        ssh -p "$SSH_PORT" "$SSH_TARGET" "$1"
    else
        bash -c "$1"
    fi
}

# Copy a directory's contents to a path on the target
copy_dir() {
    local from="$1" to="$2"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    would copy $from -> $to"
        return
    fi
    if [[ -n "$SSH_TARGET" ]]; then
        tar -C "$from" --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitkeep' -cf - . \
            | ssh -p "$SSH_PORT" "$SSH_TARGET" "mkdir -p '$to' && tar -C '$to' -xf -"
    else
        mkdir -p "$to"
        tar -C "$from" --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitkeep' -cf - . | tar -C "$to" -xf -
    fi
}

copy_file() {
    local from="$1" to="$2"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    would copy $from -> $to"
        return
    fi
    if [[ -n "$SSH_TARGET" ]]; then
        ssh -p "$SSH_PORT" "$SSH_TARGET" "mkdir -p '$(dirname "$to")' && cat > '$to'" < "$from"
    else
        mkdir -p "$(dirname "$to")"
        cp "$from" "$to"
    fi
}

step "Checking source"
[[ -f "$SRC/manifest.json" ]] || fail "integration not found at $SRC"
VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$SRC/manifest.json")"
python3 -m py_compile "$SRC"/*.py "$SRC"/providers/*.py || fail "Python syntax error in integration"
ok "WhoLLM $VERSION, sources compile"

step "Checking Home Assistant config at ${SSH_TARGET:+$SSH_TARGET:}$HA_CONFIG"
on_target "test -f '$HA_CONFIG/configuration.yaml'" \
    || fail "no configuration.yaml in $HA_CONFIG (is this the HA config directory?)"
ok "found configuration.yaml"

DEST="$HA_CONFIG/custom_components/whollm"
BACKUP="$HA_CONFIG/.whollm_backup"

step "Installing integration to $DEST"
if [[ $DRY_RUN -eq 0 ]]; then
    on_target "if [ -d '$DEST' ]; then rm -rf '$BACKUP' && mv '$DEST' '$BACKUP'; fi"
fi
copy_dir "$SRC" "$DEST"
[[ $DRY_RUN -eq 1 ]] || on_target "test -f '$DEST/manifest.json'" || fail "copy did not land"
ok "integration installed (previous copy, if any, kept at $BACKUP)"

if [[ $WITH_CARD -eq 1 ]]; then
    step "Installing Lovelace card"
    copy_file "$REPO_DIR/www/whollm-card.js" "$HA_CONFIG/www/whollm-card.js"
    ok "add resource /local/whollm-card.js (JavaScript module) under Settings -> Dashboards -> Resources"
fi

if [[ $WITH_BLUEPRINTS -eq 1 ]]; then
    step "Installing blueprints"
    for bp in "$REPO_DIR"/blueprints/*.yaml; do
        copy_file "$bp" "$HA_CONFIG/blueprints/automation/whollm/$(basename "$bp")"
    done
    ok "blueprints available under Settings -> Automations -> Blueprints after reload"
fi

if [[ -n "$OLLAMA_URL" ]]; then
    step "Checking Ollama at $OLLAMA_URL"
    if TAGS="$(curl -fsS --max-time 5 "$OLLAMA_URL/api/tags")"; then
        MODELS="$(printf '%s' "$TAGS" | python3 -c 'import json,sys; print(", ".join(m["name"] for m in json.load(sys.stdin).get("models", [])))')"
        if [[ -n "$MODELS" ]]; then
            ok "reachable; models: $MODELS"
        else
            echo "    warning: Ollama is up but has no models. Run: ollama pull llama3.2"
        fi
    else
        echo "    warning: could not reach Ollama from this machine."
        echo "    Home Assistant must be able to reach it; if Ollama runs elsewhere, start it with"
        echo "    OLLAMA_HOST=0.0.0.0 so it listens on your LAN, not just localhost."
    fi
fi

cat <<EOF

Done. Next steps:
  1. Restart Home Assistant (Settings -> System -> Restart).
  2. Settings -> Devices & Services -> Add Integration -> "WhoLLM".
  3. After setup, open WhoLLM -> Configure -> "Person Device Ownership" and map each
     person's phone / PC / watch trackers. This is what lets WhoLLM tell people apart.
  4. Optional debug logging, in configuration.yaml:
       logger:
         logs:
           custom_components.whollm: debug

See docs/HOME_SETUP.md for the full go-live checklist.
EOF
