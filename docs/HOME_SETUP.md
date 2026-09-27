# Home Setup: Go-Live Checklist

Step-by-step guide to running WhoLLM in a real house. Budget about 30 minutes,
plus model download time. Work through it top to bottom; each step has a check
so you know it worked before moving on.

## 0. Decide where Ollama runs

WhoLLM makes one LLM call per person per poll (default every 30 s), so Ollama
needs a machine that is always on.

| Setup | Good for |
|-------|----------|
| Same box as HA (Container / Core installs) | Small households, CPU is fine with `llama3.2:1b` |
| Separate PC / NAS with a GPU | Faster answers, bigger models, vision |
| HA OS on a Pi / N100 | Run Ollama on *another* machine; the Pi is too slow |

## 1. Install Ollama and pull models

On the Ollama machine:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2          # presence reasoning (~2 GB); use llama3.2:1b on slow CPUs
ollama pull moondream         # optional: camera identification (default vision model)
```

If Ollama runs on a different machine from Home Assistant, make it listen on
your LAN rather than only localhost:

```bash
sudo systemctl edit ollama
# add:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl restart ollama
```

**Check:** from the Home Assistant host, `curl http://<ollama-ip>:11434/api/tags`
lists your models.

## 2. Install the integration

Pick one:

**HACS (recommended):** HACS → Integrations → ⋮ → Custom repositories → add
`https://github.com/owens-ben/WhoLLM` as an Integration → install **WhoLLM**.

**Install script, from a clone of this repo:**

```bash
# HA Container / Core / Supervised, run on the HA host:
./scripts/install.sh /path/to/ha/config --card --blueprints --ollama http://<ollama-ip>:11434

# HA OS, with the "Advanced SSH & Web Terminal" add-on running:
./scripts/install.sh --ssh root@homeassistant.local --card --blueprints --ollama http://<ollama-ip>:11434
```

The script checks it is pointing at a real HA config directory, backs up any
previous install to `<config>/.whollm_backup`, and checks that Ollama is reachable.

Then restart Home Assistant (Settings → System → Restart).

**Check:** Settings → System → Logs shows no `whollm` import errors.

## 3. Run the setup wizard

Settings → Devices & Services → **Add Integration** → **WhoLLM**.

1. **Provider:** `ollama`, URL `http://<ollama-ip>:11434`. The wizard tests the
   connection before continuing.
2. **Model:** pick the model you pulled.
3. **Rooms:** tick the rooms you have; add custom ones comma-separated.
4. **People / pets:** comma-separated names, e.g. `Alice, Bob` and `Rex`.
5. **Room entities:** for each room, pick the sensors that show activity there:
   motion sensors, lights, TVs / media players, PC switches, cameras.

**Check:** `sensor.<name>_room` exists for each person and pet, and updates
within one poll interval.

## 4. Map people to their devices (important)

This is what lets WhoLLM tell people apart. Without it, everyone in the same
room looks alike to the LLM.

WhoLLM → **Configure** → **Person Device Ownership**. For each person, pick
their phone tracker (`device_tracker.*`), PC, watch, and so on.

A phone that is `not_home` makes the person **away** straight away, skipping the LLM.

Changes apply immediately; the integration reloads itself, with no HA restart needed.

## 5. Watch it for a day

Turn on debug logging while tuning, in `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.whollm: debug
```

On each `sensor.<name>_room`, check these attributes:

- `confidence`: consistently below 0.5 means the room needs better sensors.
- `indicators`: which signals drove the guess.
- `raw_response`: what the LLM actually said.

When it gets something wrong, tell it. This also teaches the habit model:

```yaml
service: whollm.correct_room
data:
  entity_id: sensor.alice_room
  room: office
```

After a few days, `whollm.get_patterns` (Developer Tools → Actions, with
"return response") shows the routines it has picked up.

## 6. Optional extras

- **Lovelace card:** add resource `/local/whollm-card.js` (JavaScript module)
  under Settings → Dashboards → Resources, then add a card of `type: custom:whollm-card`.
- **Blueprints:** Settings → Automations → Blueprints → the `whollm` folder.
  Includes welcome-home lights, follow-me music, room-based climate and a goodbye routine.
- **Standalone dashboard:** see [dashboard/README.md](../dashboard/README.md).
  Set `HA_HOST=<ha-ip>:8123` when starting the container.

## Files WhoLLM writes

All in your HA config directory. They contain household movement data, so
leave them out of any config you share publicly:

| File | Contents |
|------|----------|
| `whollm_events.jsonl` | Every prediction (auto-pruned; 30 days / 100 MB by default) |
| `whollm_history.json` | Room-to-room transitions |
| `whollm_learned_habits.json` | Learned time-of-day patterns |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Failed to connect" in the wizard | Ollama is not listening on the LAN (`OLLAMA_HOST=0.0.0.0`) or a firewall blocks 11434 |
| Everyone shows the same room | Complete step 4 (person devices) |
| Sensors flip-flop | Raise the poll interval, or remove noisy sensors from a room |
| Slow or `timeout` guesses | Use `llama3.2:1b`, or raise Timeout in Configure → General Settings |
| Wrong room for hours | Use `whollm.correct_room`; check `indicators` for a mis-mapped sensor |
