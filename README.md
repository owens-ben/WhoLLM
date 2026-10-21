# Roomer

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/v/release/owens-ben/Roomer?include_prereleases)](https://github.com/owens-ben/Roomer/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2024.1%2B-blue.svg)](https://www.home-assistant.io/)

**A Home Assistant integration that works out *who* is in which room — not just whether a room is occupied — by combining the weak signals your home already has.**

> **Status: Alpha** — Running, but expect rough edges.

> **Formerly WhoLLM.** This project was renamed from WhoLLM (domain `whollm`) to Roomer (domain `roomer`). The rename is **breaking** — see [Migrating from WhoLLM](#migrating-from-whollm).

---

## The Problem

Occupancy sensors tell you a room is occupied. They don't tell you *who* is there.

Without person-level tracking, you can't:
- Apply personal lighting preferences when someone enters a room
- Know when the whole family is gathered in one place
- Send "you've been at your desk for 3 hours" reminders to a specific person
- Distinguish between a pet and a person triggering motion

## The Solution

Roomer combines multiple weak signals—device trackers, PCs, motion, media, learned habits—the same way a person would. Today the reasoning step is done by an LLM (local via Ollama by default), with heuristic and habit-based fallbacks:

```
Motion in office
+ Alice's PC is on
+ Alice's phone on home WiFi
+ Bob's car is gone
─────────────────────────────
→ Alice is in the office (high confidence)
```

No single sensor is definitive. Together, they're enough for a good guess.

---

## Features

- **Multi-person + pet tracking** — Track everyone in your household
- **Device-based room presence** — A phone, watch or tag located to a room (Bermuda, ESPresense, room-assistant) places its owner there directly
- **Who's in each room** — One sensor per room listing the people and pets in it
- **Privacy-first** — All inference runs locally via Ollama
- **Confidence scoring** — Each prediction includes confidence level and reasoning
- **Vision identification** — Optional camera-based identification using vision LLMs
- **Graceful degradation** — Falls back through heuristics when LLM is unavailable
- **Habit learning** — Learns patterns from your history over time
- **Flexible configuration** — Map specific sensors to rooms and devices to people

---

## Installation

### Prerequisites

1. **Home Assistant 2024.1+**
2. **Ollama** running locally or on your network:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.2        # For text reasoning (~2GB)
   ollama pull llava:7b        # Optional: for vision identification (~4GB)
   ```

### Via HACS (Recommended)

1. Open HACS → Integrations → ⋮ Menu → Custom repositories
2. Add `https://github.com/owens-ben/Roomer` as an Integration
3. Search for "Roomer" and install
4. Restart Home Assistant
5. Go to Settings → Devices & Services → Add Integration → Roomer

### Manual Installation

1. Download the `custom_components/roomer` folder from this repository
2. Copy it to your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant
4. Add the integration via Settings → Devices & Services

### Migrating from WhoLLM

Roomer is the same integration under a new name, but the Home Assistant domain changed from `whollm` to `roomer`, so HA treats it as a new integration. There is no automatic migration.

1. **Remove the old integration** — Settings → Devices & Services → WhoLLM → Delete. Do this first so the old entities are removed and their entity IDs are freed for Roomer to reuse.
2. **Delete** `config/custom_components/whollm` (and remove the old custom repository from HACS if you used it).
3. **Install** `custom_components/roomer` (via HACS or manually, as above) and restart Home Assistant.
4. **Re-add** the integration via Settings → Devices & Services → Add Integration → Roomer, and redo your room/person mappings.
5. **Update anything that references the old names:**
   - **Services:** `whollm.*` → `roomer.*` (e.g. `whollm.identify_person` → `roomer.identify_person`)
   - **Events:** `whollm_room_changed`, `whollm_person_arrived`, `whollm_person_left`, `whollm_vision_identification`, `whollm_storage_cleanup` → `roomer_room_changed`, `roomer_person_arrived`, `roomer_person_left`, `roomer_vision_identification`, `roomer_storage_cleanup`
   - **Entities:** entity IDs are still based on the names you configure (`sensor.{name}_room`, `binary_sensor.{name}_in_{room}`), but they are new entities under the `roomer` platform with new unique IDs. If an old entity was not removed, the new one gets a `_2` suffix — delete the stale entity and rename the new one.
   - **Logging config:** `custom_components.whollm` → `custom_components.roomer`
   - Automations, scripts, blueprints and dashboards that use any of the above.
6. **Learned data starts fresh.** History, learned habits and the event log are now stored as `roomer_history.json`, `roomer_learned_habits.json` and `roomer_events.jsonl`; the old `whollm_*` files are not read and can be deleted.

---

## Configuration

The setup wizard guides you through:

| Option | Description | Default |
|--------|-------------|---------|
| **Provider** | LLM provider (Ollama recommended) | `ollama` |
| **URL** | Ollama server URL | `http://localhost:11434` |
| **Model** | Model for reasoning | `llama3.2` |
| **Rooms** | Rooms to track | — |
| **Persons** | People to track | — |
| **Pets** | Pets to track (optional) | — |
| **Poll Interval** | How often to update (seconds) | `30` |

### Room-Entity Mapping

After initial setup, configure which sensors inform each room:

- **Motion sensors** — `binary_sensor.office_motion`
- **Lights** — `light.office_lights`
- **Media players** — `media_player.living_room_tv`
- **Computers** — `switch.alice_pc`, `device_tracker.alice_pc`
- **Doors** — `binary_sensor.front_door`
- **Cameras with AI** — Cameras that detect people/animals

### Person-Device Mapping

Associate devices with specific people (and pets) under **Configure → Person Device Ownership**:
- Alice's phone → `device_tracker.alice_iphone` (home/away)
- Alice's phone, located to a room → `sensor.alice_phone_area` (room presence, see below)
- Alice's PC → `switch.alice_pc`
- Bob's car → `device_tracker.bob_car`
- Whiskers' collar tag → `sensor.whiskers_tag_area`

### Device-Based Room Presence

This is the most reliable way to tell people apart. Bluetooth room-presence tools such as [Bermuda](https://github.com/agittins/bermuda), [ESPresense](https://espresense.com/) and [room-assistant](https://www.room-assistant.io/) report which room a phone, watch or tag is in. Assign that entity to its owner under Person Device Ownership, and Roomer places them in the room it reports:

| Tool | Typical entity to assign | Reports |
|------|------------------|---------|
| Bermuda | `sensor.<device>_area` | `Office` |
| ESPresense (`mqtt_room`) | `sensor.<device>` | `office` |
| ESPresense Companion | `device_tracker.<device>` | `office` |
| room-assistant | `sensor.<device>_room` | `office` |

Roomer also reads a room from an `area_id`, `area_name`, `area` or `room` attribute.

- **Room names must match** the rooms you set up in Roomer. Capitals, spaces and dashes don't matter, so `Living Room` matches `living_room`. When an `*_area` or `*_room` sensor, or a device tracker, reports a room Roomer doesn't have, a warning in the log names the entity and the value. Rename the area, or the room, so they match.
- **A located person skips the LLM.** Their room comes straight from the device, with `source: device_room` and a confidence of 0.9. If their devices disagree (a phone left on the desk while the watch moves with them), the room most devices report wins, then the device that moved most recently, at 0.7 confidence.
- **GPS trackers are only used for home/away.** A phone's GPS tracker reports HA zones, so a work zone called `Office` is never mistaken for the office room.
- **No room, no override.** When the device reports `unknown`, `not_home` or nothing, the person goes through the normal pipeline.
- **Home/away still applies.** If a home/away tracker says the person left *after* the room device last changed, they are `away`. If the room device changed more recently (Bluetooth usually sees you arrive before Wi-Fi does), the room wins.
- Set a timeout in your Bluetooth tool (for example `away_timeout` for `mqtt_room`), so a phone that leaves stops reporting its last room.

---

## Entities Created

### Sensors

| Entity | Description |
|--------|-------------|
| `sensor.{name}_room` | Current room for each person/pet |
| `sensor.{room}_occupants` | Who is in each room right now |
| `sensor.vision_last_identification` | Last camera identification result |

**`sensor.{name}_room` attributes:**
- `confidence` — Prediction confidence (0-1)
- `indicators` — Signals that contributed to the prediction
- `raw_response` — LLM's answer, or which device located the person
- `source` — Where the room came from: `device_room`, `device_tracker` (away) or `llm`

**`sensor.{room}_occupants`** — the state is the number of **people** in the room, so a pet on its own doesn't count as the room being occupied. Attributes:
- `people` — Names of the people in the room, e.g. `["Alice", "Bob"]`
- `pets` — Names of the pets in the room
- `confidence` — Confidence for each of them, e.g. `{"Alice": 0.9, "Bob": 0.62}`
- `sources` — Where each one's room came from

### Binary Sensors

| Entity | Description |
|--------|-------------|
| `binary_sensor.{name}_in_{room}` | Whether person/pet is in specific room |

Example: `binary_sensor.alice_in_office`, `binary_sensor.fido_in_living_room`

---

## Services

### `roomer.identify_person`

Trigger vision-based identification from a camera.

```yaml
service: roomer.identify_person
data:
  camera_entity_id: camera.living_room
  detection_type: person  # or "animal"
```

### `roomer.enable_tracking` / `roomer.disable_tracking`

Control camera auto-tracking for PTZ cameras.

```yaml
service: roomer.enable_tracking
data:
  camera_name: e1_zoom
```

### `roomer.request_vision_update`

Manually trigger vision identification for all configured cameras.

```yaml
service: roomer.request_vision_update
```

---

## Example Automations

### Personal Lighting Preferences

```yaml
automation:
  - alias: "Alice's office lighting"
    trigger:
      - platform: state
        entity_id: binary_sensor.alice_in_office
        to: "on"
    action:
      - service: light.turn_on
        target:
          entity_id: light.office
        data:
          brightness_pct: 80
          color_temp_kelvin: 4000
```

### Family Movie Night Detection

```yaml
automation:
  - alias: "Everyone in living room"
    trigger:
      - platform: template
        value_template: >
          {{ is_state('binary_sensor.alice_in_living_room', 'on') and
             is_state('binary_sensor.bob_in_living_room', 'on') }}
    action:
      - service: scene.turn_on
        target:
          entity_id: scene.movie_mode
```

### Lights Off When a Room Empties

```yaml
automation:
  - alias: "Office lights off when nobody's there"
    trigger:
      - platform: state
        entity_id: sensor.office_occupants
        to: "0"
        for:
          minutes: 5
    action:
      - service: light.turn_off
        target:
          entity_id: light.office
```

### Desk Time Reminder

```yaml
automation:
  - alias: "Alice desk reminder"
    trigger:
      - platform: state
        entity_id: binary_sensor.alice_in_office
        to: "on"
        for:
          hours: 3
    action:
      - service: notify.alice_phone
        data:
          message: "You've been at your desk for 3 hours. Time for a break!"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Home Assistant                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Sensors ──┐                                               │
│   Lights ───┤                                               │
│   Media ────┼──→ Context ──→ Ollama ──→ Per-Person         │
│   Devices ──┤    Builder       │        Sensors            │
│   Cameras ──┘                  │                            │
│                                ▼                            │
│                     ┌─────────────────────┐                 │
│                     │  Fallback Layers    │                 │
│                     │  1. LLM reasoning   │                 │
│                     │  2. Heuristics      │                 │
│                     │  3. Habit patterns  │                 │
│                     │  4. ML models       │                 │
│                     └─────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### "Timeout" or low confidence predictions

The LLM is taking too long to respond. Try:
- Use a smaller model: `ollama pull llama3.2:1b`
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Increase poll interval to reduce load

### Integration not loading

Check Home Assistant logs:
```
Settings → System → Logs → Filter by "roomer"
```

Common issues:
- Ollama URL incorrect or unreachable
- Missing model (run `ollama pull <model>`)

### Predictions seem wrong

1. Check the `indicators` attribute on the sensor to see what signals are being used
2. Verify room-entity mappings are correct
3. Consider adding more sensors to improve accuracy

### Vision identification not working

- Ensure you have a vision model: `ollama pull llava:7b`
- Configure vision model in integration options
- Check camera entity is accessible

---

## Performance Tips

| Scenario | Recommendation |
|----------|----------------|
| Slow predictions | Use smaller model (`llama3.2:1b`, `phi3:mini`) |
| High CPU usage | Increase poll interval to 60+ seconds |
| GPU available | Configure Ollama to use GPU for faster inference |
| Many people/rooms | Consider dedicated Ollama instance |

---

## Roadmap

- [ ] Event-driven updates (reduce polling)
- [ ] ML model training from HA history
- [ ] More LLM providers (OpenAI, Anthropic, local llama.cpp)
- [ ] Better prompt engineering
- [x] Dashboard card (`www/roomer-card.js`, people and rooms)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas that need help:
- **Prompt engineering** — Better prompts = better accuracy
- **Testing** — More test coverage
- **Documentation** — Examples, troubleshooting guides

---

## Acknowledgments

- [Ollama](https://ollama.ai) — Making local LLMs accessible
- [Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) — Inspiration for room-level presence

---

## License

MIT License — see [LICENSE](LICENSE) for details.
