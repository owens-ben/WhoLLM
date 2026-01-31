# WhoLLM

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/v/release/owens-ben/WhoLLM?include_prereleases)](https://github.com/owens-ben/WhoLLM/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2024.1%2B-blue.svg)](https://www.home-assistant.io/)

**A Home Assistant integration that uses local LLMs to determine *who* is in which room.**

> **Status: Alpha** — Running in production, but expect rough edges.

---

## The Problem

Occupancy sensors tell you a room is occupied. They don't tell you *who* is there.

Without person-level tracking, you can't:
- Apply personal lighting preferences when someone enters a room
- Know when the whole family is gathered in one place
- Send "you've been at your desk for 3 hours" reminders to a specific person
- Distinguish between a pet and a person triggering motion

## The Solution

WhoLLM uses a local LLM to reason across multiple weak signals—the same way humans do:

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
2. Add `https://github.com/owens-ben/WhoLLM` as an Integration
3. Search for "WhoLLM" and install
4. Restart Home Assistant
5. Go to Settings → Devices & Services → Add Integration → WhoLLM

### Manual Installation

1. Download the `custom_components/whollm` folder from this repository
2. Copy it to your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant
4. Add the integration via Settings → Devices & Services

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

Associate devices with specific people:
- Alice's phone → `device_tracker.alice_iphone`
- Alice's PC → `switch.alice_pc`
- Bob's car → `device_tracker.bob_car`

---

## Entities Created

### Sensors

| Entity | Description |
|--------|-------------|
| `sensor.{name}_room` | Current room for each person/pet |
| `sensor.vision_last_identification` | Last camera identification result |

**Attributes:**
- `confidence` — Prediction confidence (0-100%)
- `indicators` — Signals that contributed to the prediction
- `reasoning` — LLM's explanation (when available)
- `last_updated` — Timestamp of last update

### Binary Sensors

| Entity | Description |
|--------|-------------|
| `binary_sensor.{name}_in_{room}` | Whether person/pet is in specific room |

Example: `binary_sensor.alice_in_office`, `binary_sensor.fido_in_living_room`

---

## Services

### `whollm.identify_person`

Trigger vision-based identification from a camera.

```yaml
service: whollm.identify_person
data:
  camera_entity_id: camera.living_room
  detection_type: person  # or "animal"
```

### `whollm.enable_tracking` / `whollm.disable_tracking`

Control camera auto-tracking for PTZ cameras.

```yaml
service: whollm.enable_tracking
data:
  camera_name: e1_zoom
```

### `whollm.request_vision_update`

Manually trigger vision identification for all configured cameras.

```yaml
service: whollm.request_vision_update
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
Settings → System → Logs → Filter by "whollm"
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
- [ ] Dashboard card

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
