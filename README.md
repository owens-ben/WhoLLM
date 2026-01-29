# WhoLLM

**Status: Alpha** — Running in production, but expect rough edges.

A Home Assistant integration that answers the question occupancy sensors can't: *who* is in which room?

## The Gap

[Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) solved room-level presence. It tells you the office is occupied. It doesn't tell you it's *Alice* in the office.

Without person-level tracking, you can't:
- Apply personal lighting preferences per person
- Know when the whole family is in one room
- Send "you've been at your desk for 3 hours" reminders to a specific person
- Distinguish between a pet and a person triggering motion

## The Idea

An LLM can reason across weak signals the way humans do:

```
Motion in office
+ Alice's PC is on
+ Alice's phone on home WiFi
+ Bob's car is gone
─────────────────────────
→ Alice is in the office (high confidence)
```

No single sensor is definitive. Together, they're enough for a good guess.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Home Assistant                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Sensors ──┐                                           │
│   Lights ───┤                                           │
│   Media ────┼──→ Context ──→ LLM ──→ Per-Person        │
│   Devices ──┤    Builder      ↓      Sensors           │
│   History ──┘              Ollama                       │
│                           (local)                       │
│                                                         │
│   ┌─────────────────────────────────────────────┐      │
│   │ Fallback Layers                             │      │
│   │  1. LLM reasoning                           │      │
│   │  2. Confidence combiner (heuristics)        │      │
│   │  3. Habit patterns (time-of-day priors)     │      │
│   │  4. Optional: ML models trained on history  │      │
│   └─────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## What Works

- **Multi-person tracking** — Tracks multiple people + pets
- **LLM providers** — Ollama (local), CrewAI, extensible
- **Confidence scoring** — Each prediction explains its reasoning
- **Graceful degradation** — Falls back through heuristics if LLM fails
- **Privacy-first** — All inference runs locally

## What's Next

- [ ] Better prompt engineering for accuracy
- [ ] Event-driven updates (reduce polling)
- [ ] ML model training from your HA history
- [ ] More LLM providers (OpenAI, Anthropic, llama.cpp)

## Installation

### Prerequisites

1. Home Assistant 2024.1+
2. [Ollama](https://ollama.ai) running locally: `curl https://ollama.ai/install.sh | sh`
3. A model: `ollama pull llama3.2`

### HACS

1. Add custom repository: `https://github.com/owens-ben/WhoLLM`
2. Install "WhoLLM"
3. Restart Home Assistant
4. Settings → Devices & Services → Add Integration → WhoLLM

### Manual

Copy `custom_components/whollm` to your Home Assistant config directory.

## Configuration

The setup flow asks for:
- Ollama URL (default: `http://localhost:11434`)
- Model to use
- Rooms to track
- Person names (comma-separated)
- Pet names (optional)
- Poll interval (default: 10s)

## Entities Created

| Entity | Example | Description |
|--------|---------|-------------|
| `sensor.{name}_room` | `sensor.alice_room` | Current room assignment |
| `binary_sensor.{name}_in_{room}` | `binary_sensor.alice_in_office` | Per-room presence |

Attributes include `confidence`, `indicators`, and `reasoning`.

## Example Automations

```yaml
# Personal lighting preferences
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
        kelvin: 4000

# Family movie time detection
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

## How It Works

1. **Context gathering** — Pulls all relevant HA state: lights, motion, media, device trackers, recent history
2. **Prompt construction** — Builds structured context for the LLM with temporal information
3. **LLM inference** — Ollama reasons about who is where
4. **Confidence combination** — Merges LLM output with heuristic signals (PC on → likely in office)
5. **Fallback chain** — If LLM fails, heuristics and time-of-day patterns provide baseline predictions

## Contributing

This is early. There's plenty to improve:

- **Prompt engineering** — Better prompts = better accuracy
- **Testing** — Need unit tests and integration tests
- **Providers** — Add OpenAI, Anthropic, local llama.cpp
- **Docs** — More examples, troubleshooting guides

If you try it and it breaks, [open an issue](https://github.com/owens-ben/WhoLLM/issues).

## Acknowledgments

- [Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) for solving room-level presence
- [Ollama](https://ollama.ai) for making local LLMs accessible

## License

MIT
