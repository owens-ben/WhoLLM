# LLM Room Presence

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/release/owens-ben/llm-room-presence.svg)](https://github.com/owens-ben/llm-room-presence/releases)
[![License](https://img.shields.io/github/license/owens-ben/llm-room-presence.svg)](LICENSE)

**Intelligent room presence detection for Home Assistant using local LLMs.**

Traditional presence detection fails when users are stationary (working at desk, watching TV) because motion sensors timeout. This integration uses an LLM to reason about multiple signals holistically - lights, motion, media players, and device trackers - to provide intelligent presence guesses.

## Features

- 🧠 **LLM-Powered** - Uses local LLMs (Ollama) to intelligently deduce room presence
- 👥 **Multi-Person Support** - Track multiple household members independently
- 🐾 **Pet Tracking** - Track pets with pet-specific reasoning logic
- 🏠 **Room-Aware** - Creates binary sensors for each person/pet × room combination
- 🔌 **Bayesian Integration** - Binary sensors feed perfectly into Home Assistant's Bayesian sensors
- 🔒 **Privacy-First** - All processing happens locally with your own LLM

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Home Assistant │────▶│   LLM Presence  │────▶│     Ollama      │
│    Sensors      │     │   Integration   │     │    (Local)      │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  sensor.alice_room        │
                    │  binary_sensor.alice_     │
                    │    in_office/bedroom/   │
                    └─────────────────────────┘
```

1. Every N seconds, the integration collects states from:
   - 💡 Lights (`light.*`)
   - 🏃 Motion sensors (`binary_sensor.*motion*`)
   - 📺 Media players (`media_player.*`)
   - 📱 Device trackers (`device_tracker.*`, `person.*`)

2. Formats context into a prompt for the LLM

3. LLM reasons about the signals and returns a room guess

4. Updates sensor entities that can feed into your Bayesian sensors

## Requirements

- Home Assistant 2024.1.0 or newer
- [Ollama](https://ollama.ai/) running locally or on your network
- A suitable model (recommended: `llama3.2`, `mistral`, or `phi3`)

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Click the three dots menu → Custom repositories
3. Add `https://github.com/owens-ben/llm-room-presence` as an Integration
4. Search for "LLM Room Presence" and install
5. Restart Home Assistant

### Manual

1. Download the latest release from GitHub
2. Copy `custom_components/llm_presence` to your Home Assistant `custom_components` directory
3. Restart Home Assistant

## Configuration

1. Go to Settings → Devices & Services
2. Click "Add Integration"
3. Search for "LLM Room Presence"
4. Follow the setup wizard:
   - Enter your Ollama URL (e.g., `http://192.168.1.101:11434`)
   - Select a model
   - Choose rooms to track
   - Enter names of persons and pets

## Entities Created

For each person/pet configured, the integration creates:

| Entity | Type | Description |
|--------|------|-------------|
| `sensor.{name}_room` | Sensor | Current room guess (office, bedroom, etc.) |
| `binary_sensor.{name}_in_office` | Binary Sensor | Is person/pet in office? |
| `binary_sensor.{name}_in_living_room` | Binary Sensor | Is person/pet in living room? |
| ... | ... | One binary sensor per room |

## Bayesian Integration

The binary sensors integrate perfectly with Home Assistant's Bayesian sensor:

```yaml
binary_sensor:
  - platform: bayesian
    name: "Ben Office"
    prior: 0.1
    probability_threshold: 0.65
    observations:
      # LLM guess - high weight
      - platform: state
        entity_id: binary_sensor.alice_in_office
        to_state: "on"
        prob_given_true: 0.90
        prob_given_false: 0.10
      # Motion sensor
      - platform: state
        entity_id: binary_sensor.office_motion
        to_state: "on"
        prob_given_true: 0.85
        prob_given_false: 0.05
```

## Provider Support

| Provider | Status | Notes |
|----------|--------|-------|
| Ollama | ✅ Supported | Recommended |
| OpenAI | 🔜 Planned | API key required |
| Anthropic | 🔜 Planned | API key required |
| Local (llama.cpp) | 🔜 Planned | Direct model loading |

## Troubleshooting

### "Cannot connect to LLM provider"

1. Verify Ollama is running: `curl http://your-ollama-url:11434/api/tags`
2. Check firewall rules allow connection
3. Ensure the URL includes the port (default: 11434)

### Inaccurate presence detection

1. Try a more capable model (llama3.2 > phi3 > mistral)
2. Reduce poll interval for more frequent updates
3. Check if relevant sensors are available in Home Assistant

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

- Built for the Home Assistant community
- Uses [Ollama](https://ollama.ai/) for local LLM inference

