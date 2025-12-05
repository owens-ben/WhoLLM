# LLM Room Presence

A Home Assistant custom integration that extends [Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) to intelligently deduce **who** is in which room at any given time using local LLMs.

## The Problem

While [Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) excels at determining **if** a room is occupied, it doesn't tell you **who** is in that room. This limitation makes it difficult to:
- Personalize automations for specific household members
- Track individual presence across multiple rooms
- Create person-specific smart home behaviors
- Understand household activity patterns per person

## The Solution

**LLM Room Presence** extends Area Occupancy Detection by using a local LLM (Ollama) to intelligently deduce **who** is in which room based on contextual analysis:

- 🧠 **Temporal Context Analysis** - Considers the timeline of events before the current moment
  - Example: If motion is detected in the living room and only one person is home, high confidence that person is in the living room
- 💡 **Multi-Signal Reasoning** - Analyzes multiple signals together:
  - Which lights are on?
  - Recent motion activity patterns?
  - Media playing?
  - Device tracker locations?
  - Historical occupancy patterns?
- 👥 **Person-Specific Logic** - Uses knowledge about household members to make intelligent guesses
- 🐾 **Pet Tracking** - Applies pet-specific logic for tracking furry family members

The LLM considers all signals holistically, along with temporal context and household composition, to make an intelligent guess about **which person** is in **which room** at any given time.

## Features

### Implemented (v0.1.0)
- 🧠 **LLM-Powered** - Local inference with Ollama
- 👥 **Multi-Person** - Track multiple household members
- 🐾 **Pet Tracking** - Track pets with pet-specific logic
- 🏠 **Binary Sensors** - Creates binary sensors for each person/pet × room combination
- 🔒 **Privacy-First** - All processing happens locally
- ⚙️ **Configurable** - Customizable poll intervals and room selection
- 📊 **Confidence Scores** - Each detection includes confidence level and indicators
- 🔄 **Real-time Updates** - Configurable polling interval (default: 10 seconds)
- ⏱️ **Temporal Context** - Analyzes timeline of events to improve accuracy (e.g., if only one person is home and motion detected, high confidence assignment)
- 🔗 **Area Occupancy Integration** - Works alongside [Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) to add person-level tracking

### Coming Soon
- Additional LLM providers (OpenAI, Anthropic)
- Bayesian sensor integration
- Custom room names
- Historical tracking

## Prerequisites

1. **Home Assistant** 2024.1.0 or later
2. **Ollama** installed and running (see [Ollama Installation](https://ollama.ai))
3. At least one LLM model installed in Ollama (e.g., `llama3.2`, `mistral`, etc.)
4. **(Optional but Recommended)** [Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) integration - While not required, this integration works best when used alongside Area Occupancy Detection, which provides room-level occupancy probabilities that can inform person-level assignments

### Installing Ollama

```bash
# Linux/macOS
curl https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai
```

### Installing a Model

```bash
ollama pull llama3.2
# or
ollama pull mistral
```

## Installation

### HACS Installation (Recommended)

1. Install [HACS](https://hacs.xyz) if you haven't already
2. Go to HACS → Integrations
3. Click the three dots (⋮) in the top right
4. Select "Custom repositories"
5. Add this repository URL: `https://github.com/owens-ben/llm-room-presence`
6. Set category to "Integration"
7. Click "Add"
8. Search for "LLM Room Presence" and install it
9. Restart Home Assistant

### Manual Installation

1. Copy the `custom_components/llm_presence` folder to your Home Assistant `custom_components` directory
2. Restart Home Assistant
3. Go to Settings → Devices & Services → Add Integration
4. Search for "LLM Room Presence"

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for **LLM Room Presence**
3. Enter your Ollama URL (default: `http://localhost:11434`)
4. Select your poll interval (how often to check presence, default: 10 seconds)
5. Choose the LLM model to use
6. Select which rooms to track
7. Enter person names (comma-separated, e.g., "Alice, Bob")
8. Optionally enter pet names (comma-separated, e.g., "Max, Luna")

## Usage

After configuration, the integration creates:

### Sensors
- `sensor.{person_name}_room` - Shows the current room for each person/pet
  - Attributes: `confidence`, `raw_response`, `indicators`

### Binary Sensors
- `binary_sensor.{person_name}_in_{room}` - `on` when person/pet is in that room
  - Attributes: `confidence`, `current_room`

### Example Automations

```yaml
# Turn on lights when someone enters a room
automation:
  - alias: "Turn on office lights when Alice enters"
    trigger:
      - platform: state
        entity_id: binary_sensor.alice_in_office
        to: "on"
    action:
      - service: light.turn_on
        target:
          entity_id: light.office_lights

# Use in Bayesian sensor
binary_sensor:
  - platform: bayesian
    name: "Alice in Office (High Confidence)"
    prior: 0.5
    observations:
      - entity_id: binary_sensor.alice_in_office
        prob_given_true: 0.95
        prob_given_false: 0.05
```

## How It Works

1. The integration gathers sensor data from Home Assistant:
   - Light states
   - Motion sensor states
   - Media player states
   - Device tracker/person states
   - Historical occupancy patterns (from Area Occupancy Detection if available)

2. **Temporal Context Analysis**: The system considers the timeline of events leading up to the current moment:
   - Recent motion detections across rooms
   - Who is currently home (from device trackers)
   - Recent state changes in lights, media, and appliances
   - Example: If motion is detected in the living room and only one person is home, the system can confidently assign that person to the living room

3. This contextual information is sent to Ollama with a prompt asking **which person/pet** is in **which room**

4. The LLM analyzes all signals together, considering temporal context and household composition, and returns a room assignment with confidence for each person/pet

5. Results are exposed as sensors and binary sensors in Home Assistant, allowing for person-specific automations

## Troubleshooting

### "Cannot connect to Ollama"
- Ensure Ollama is running: `ollama serve` or check if the service is running
- Verify the URL is correct (default: `http://localhost:11434`)
- Check firewall settings if Ollama is on a different machine

### "No models available"
- Install at least one model: `ollama pull llama3.2`
- Check available models: `ollama list`

### Low accuracy
- Try a larger model (e.g., `llama3.1` instead of `llama3.2`)
- Ensure you have good sensor coverage (lights, motion sensors)
- Adjust poll interval if needed

## Project Plan / Roadmap

### Current Status (v0.1.0)
✅ **Complete and Functional**
- Ollama provider fully implemented
- Multi-person and pet tracking
- Configurable rooms and poll intervals
- Sensor and binary sensor platforms
- Full Home Assistant integration with config flow

### Planned Features

#### Phase 1: Additional LLM Providers
- [ ] **OpenAI Provider** - Support for OpenAI API (GPT-3.5, GPT-4)
- [ ] **Anthropic Provider** - Support for Claude API
- [ ] **Local Provider** - Direct integration with local model files

#### Phase 2: Enhanced Features
- [ ] **Bayesian Integration** - Direct integration with Home Assistant Bayesian sensors
- [ ] **Confidence Thresholds** - Configurable confidence levels per room/person
- [ ] **Historical Tracking** - Track presence history and patterns
- [ ] **Custom Room Names** - Allow users to define custom room names beyond defaults
- [ ] **Sensor Filtering** - Allow users to select which sensors to include in context

#### Phase 3: Advanced Features
- [ ] **Time-based Logic** - Consider time of day in presence detection
- [ ] **Learning Mode** - Learn from user corrections to improve accuracy
- [ ] **Multi-home Support** - Track presence across multiple Home Assistant instances
- [ ] **Webhook Triggers** - Event-driven updates instead of polling

### Contributing

Contributions are welcome! Areas where help is especially appreciated:
- Implementing additional LLM providers (OpenAI, Anthropic)
- Improving prompt engineering for better accuracy
- Adding tests and documentation
- Performance optimizations

## Development

```bash
git clone https://github.com/owens-ben/llm-room-presence.git
cd llm-room-presence
```

## License

MIT License - see [LICENSE](LICENSE) for details.
