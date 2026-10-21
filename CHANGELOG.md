# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Test suite with pytest (52 tests covering habits, providers, coordinator)
- CI workflow for running tests on Python 3.11 and 3.12
- pyproject.toml for modern Python project configuration
- Ruff linting configuration
- Comprehensive README with badges, troubleshooting, and examples

### Changed
- **BREAKING: Renamed the project from WhoLLM to Roomer.** The Home Assistant domain changed from `whollm` to `roomer` and the integration now lives in `custom_components/roomer`. Remove the old integration and `custom_components/whollm`, install Roomer, and re-add it; there is no automatic migration.
- **BREAKING:** Services moved from `whollm.*` to `roomer.*`, and events were renamed to `roomer_room_changed`, `roomer_person_arrived`, `roomer_person_left`, `roomer_vision_identification` and `roomer_storage_cleanup`. Automations, blueprints and dashboards that reference the old names must be updated.
- **BREAKING:** Entities are recreated under the new domain (new unique IDs), and storage files are now `roomer_history.json`, `roomer_learned_habits.json` and `roomer_events.jsonl`, so learned habits and history start fresh.
- Docs now lead with the problem (who is in which room, from combined weak signals) rather than the LLM; an LLM provider is still required.
- Removed emojis from indicator strings in provider code
- **Confidence scores are higher and mean more.** The confidence combiner now combines agreeing signals instead of dividing their sum by 2.60, so motion plus the AI scores 0.74 instead of 0.37, and conflicting rooms lower the winner's confidence. Raise any `confidence` thresholds you lowered to cope with the old values (`0.6` to `0.7` is a good start). See the Accuracy & limits docs.
- The Ollama prompt now asks the model for a `CONFIDENCE:` value; `CONFIDENCE: 85` and `85%` are read as 0.85.
- **BREAKING (blueprints):** Welcome Home Lights and Goodbye Routine trigger on the room sensor changing from or to `away` instead of the `roomer_person_arrived` / `roomer_person_left` events. Goodbye Routine takes a list of room sensors and runs only once all of them are still `away` after the delay. Re-create automations made from it.
- Follow Me Music's default minimum confidence is now 0.7.

### Fixed
- Habits were rarely learned, because correct answers almost never reached the 0.7 learning threshold.
- The confidence combiner's "multiple signals agree" bonus could never apply.
- Blueprints used `!input` inside templates without a `variables:` block, so their template conditions (room, confidence, optional inputs) never worked.
- Goodbye Routine stopped at the first optional input left empty (for example no lights), and failed validation when lights, locks, climate or alarm were left empty.
- Welcome Home Lights' "only after dark" check could never pass (`after: sunset` and `before: sunrise` in one condition); it now uses `sun.sun` being below the horizon.
- `automations/vision_on_motion.yaml` referenced `sensor.llm_vision_last_identification`; it is `sensor.vision_last_identification`.
- Home Assistant 2025.x compatibility: use direct import for `async_track_time_interval` instead of deprecated `hass.helpers.event` access

## [0.1.0-alpha] - 2025-01-29

### Added
- Initial release of WhoLLM integration
- LLM-based presence detection using Ollama
- Multi-person and pet tracking
- Configurable room-entity mappings via UI
- Person-device ownership mappings (e.g., assign a PC to a person)
- Confidence combining from multiple signal sources:
  - LLM reasoning
  - Sensor indicators (motion, media, computers, cameras)
  - Learned habit patterns
- Fallback chain when LLM is unavailable
- Habit learning system that learns patterns over time (no hardcoded defaults)
- Vision identification service using Ollama vision models
- Event logging for ML training data collection
- Storage cleanup service with configurable retention
- CrewAI provider support (experimental)

### Providers
- **Ollama**: Primary provider, runs locally
- **CrewAI**: Experimental support for Claude-based reasoning

### Entities Created
- `sensor.{name}_room` - Current room for each person/pet
- `binary_sensor.{name}_in_{room}` - Per-room presence binary sensors
- `sensor.vision_last_identification` - Last vision identification result

### Services
- `whollm.identify_person` - Trigger vision identification from a camera
- `whollm.enable_tracking` / `whollm.disable_tracking` - Camera tracking control
- `whollm.cleanup_storage` - Manual storage cleanup

[Unreleased]: https://github.com/owens-ben/WhoLLM/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/owens-ben/WhoLLM/releases/tag/v0.1.0-alpha
