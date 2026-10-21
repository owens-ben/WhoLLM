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

### Fixed
- `roomer.refresh`, `roomer.correct_room`, `roomer.get_history`, `roomer.clear_history` and `roomer.get_patterns` were defined in `services.py` and listed in `services.yaml` but never registered with Home Assistant. They are now registered at setup.
- `roomer.get_history` and `roomer.get_patterns` now return their results. They are registered as response services and respond with a dict (`{"entity_name", "transitions"}` / `{"entity_name", "patterns"}`), since Home Assistant does not accept bare lists.
- Room transitions are now recorded to `roomer_history.json` (the history tracker was never fed), so `get_history` and `get_patterns` have data. The file lives in the Home Assistant config directory.
- `roomer.get_history` rejects a `limit` outside 1-100 (`limit: 0` used to return the entire history).
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
