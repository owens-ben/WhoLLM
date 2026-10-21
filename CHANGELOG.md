# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Options toggle for arrival/departure events (`notifications_enabled`)
- Activity Log blueprint: posts room changes with confidence and indicators to any notify action (Slack, Discord, mobile)
- Test suite with pytest (341 tests)
- CI workflow for running tests on Python 3.11 and 3.12
- pyproject.toml for modern Python project configuration
- Ruff linting configuration
- Comprehensive README with badges, troubleshooting, and examples

### Changed
- CI now fails on ruff lint/format errors (ruff pinned to 0.16.9); integration code is lint-clean
- Test dependencies moved to `requirements_test.txt` (plain `homeassistant` instead of the pytest plugin, which the tests don't use)
- Removed emojis from indicator strings in provider code
- Storage paths (event log, habits, feedback, ML models) now follow the HA config directory instead of a hardcoded `/config`
- ML models now load from `<config>/whollm/models/` so HACS updates don't remove them (the old location still works)
- Dashboard nginx config is now a template: set `HA_URL` instead of editing the file
- Added `local/` overlay and `scripts/deploy.sh` for household-specific settings

### Fixed
- `refresh`, `correct_room`, `get_history`, `clear_history` and `get_patterns` services were never registered, and `services.py` failed to import after the singleton removal. `get_history` and `get_patterns` now return responses
- Arrival, departure and room-change events never fired because `NotificationManager` was never created. They now fire, except for the first reading after startup
- Room transition history was never recorded. It's now saved every 10 transitions and on unload; the periodic save also no longer runs on every transition once the buffer is full
- Dashboard vision panel and "identify" button used the old `llm_presence` domain and entity IDs
- `setup_local_testing.sh`, `check_integration.sh` and `sync_training_data.sh` referenced the old `llm_presence` names
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
