# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-27

First release intended for real-home use. Verified end to end against a real
Home Assistant core (2025.1 and 2026.2) with a stand-in Ollama server.

### Added
- OpenAI and Anthropic providers alongside Ollama
- Device-tracker away detection and person-device ownership mapping
- Multi-zone support (work, gym, school, ...) and arrival/departure events
- Movement history, pattern detection and manual room corrections
- Services: `refresh`, `correct_room`, `get_history`, `clear_history`, `get_patterns`
  (`get_history` / `get_patterns` return data via service responses)
- Lovelace card (`www/whollm-card.js`), automation blueprints, and a standalone
  dashboard with Home Assistant OAuth sign-in
- `scripts/install.sh`: installs into a local or SSH-reachable HA config, with
  validation, backup and an Ollama reachability check
- `docs/HOME_SETUP.md`: step-by-step go-live checklist
- End-to-end tests that drive a real HA core (config flow, options flow,
  entities, services, unload)

### Fixed
- Integration services were never registered: `services.py` was not loaded and
  imported a helper removed in an earlier refactor
- Movement history was never recorded: the history tracker was never created
- Options flow crashed on current Home Assistant (`config_entry` has no setter),
  which made Configure unusable
- Saving one Configure page wiped settings saved on another, and option changes
  only applied after a restart (no update listener was registered)
- Config flow leaked an aiohttp session on every connection test
- `identify_person` / `cleanup_storage` results were discarded (no response support)
- Daily cleanup timer was never cancelled on reload
- Blocking file I/O in the event loop during setup and cleanup
- Storage files were hard-coded to `/config`, which broke HA Core installs
- Dashboard called the pre-rename `llm_presence` service and vision sensor
- Training-data sync script looked for the pre-rename events file

### Changed
- Minimum Home Assistant is now 2025.1 (oldest version tested); Python 3.12+
- Dashboard proxy target is configured with `HA_HOST` instead of a hard-coded address
- `setup_local_testing.sh` and `check_integration.sh`, which pointed at the
  pre-rename folder, are replaced by `scripts/install.sh`
- Lint (ruff) now fails CI instead of only reporting

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

[Unreleased]: https://github.com/owens-ben/WhoLLM/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/owens-ben/WhoLLM/compare/v0.1.0...v0.2.0
[0.1.0-alpha]: https://github.com/owens-ben/WhoLLM/releases/tag/v0.1.0
