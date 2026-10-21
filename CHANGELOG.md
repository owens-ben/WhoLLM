# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Device tracker support: a person whose assigned `device_tracker` entities are all away is reported as `away`
- Improved confidence scoring with configurable per-signal weights (Configure, Confidence Weights)
- Custom Lovelace card (`www/whollm-card.js`)
- Standalone web dashboard sign-in with Home Assistant OAuth2, and Playwright browser tests for it
- Four automation blueprints: Welcome Home Lights, Room-Based Climate, Follow Me Music, Goodbye Routine
- Five simulated beta-tester households (`beta-testers/`)
- Modules that are written and tested but **not yet connected** to the running integration:
  - OpenAI and Anthropic providers
  - Movement history and pattern detection
  - Extra services (`refresh`, `correct_room`, `get_history`, `clear_history`, `get_patterns`)
  - Arrival, departure and room-change notification events
  - Multi-zone support
- Documentation guide in `docs/` (getting started, configuration, how it works, entities and services, automations,
  dashboard, troubleshooting, status, development, ML and training data), with diagrams and screenshots
- Scripts to regenerate the documentation images (`scripts/docs/`)
- Test suite with pytest (52 tests covering habits, providers, coordinator)
- CI workflow for running tests on Python 3.11 and 3.12
- pyproject.toml for modern Python project configuration
- Ruff linting configuration
- Comprehensive README with badges, troubleshooting, and examples

### Changed
- README rewritten as a short landing page that links into the new guide
- `DEVELOPMENT.md` moved to `docs/development.md` and updated
- `docs/ML_ENHANCEMENT_PROPOSAL.md` now states clearly that it is a proposal
- Refactored the coordinator, config flow and providers (shared HTTP session, one retry on failure); only the Ollama and
  CrewAI providers are offered in the wizard
- Removed emojis from indicator strings in provider code

### Fixed
- Home Assistant 2025.x compatibility: use direct import for `async_track_time_interval` instead of deprecated `hass.helpers.event` access
- Standalone dashboard: nginx proxy requests were rejected by Home Assistant's trusted proxies check
- Standalone dashboard: OAuth sign-in button did not work in non-secure (plain HTTP) contexts
- `automations/vision_on_motion.yaml` referenced an entity that does not exist; it now uses
  `sensor.vision_last_identification`

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
