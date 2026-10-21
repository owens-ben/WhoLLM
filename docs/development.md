# Development

[← Status & roadmap](status.md) · [Guide home](README.md) · Next: [ML & training data →](ml-and-training-data.md)

Everything you need to change WhoLLM and try it in a real Home Assistant. For the contribution process (branches, pull
requests, style), see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Where things live

```
custom_components/whollm/
├── __init__.py        setup, camera/storage services, daily cleanup
├── config_flow.py     setup wizard and options menu
├── coordinator.py     the detection pipeline (one poll = one update)
├── providers/         base.py (prompts, parsing, fallback), ollama.py, crewai.py, openai.py, anthropic.py
├── habits.py          HabitPredictor and ConfidenceCombiner (the weighing step)
├── event_logger.py    JSONL event log with retention
├── vision.py          camera snapshot identification and tracking switches
├── sensor.py  binary_sensor.py   the entities
├── const.py  strings.json  translations/en.json  services.yaml
└── notifications.py  zones.py  history.py  services.py  ml_predictor.py     written and tested, not connected yet
```

| Module | In the running integration? |
|---|---|
| `__init__`, `config_flow`, `coordinator`, `habits`, `event_logger`, `vision`, `sensor`, `binary_sensor` | ✅ yes |
| `providers/base`, `ollama`, `crewai` | ✅ yes |
| `providers/openai`, `providers/anthropic` | 🚧 not in the provider registry |
| `notifications`, `zones`, `history`, `services`, `ml_predictor` | 🚧 not imported by the integration (`history` only by `services`) |

The full picture, with user-facing effects, is in [Status & roadmap](status.md). The best place to start reading the
logic is `_process_entity` in `coordinator.py` and `ConfidenceCombiner.combine` in `habits.py`, which the
[How it works](how-it-works.md) page walks through.

## Set up a dev environment

Python 3.11 or newer (CI runs 3.11, 3.12 and 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate

# the same packages CI installs
pip install pytest pytest-asyncio pytest-cov aiohttp voluptuous pillow
pip install pytest-homeassistant-custom-component

# optional
pip install ruff pre-commit
pre-commit install
```

(`pip install -e ".[dev]"` is an alternative that pulls the extras declared in `pyproject.toml`.)

## Run the checks

```bash
pytest tests/ -v                      # unit tests
pytest tests/ --cov=custom_components/whollm --cov-report=term-missing

ruff check custom_components/whollm/  # lint
ruff format custom_components/whollm/ # format
```

CI runs the tests on Python 3.11 to 3.13 and reports lint results without failing the build yet. A separate nightly
workflow runs Home Assistant's **hassfest** and the **HACS** validation.

### Dashboard browser tests

The standalone dashboard has Playwright tests in `e2e/`. They expect the dashboard on port 3380:

```bash
npm install
(cd dashboard && docker compose up -d)   # serves the dashboard on :3380
npx playwright test
```

### Simulated households

[`beta-testers/`](../beta-testers/README.md) holds five simulated home setups (a studio flat, a family with a pet, a
sensor-heavy smart home, a minimal setup and a remote worker):

```bash
python beta-testers/run_all_testers.py
python beta-testers/run_all_testers.py --tester alice
```

## Try your changes in Home Assistant

### 1. Link the integration into your config

Find your Home Assistant config directory:

| Install type | Config directory |
|---|---|
| Home Assistant OS | `/config/` (via Samba or SSH) |
| Container | the volume mounted at `/config/` |
| Core (venv) | usually `~/.homeassistant/` |
| Supervised | `/usr/share/hassio/homeassistant/` |

A symlink means edits show up after a restart, with no copying:

```bash
ln -s /path/to/WhoLLM/custom_components/whollm /path/to/ha/config/custom_components/whollm
```

> [!NOTE]
> The helper scripts `check_integration.sh` and `setup_local_testing.sh` in the repo root still refer to the old name
> `llm_presence` and stop immediately. Use the symlink above instead. See [Status](status.md#known-issues).

### 2. Turn on debug logging

```yaml
logger:
  default: info
  logs:
    custom_components.whollm: debug
```

### 3. Restart and check

Restart Home Assistant (Settings → System → Restart, or `docker restart homeassistant`) and look for this log line:

```text
INFO (MainThread) [custom_components.whollm] Setting up WhoLLM integration
```

If it is missing, check that the folder is in place, `manifest.json` is valid
(`python3 -m json.tool custom_components/whollm/manifest.json`), and that the files compile
(`python3 -m py_compile custom_components/whollm/*.py`).

Then **Settings → Devices & Services → Add Integration → WhoLLM**, and confirm in **Developer Tools → States** that
`sensor.<name>_room` and `binary_sensor.<name>_in_<room>` exist.

### 4. The edit loop

1. Edit the code.
2. Restart Home Assistant. A **Reload** of the integration only re-reads its configuration; it does not re-import
   changed Python files.
3. Watch the logs: `tail -f /path/to/ha/config/home-assistant.log | grep whollm`.

To test config-flow changes, delete the integration, restart, and add it again.

## Working on the docs

The guide in `docs/` is plain Markdown that renders on GitHub. Two kinds of image live in `docs/assets/`, and both
can be regenerated:

| Image | Tool | Run |
|---|---|---|
| Diagrams: `hero.svg`, `pipeline.svg`, `weights.svg`, `wizard.svg` | [`scripts/docs/generate_diagrams.py`](../scripts/docs/generate_diagrams.py) | `python scripts/docs/generate_diagrams.py` |
| Screenshots: `dashboard-live.png`, `dashboard-signin.png`, `lovelace-card.png` | [`scripts/docs/capture_screenshots.js`](../scripts/docs/capture_screenshots.js) | `npm install`, then `node scripts/docs/capture_screenshots.js` |

- `weights.svg` is read straight from `DEFAULT_CONFIDENCE_WEIGHTS` in `const.py`. Re-run the generator whenever the
  default weights change.
- The screenshots are rendered from the **real** dashboard and card code using made-up demo data, through a tiny
  stand-in for Home Assistant. Re-run the script whenever you change how those look. If Playwright's own Chromium is not
  installed, set `CHROMIUM_PATH` to any Chromium or Chrome.
- The numbers in [How it works](how-it-works.md#signals-weights-and-confidence) were produced by running the real
  `ConfidenceCombiner`. If you change the weighing logic, re-check them.

When you change behaviour, update the page that describes it and the [Status](status.md) table if a feature moves
between ✅, ⚠️ and 🚧.

## Common problems

| Problem | Try |
|---|---|
| Integration does not appear | Check the folder path, restart, hard-refresh the browser, read the log. |
| `ModuleNotFoundError: aiohttp` | Home Assistant installs it from `manifest.json`. Check the log for *Installing requirements*. |
| Cannot reach Ollama from Home Assistant | `curl http://<host>:11434/api/tags` from the Home Assistant host. See [Troubleshooting](troubleshooting.md). |
