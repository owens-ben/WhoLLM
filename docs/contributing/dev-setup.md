# Dev setup

This page gets you from a fresh clone to a passing test run and a running copy of Roomer in your own Home Assistant.

## Prerequisites

- Python 3.13 or 3.14. CI tests both: on 3.14 the test dependencies resolve to the current Home Assistant release, on 3.13 to the last release that supported it (2026.2).
- Git.
- To try changes against a real house: a Home Assistant instance (2024.1.0 or later, per `hacs.json`) and an [Ollama](https://ollama.com) server with at least one model pulled.

## Clone and create a virtualenv

```bash
git clone https://github.com/owens-ben/roomer.git
cd roomer
python3 -m venv .venv
source .venv/bin/activate
```

## Install test dependencies

CI installs [`requirements_test.txt`](https://github.com/owens-ben/roomer/blob/main/requirements_test.txt); do the same so your results match:

```bash
pip install -r requirements_test.txt
```

`pytest-homeassistant-custom-component` pulls in a matching `homeassistant` package along with pytest, pytest-asyncio and pytest-cov. It is large and takes a while the first time. For linting, also install ruff at the version CI uses:

```bash
pip install ruff==0.17.1
```

!!! note
    `pyproject.toml` also defines a `dev` extra (`pip install -e ".[dev]"`). CI does not use it. If the two ever disagree, trust `requirements_test.txt` and [`.github/workflows/ci.yml`](https://github.com/owens-ben/roomer/blob/main/.github/workflows/ci.yml).

## Run the tests

```bash
pytest
```

`pyproject.toml` sets `asyncio_mode = "auto"`, `testpaths = ["tests"]` and `-v --tb=short`. CI also prints a coverage report:

```bash
pytest --cov=custom_components/roomer --cov-report=term-missing:skip-covered
```

To run a single file or test while iterating:

```bash
pytest tests/test_providers.py -v
pytest tests/test_coordinator.py -k device_tracker -v
```

## Lint and format

Ruff is configured in `pyproject.toml`: target `py311`, line length 120, rule sets `E W F I B C4 UP`, with `E501` and `B008` ignored. CI checks the whole repository and fails the build on any finding:

```bash
ruff check .
ruff format --check .
```

To auto-fix and format:

```bash
ruff check --fix .
ruff format .
```

The repo also ships a [`.pre-commit-config.yaml`](https://github.com/owens-ben/roomer/blob/main/.pre-commit-config.yaml) with whitespace, YAML/JSON checks and the same ruff version as CI. To use it:

```bash
pip install pre-commit
pre-commit install
```

## Dashboard end-to-end tests

The standalone dashboard in `dashboard/` has Playwright tests in `e2e/`. They need Node 22; Playwright starts a static file server for `dashboard/` on port 3380 by itself, and no Home Assistant is needed (HA API calls are mocked).

```bash
npm ci
npx playwright install chromium   # once; or set PW_CHANNEL=chrome to use an installed Google Chrome
npx playwright test
```

CI runs them only when `dashboard/`, `e2e/` or the Playwright/npm config change.

## Test against a real Home Assistant

Unit tests mock HA. To check config-flow screens, entity names and real LLM responses, load your working copy into a real HA instance. Symlinking is the most convenient option: you edit files in your clone and restart HA to pick up the changes.

### `setup_local_testing.sh`

[`setup_local_testing.sh`](https://github.com/owens-ben/roomer/blob/main/setup_local_testing.sh) links (or copies) `custom_components/roomer` into your HA config directory. It is interactive:

1. Checks that `custom_components/roomer` exists next to the script.
2. Looks for a directory containing `configuration.yaml` in `~/.homeassistant`, `~/.config/homeassistant`, `/config`, `/usr/share/hassio/homeassistant` and `/data/homeassistant`. If none matches, it prompts you for a path.
3. Creates `<config>/custom_components/` if needed.
4. If `<config>/custom_components/roomer` already exists, offers to replace it with a symlink, replace it with a copy, or cancel. Both replace options `rm -rf` the existing directory. Otherwise it asks whether to symlink, and copies if you answer no.
5. Prints next steps: restart HA, add the integration, enable debug logging.

```bash
./setup_local_testing.sh
```

To do the same by hand:

```bash
ln -s "$(pwd)/custom_components/roomer" /path/to/ha/config/custom_components/roomer
```

!!! tip
    A symlink only works if HA can see the target path. If HA runs in Docker, the clone has to be inside a mounted volume, or you have to copy the files instead.

### `check_integration.sh`

[`check_integration.sh`](https://github.com/owens-ben/roomer/blob/main/check_integration.sh) is a quick sanity check once the integration is linked:

```bash
./check_integration.sh /config   # argument defaults to /config
```

It:

1. Fails if `<config>/custom_components/roomer` is not a **symlink**. A copied install is reported as missing.
2. Checks that `manifest.json`, `__init__.py` and `config_flow.py` are present.
3. Pretty-prints the `domain`, `name` and `version` lines from `manifest.json`.
4. Runs `docker logs homeassistant --tail 50` and greps for Roomer-related errors. This step assumes HA runs in a Docker container named `homeassistant`.

### Enable debug logging

Add this to `configuration.yaml` and restart HA:

```yaml
logger:
  default: info
  logs:
    custom_components.roomer: debug
```

At debug level the coordinator logs each prompt length, the raw LLM reply, combiner decisions ("Confidence combiner changed Alice from office to living_room ...") and room transitions. On a successful load you should see:

```text
INFO (MainThread) [custom_components.roomer] Setting up Roomer integration
```

### Edit, reload, repeat

1. Edit files in your clone.
2. Restart Home Assistant. Python changes need a restart, not just an integration reload.
3. Watch the logs, for example with `tail -f /path/to/config/home-assistant.log | grep roomer` or `docker logs -f homeassistant 2>&1 | grep -i roomer`.
4. To test config-flow changes, delete the Roomer integration under **Settings → Devices & Services**, restart and add it again.

!!! note
    Saving options triggers the update listener registered in `async_setup_entry`, which reloads the entry through `hass.config_entries.async_reload()`. Code changes still need a restart.

### Troubleshooting a local install

```bash
# Is the manifest valid JSON?
python3 -m json.tool custom_components/roomer/manifest.json

# Any syntax errors?
python3 -m py_compile custom_components/roomer/*.py custom_components/roomer/providers/*.py

# Is Ollama reachable from where HA runs?
curl http://localhost:11434/api/tags
```

`DEVELOPMENT.md` in the repo root has more detail on where the HA config directory lives for each installation type.

## Where to go next

- [Architecture](architecture.md): how the pieces fit together.
- [Adding a provider](adding-provider.md): plug in another LLM backend.
- [ML models](ml-models.md): the offline training scripts.
- [Releasing](releasing.md): cutting a version.
