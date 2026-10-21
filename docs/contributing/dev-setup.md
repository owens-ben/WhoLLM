# Dev setup

This page gets you from a fresh clone to a passing test run and a running copy of Roomer in your own Home Assistant.

## Prerequisites

- Python 3.11, 3.12 or 3.13. CI tests all three, and `pyproject.toml` sets `requires-python = ">=3.11"`.
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

Install exactly what [`.github/workflows/test.yml`](https://github.com/owens-ben/roomer/blob/main/.github/workflows/test.yml) installs, so your results match CI:

```bash
python -m pip install --upgrade pip
pip install pytest pytest-asyncio pytest-cov aiohttp voluptuous pillow
pip install pytest-homeassistant-custom-component
```

`pytest-homeassistant-custom-component` pulls in a matching `homeassistant` package, which is large and takes a while the first time. For linting, also install ruff:

```bash
pip install ruff
```

!!! note
    `pyproject.toml` also defines a `dev` extra (`pip install -e ".[dev]"`) with pytest, ruff, mypy and `homeassistant`. CI does not use it. If the two ever disagree, trust the workflow file. `CONTRIBUTING.md` mentions a `requirements_dev.txt`, but that file does not exist.

## Run the tests

```bash
pytest tests/ -v --tb=short
```

`pyproject.toml` sets `asyncio_mode = "auto"` and `testpaths = ["tests"]`, so a plain `pytest` works too. To get the coverage report CI produces on Python 3.11:

```bash
pytest tests/ --cov=custom_components/roomer --cov-report=term-missing
```

To run a single file or test while iterating:

```bash
pytest tests/test_providers.py -v
pytest tests/test_coordinator.py -k device_tracker -v
```

!!! warning "Known issue"
    `tests/test_services.py` currently has 13 failures. `custom_components/roomer/services.py` imports `get_habit_predictor` from `habits.py`, and that function does not exist, so the module fails to import. A fix is in progress. Until it lands, expect these failures. Every other test file should pass.

## Lint and format

Ruff is configured in `pyproject.toml`: target `py311`, line length 120, rule sets `E W F I B C4 UP`, with `E501` and `B008` ignored. CI runs:

```bash
ruff check custom_components/roomer/
ruff format --check custom_components/roomer/
```

Both lint steps in CI have `continue-on-error: true`, so they report problems but do not fail the build yet. Please keep new code clean anyway. To auto-fix and format:

```bash
ruff check --fix custom_components/roomer/
ruff format custom_components/roomer/
```

The repo also ships a [`.pre-commit-config.yaml`](https://github.com/owens-ben/roomer/blob/main/.pre-commit-config.yaml) with whitespace, YAML/JSON checks and the ruff hooks. To use it:

```bash
pip install pre-commit
pre-commit install
```

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
    The integration does not register an options update listener. After you change options in the UI, use **Reload** on the integration (or restart) before the coordinator picks them up.

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
