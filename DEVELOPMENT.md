# Development Setup Guide

This guide will help you set up the WhoLLM integration for local testing in your Home Assistant instance.

## Prerequisites

1. **Home Assistant** installed and running (2024.1.0 or later)
2. **Ollama** installed and running
3. At least one LLM model installed in Ollama

## Step 1: Locate Your Home Assistant Config Directory

The location depends on your Home Assistant installation type:

- **Home Assistant OS (HassOS)**: `/config/` (accessible via Samba or SSH)
- **Home Assistant Container**: Mounted volume (usually `/config/`)
- **Home Assistant Core (Python venv)**: Usually `~/.homeassistant/` or `~/.config/homeassistant/`
- **Home Assistant Supervised**: `/usr/share/hassio/homeassistant/` or `/config/`

## Step 2: Copy Integration to Custom Components

### Option A: Symlink (Recommended for Development)

This allows you to edit code and see changes after restart:

```bash
# Find your HA config directory first, then:
ln -s /path/to/whollm/custom_components/whollm /path/to/your/ha/config/custom_components/whollm
```

### Option B: Copy Files

```bash
# Copy the integration folder
cp -r /path/to/whollm/custom_components/whollm /path/to/your/ha/config/custom_components/
```

## Step 3: Verify Structure

Your Home Assistant config directory should have this structure:

```
config/
├── configuration.yaml
├── custom_components/
│   └── whollm/
│       ├── __init__.py
│       ├── manifest.json
│       ├── config_flow.py
│       ├── const.py
│       ├── coordinator.py
│       ├── sensor.py
│       ├── binary_sensor.py
│       ├── strings.json
│       ├── translations/
│       │   └── en.json
│       └── providers/
│           ├── __init__.py
│           ├── base.py
│           └── ollama.py
└── ...
```

## Step 4: Enable Debug Logging (Optional but Recommended)

Add this to your `configuration.yaml` to see detailed logs:

```yaml
logger:
  default: info
  logs:
    custom_components.whollm: debug
```

## Step 5: Restart Home Assistant

Restart Home Assistant to load the custom integration:

- **Home Assistant OS**: Go to Settings → System → Restart
- **Docker**: `docker restart homeassistant`
- **Python venv**: Restart the Home Assistant service

## Step 6: Verify Integration is Loaded

After restart, check the logs for:

```
INFO (MainThread) [custom_components.whollm] Setting up WhoLLM integration
```

If you see errors, check:
- The `custom_components/whollm` folder exists
- All Python files are present
- `manifest.json` is valid JSON
- Dependencies are installed (aiohttp)

## Step 7: Add Integration via UI

1. Go to **Settings** → **Devices & Services**
2. Click **+ Add Integration**
3. Search for **"WhoLLM"**
4. Follow the setup wizard:
   - Enter Ollama URL (default: `http://localhost:11434`)
   - Select poll interval
   - Choose model
   - Select rooms
   - Enter person/pet names

## Step 8: Verify Entities are Created

After configuration, check that entities are created:

- `sensor.{person_name}_room` - Room sensor for each person/pet
- `binary_sensor.{person_name}_in_{room}` - Binary sensors for each person × room combination

You can verify in:
- **Settings** → **Devices & Services** → **WhoLLM**
- **Developer Tools** → **States** (search for `whollm`)

## Troubleshooting

### Integration Not Appearing

1. **Check logs** for import errors:
   ```bash
   # In Home Assistant logs
   grep -i "whollm" home-assistant.log
   ```

2. **Verify manifest.json**:
   ```bash
   python3 -m json.tool custom_components/whollm/manifest.json
   ```

3. **Check Python syntax**:
   ```bash
   python3 -m py_compile custom_components/whollm/*.py
   ```

### Dependencies Not Installed

If you see `ModuleNotFoundError: No module named 'aiohttp'`:

- **Home Assistant OS**: Dependencies should install automatically
- **Manual install**: The integration will attempt to install via `manifest.json` requirements
- **Check**: Look for `Installing requirements` in logs

### Ollama Connection Issues

1. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Check firewall** if Ollama is on a different machine

3. **Test connection** in the integration config flow - it will test before allowing you to proceed

### Making Code Changes

1. **Edit files** in your development directory
2. **Restart Home Assistant** to reload changes
3. **Check logs** for any new errors
4. **Reload integration** if needed: Settings → Devices & Services → WhoLLM → Reload

## Development Tips

### Quick Restart Script

Create a script to quickly restart HA during development:

```bash
#!/bin/bash
# restart-ha.sh
# Adjust path to your HA instance
systemctl restart home-assistant
# or
docker restart homeassistant
```

### Watch Logs

```bash
# Follow Home Assistant logs
tail -f /path/to/ha/config/home-assistant.log | grep whollm
```

### Test Configuration Flow

1. Remove integration: Settings → Devices & Services → WhoLLM → Delete
2. Restart HA
3. Add integration again to test config flow changes

## Next Steps

- Check the [Home Assistant Developer Documentation](https://developers.home-assistant.io/docs/development_testing/) for testing guidelines
- Review integration logs for any issues
- Test with real sensors and Ollama
- Consider adding unit tests (see testing documentation)

