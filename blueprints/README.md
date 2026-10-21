# Roomer Automation Blueprints

Ready-to-use Home Assistant automation blueprints for common Roomer use cases.

## Available Blueprints

### Welcome Home Lights
Turn on lights when someone arrives home.

[![Import Blueprint](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/owens-ben/Roomer/blob/main/blueprints/welcome_home_lights.yaml)

### Room-Based Climate Control
Adjust temperature based on room occupancy.

[![Import Blueprint](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/owens-ben/Roomer/blob/main/blueprints/room_based_climate.yaml)

### Follow Me Music
Transfer music playback as you move between rooms.

[![Import Blueprint](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/owens-ben/Roomer/blob/main/blueprints/follow_me_music.yaml)

### Goodbye Routine
Run actions when leaving home (lights off, lock doors, set thermostat).

[![Import Blueprint](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/owens-ben/Roomer/blob/main/blueprints/goodbye_routine.yaml)

## Installation

### Option 1: Import Button
Click the "Import Blueprint" button above.

### Option 2: Manual Import
1. Go to **Settings** → **Automations & Scenes** → **Blueprints**
2. Click **Import Blueprint**
3. Paste the blueprint URL

### Option 3: Copy Files
1. Copy the `.yaml` files to `/config/blueprints/automation/roomer/`
2. Restart Home Assistant

## Usage

After importing a blueprint:

1. Go to **Settings** → **Automations & Scenes**
2. Click **Create Automation**
3. Select **Use Blueprint**
4. Choose the Roomer blueprint
5. Configure the inputs
6. Save

## Requirements

These blueprints require:
- Roomer integration installed and configured
- At least one person with a room sensor (`sensor.{name}_room`)
- For notifications: Roomer notifications enabled (default)

## Custom Blueprints

Feel free to modify these blueprints or create your own using Roomer events:

- `roomer_person_arrived` - Fired when someone comes home
- `roomer_person_left` - Fired when someone leaves
- `roomer_room_changed` - Fired when someone changes rooms
