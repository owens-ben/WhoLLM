# WhoLLM Automation Blueprints

Ready-to-use Home Assistant automation blueprints for common WhoLLM use cases.

> **Status:** *Room-Based Climate* and *Follow Me Music* work today. *Welcome Home Lights* and *Goodbye Routine*
> listen for events (`whollm_person_arrived`, `whollm_person_left`) that the integration does not fire yet, so they will
> not trigger. See [Automations](../docs/automations.md#blueprints) for working recipes in the meantime.

## Available Blueprints

### Welcome Home Lights
Turn on lights when someone arrives home. *(Waits for `whollm_person_arrived`, not fired yet.)*

[![Import Blueprint](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/owens-ben/WhoLLM/blob/main/blueprints/welcome_home_lights.yaml)

### Room-Based Climate Control
Adjust temperature based on room occupancy.

[![Import Blueprint](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/owens-ben/WhoLLM/blob/main/blueprints/room_based_climate.yaml)

### Follow Me Music
When you change rooms, pause the speaker you left and press play on the one you entered.

[![Import Blueprint](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/owens-ben/WhoLLM/blob/main/blueprints/follow_me_music.yaml)

### Goodbye Routine
Run actions when leaving home (lights off, lock doors, set thermostat). *(Waits for `whollm_person_left`, not fired yet.)*

[![Import Blueprint](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/owens-ben/WhoLLM/blob/main/blueprints/goodbye_routine.yaml)

## Installation

### Option 1: Import Button
Click the "Import Blueprint" button above.

### Option 2: Manual Import
1. Go to **Settings** → **Automations & Scenes** → **Blueprints**
2. Click **Import Blueprint**
3. Paste the blueprint URL

### Option 3: Copy Files
1. Copy the `.yaml` files to `/config/blueprints/automation/whollm/`
2. Restart Home Assistant

## Usage

After importing a blueprint:

1. Go to **Settings** → **Automations & Scenes**
2. Click **Create Automation**
3. Select **Use Blueprint**
4. Choose the WhoLLM blueprint
5. Configure the inputs
6. Save

## Requirements

These blueprints require:
- WhoLLM integration installed and configured
- At least one person with a room sensor (`sensor.{name}_room`)
- For arrival and departure detection: a `device_tracker` assigned to the person (Configure, Person Device Ownership),
  so the sensor can report `away`

## Custom Blueprints

Feel free to modify these blueprints or create your own. Today the dependable trigger is a state change of
`sensor.{name}_room` (or of a `binary_sensor.{name}_in_{room}`), as the climate and music blueprints do.

These events are designed but **not fired yet**, so blueprints that wait for them do nothing until they are connected:

- `whollm_person_arrived` - when someone comes home
- `whollm_person_left` - when someone leaves
- `whollm_room_changed` - when someone changes rooms

Details: [Status & roadmap](../docs/status.md).
