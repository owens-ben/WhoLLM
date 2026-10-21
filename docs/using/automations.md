# Automations

Roomer's entities work like any other Home Assistant sensor, so you can use them in normal automations. This page covers the blueprints that ship with Roomer and some copy-paste examples.

!!! tip "Which entity should I trigger on?"
    - **"Alice walks into the office"** → `binary_sensor.alice_in_office` turning `on`.
    - **"Alice moved rooms"** or **"Alice left / got home"** → a state change of `sensor.alice_room`.
    - **"Anyone moved rooms / got home / left"** → the [`roomer_room_changed`, `roomer_person_arrived` and `roomer_person_left` events](services-events.md#arrival-departure-and-room-change-events).

    See [Entities](entities.md) for the full list.

## Before you start: avoid flicker

Roomer re-checks every person every few seconds (10 by default), and the answer can briefly change when signals are close. Most automations should wait until a new room has held for a short time before acting. Add a `for:` to the trigger:

```yaml
trigger:
  - platform: state
    entity_id: binary_sensor.alice_in_office
    to: "on"
    for: "00:00:30"   # must stay "on" for 30 seconds
```

Be careful with `confidence` conditions. Values between 0.3 and 0.6 are normal for a correct answer, so a condition like `confidence > 0.8` will rarely pass. If you want one, start around `0.4` and check the actual values in your home first. [Accuracy & limits](accuracy.md) explains why.

## How to add these examples

The examples are written in YAML.

1. Go to **Settings → Automations & scenes → Create automation → Create new automation**.
2. Open the **⋮** menu (top right) and choose **Edit in YAML**.
3. Replace the contents with the example, change the entity IDs to yours, and save.

The examples use the long-standing `trigger:` / `platform:` / `service:` keys, which work on every Home Assistant version Roomer supports. Newer versions will convert them to `triggers:` / `actions:` when you save. That's expected.

## Example 1: office lights on when Alice walks in

```yaml
alias: "Office lights on when Alice arrives"
description: "Turn on the office lights when Roomer puts Alice in the office"
mode: single
trigger:
  - platform: state
    entity_id: binary_sensor.alice_in_office
    to: "on"
    for: "00:00:20"
condition:
  - condition: state
    entity_id: light.office
    state: "off"
action:
  - service: light.turn_on
    target:
      entity_id: light.office
    data:
      brightness_pct: 80
```

To turn the lights off again when she leaves, trigger on the same binary sensor going `off`. Use a longer `for:`, such as five minutes, so a quick trip to the kitchen doesn't plunge the room into darkness:

```yaml
alias: "Office lights off when Alice leaves"
mode: restart
trigger:
  - platform: state
    entity_id: binary_sensor.alice_in_office
    to: "off"
    for: "00:05:00"
action:
  - service: light.turn_off
    target:
      entity_id: light.office
```

!!! warning "A light can keep itself on"
    If `light.office` is also mapped to the office in Roomer, the light being on counts as a sign that someone is in the office. Turning lights on from Roomer and then using those same lights as evidence can keep a person "stuck" in a room. Prefer motion sensors, PCs and media players as the office's signals, or keep a long `for:` on the "off" automation.

## Example 2: notify when someone gets home

This sends a phone notification when Alice's room changes from `away` to any room in the house. It uses the room sensor, so it needs a device tracker assigned to Alice (see [Entities](entities.md) for how `away` works).

```yaml
alias: "Alice is home"
mode: single
trigger:
  - platform: state
    entity_id: sensor.alice_room
    from: "away"
condition:
  # Ignore glitches to "unknown" or "unavailable"
  - condition: template
    value_template: >
      {{ trigger.to_state.state not in ['away', 'unknown', 'unavailable'] }}
action:
  - service: notify.mobile_app_your_phone   # change to your notify service
    data:
      title: "Welcome home"
      message: >
        Alice just got home and is in the
        {{ trigger.to_state.state | replace('_', ' ') }}.
```

For "Alice left", use `to: "away"` instead and change the message.

### Using the arrival event instead

Roomer also fires `roomer_person_arrived` when someone goes from `away` to a room, and `roomer_person_left` when they go the other way. One automation can then cover everyone:

```yaml
alias: "Someone is home"
mode: queued
trigger:
  - platform: event
    event_type: roomer_person_arrived
action:
  - service: notify.mobile_app_your_phone   # change to your notify service
    data:
      title: "Welcome home"
      message: >
        {{ trigger.event.data.entity_name }} just got home and is in the
        {{ trigger.event.data.room | replace('_', ' ') }}.
```

To limit it to one person, add `event_data: {entity_name: Alice}` under the trigger. The events don't fire at startup, and an event trigger can't use `for:`. See [Services & events](services-events.md#arrival-departure-and-room-change-events) for the details and the fields each event carries.

## Example 3: "you've been at your desk too long"

Remind Alice to take a break after two hours in the office without leaving.

```yaml
alias: "Desk break reminder for Alice"
mode: single
trigger:
  - platform: state
    entity_id: binary_sensor.alice_in_office
    to: "on"
    for: "02:00:00"
condition:
  - condition: time
    after: "08:00:00"
    before: "20:00:00"
action:
  - service: notify.mobile_app_alice_phone   # change to Alice's notify service
    data:
      title: "Time for a break"
      message: "You've been in the office for 2 hours. Stand up and stretch!"
```

The two-hour timer starts over whenever `binary_sensor.alice_in_office` drops to `off`, even for one update. If the reminder never arrives, look at the binary sensor's history; brief flips to another room are the usual cause. See [Troubleshooting](troubleshooting.md).

## More ideas

```yaml
# Pause the living-room TV when nobody is left in the living room
alias: "Pause TV when the living room empties"
mode: restart
trigger:
  - platform: state
    entity_id:
      - binary_sensor.alice_in_living_room
      - binary_sensor.bob_in_living_room
    to: "off"
    for: "00:02:00"
condition:
  - condition: state
    entity_id: binary_sensor.alice_in_living_room
    state: "off"
  - condition: state
    entity_id: binary_sensor.bob_in_living_room
    state: "off"
  - condition: state
    entity_id: media_player.living_room_tv
    state: "playing"
action:
  - service: media_player.media_pause
    target:
      entity_id: media_player.living_room_tv
```

!!! note
    Pausing the TV removes a "media playing" signal from the living room, which can make Roomer place people elsewhere sooner. That is usually what you want, but keep the `for:` delay.

More examples are in [`examples/example_automation.yaml`](https://github.com/owens-ben/roomer/blob/main/examples/example_automation.yaml) on GitHub, including a Bayesian sensor that combines Roomer with a light.

## Blueprints

A blueprint is a ready-made automation where you only fill in a form. Roomer ships four, in the [`blueprints/`](https://github.com/owens-ben/roomer/tree/main/blueprints) folder:

| Blueprint | What it does | Triggers on |
|---|---|---|
| **Roomer - Follow Me Music** | When a person changes rooms, pauses the speaker in the old room and plays on the speaker in the new room | A person's room sensor changing |
| **Roomer - Room-Based Climate Control** | Sets a thermostat to an "occupied" temperature when a person enters a chosen room, and back to an "away" temperature when they leave it | A person's room sensor changing |
| **Roomer - Welcome Home Lights** | Turns on lights (optionally only after dark) when a person arrives home, then off again after five minutes | `roomer_person_arrived` event |
| **Roomer - Goodbye Routine** | After a delay, turns off lights, locks doors, sets the thermostat and arms the alarm when someone leaves | `roomer_person_left` event |

!!! warning "Known issues with the bundled blueprints"
    - The blueprints use their inputs (such as `min_confidence` or `target_room`) inside templates without declaring them as `variables`. Home Assistant doesn't make inputs available to templates that way, so the template conditions may fail or always evaluate false.
    - **Follow Me Music** defaults to a minimum confidence of `0.6`, which many correct Roomer answers don't reach. Lower it to around `0.4`.

    Until these are fixed, the hand-written examples above are the more reliable starting point.

### Importing a blueprint

**Option 1: import from the URL**

1. Go to **Settings → Automations & scenes → Blueprints**.
2. Click **Import blueprint**.
3. Paste the blueprint's GitHub URL, for example:

    ```text
    https://github.com/owens-ben/roomer/blob/main/blueprints/follow_me_music.yaml
    ```

    The other files are `room_based_climate.yaml`, `welcome_home_lights.yaml` and `goodbye_routine.yaml`.

4. Click **Preview**, then **Import blueprint**.

**Option 2: copy the files**

1. Copy the `.yaml` files to `/config/blueprints/automation/roomer/`.
2. Go to **Developer Tools → YAML** and click **Automations** under "YAML configuration reloading", or restart Home Assistant.

**Using it:** go to **Settings → Automations & scenes → Create automation**, choose **Create new automation from blueprint**, pick the Roomer blueprint, fill in the inputs (for example **Person Room Sensor** = `sensor.alice_room`) and save.

!!! example "Screenshot needed: blueprint-follow-me-form.png"
    The "Roomer - Follow Me Music" blueprint form filled in: Person Room Sensor set to `sensor.alice_room`, a room-to-speaker mapping, and Minimum Confidence lowered to 0.4.

## Vision identification on camera motion

If you have a camera with person or animal detection, [`automations/vision_on_motion.yaml`](https://github.com/owens-ben/roomer/blob/main/automations/vision_on_motion.yaml) is an example that calls `roomer.identify_person` whenever the camera detects someone. It was written for a specific Reolink camera, so replace every `e1_zoom` entity with your own. Its debounce condition refers to `sensor.llm_vision_last_identification`, an old name; change that to `sensor.vision_last_identification`. See [Services & events](services-events.md) for the service fields.
