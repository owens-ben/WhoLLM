# Automations

[← Entities & services](entities-and-services.md) · [Guide home](README.md) · Next: [Dashboard →](dashboard.md)

Recipes you can paste, blueprints you can import, and the habits that keep person-based automations from flapping.

## What to trigger on

| You want to react when… | Trigger on |
|---|---|
| Alice **enters** a room | `binary_sensor.alice_in_office` → `on` |
| Alice **leaves** a room | `binary_sensor.alice_in_office` → `off` |
| Alice moves anywhere | state change of `sensor.alice_room` |
| Alice **leaves home** | `sensor.alice_room` → `away` (needs a [device tracker](configuration.md#person-device-ownership)) |
| Alice **arrives home** | `sensor.alice_room` from `away` |
| Several people are together | a template trigger over their binary sensors |

## Three habits for stable automations

1. **Add `for:`** to triggers so a one-poll wobble does not fire anything. `for: "00:00:30"` for lights,
   minutes for climate.
2. **Check confidence** before acting on a shaky guess:
   `{{ state_attr('sensor.alice_room', 'confidence') | float(0) >= 0.6 }}`.
3. **Ignore `unknown` and `unavailable`** when you use "from" or "any change" triggers.

## Recipes

### Personal lighting when someone enters a room

```yaml
automation:
  - alias: "Alice's office lighting"
    trigger:
      - platform: state
        entity_id: binary_sensor.alice_in_office
        to: "on"
        for: "00:00:20"
    condition:
      - condition: template
        value_template: "{{ state_attr('sensor.alice_room', 'confidence') | float(0) >= 0.6 }}"
    action:
      - service: light.turn_on
        target:
          entity_id: light.office
        data:
          brightness_pct: 80
          color_temp_kelvin: 4000
```

### Lights off when a room has been empty for a while

```yaml
automation:
  - alias: "Bedroom lights off after Alice leaves"
    trigger:
      - platform: state
        entity_id: binary_sensor.alice_in_bedroom
        to: "off"
        for: "00:05:00"
    condition:
      - condition: state
        entity_id: light.bedroom
        state: "on"
    action:
      - service: light.turn_off
        target:
          entity_id: light.bedroom
```

With several people, also require the other people's sensors for that room to be `off`.

### Everyone is in the living room

```yaml
automation:
  - alias: "Family movie night"
    trigger:
      - platform: template
        value_template: >
          {{ is_state('binary_sensor.alice_in_living_room', 'on') and
             is_state('binary_sensor.bob_in_living_room', 'on') }}
        for: "00:02:00"
    action:
      - service: scene.turn_on
        target:
          entity_id: scene.movie_mode
```

### Desk-time reminder

```yaml
automation:
  - alias: "Alice desk reminder"
    trigger:
      - platform: state
        entity_id: binary_sensor.alice_in_office
        to: "on"
        for:
          hours: 3
    action:
      - service: notify.alice_phone
        data:
          message: "You've been at your desk for 3 hours. Time for a break!"
```

A Home Assistant restart resets the timer.

### Pet alert

```yaml
automation:
  - alias: "Max is in the kitchen"
    trigger:
      - platform: state
        entity_id: binary_sensor.max_in_kitchen
        to: "on"
    action:
      - service: notify.mobile_app_phone
        data:
          title: "Pet alert"
          message: "Max is in the kitchen!"
```

### Welcome home

Works today by watching the room sensor leave `away`:

```yaml
automation:
  - alias: "Welcome Alice home"
    trigger:
      - platform: state
        entity_id: sensor.alice_room
        from: "away"
    condition:
      - condition: template
        value_template: "{{ trigger.to_state.state not in ['unknown', 'unavailable'] }}"
    action:
      - service: light.turn_on
        target:
          entity_id: light.entry
```

### Leaving home

```yaml
automation:
  - alias: "Alice left"
    trigger:
      - platform: state
        entity_id: sensor.alice_room
        to: "away"
        for: "00:05:00"      # wait in case she comes straight back
    action:
      - service: light.turn_off
        target:
          area_id: living_room
```

### More

| Where | What |
|---|---|
| [`examples/example_automation.yaml`](../examples/example_automation.yaml) | Lights on/off, pet alert, and a Bayesian sensor that fuses WhoLLM with a motion sensor |
| [`automations/vision_on_motion.yaml`](../automations/vision_on_motion.yaml) | Run camera identification when a camera sees a person or animal |

## Blueprints

Import and use them from **Settings → Automations & Scenes → Blueprints**. The import buttons are in
[`blueprints/README.md`](../blueprints/README.md).

| Blueprint | What it does | Works today? |
|---|---|---|
| **Room-Based Climate** | Sets a thermostat when someone enters or leaves a chosen room | ✅ Yes. Temperature inputs are bounded for °F. |
| **Follow Me Music** | When someone changes rooms, pauses the speaker they left and presses play on the one they entered | ✅ Yes. Needs a room → speaker map that includes both the old and the new room. |
| **Welcome Home Lights** | Lights on when someone arrives | ⚠️ **Not yet.** It waits for the `whollm_person_arrived` event, which is not fired yet. Use the [welcome home recipe](#welcome-home). |
| **Goodbye Routine** | Lights off, doors locked, climate set when someone leaves | ⚠️ **Not yet.** It waits for `whollm_person_left`, which is not fired yet. Use the [leaving home recipe](#leaving-home). |

## Arrival, departure and room-change events (not yet available)

The code contains a notification manager that is designed to fire these events. It is not connected to the running
integration yet, so nothing fires them today. See [Status & roadmap](status.md).

| Event | Planned data |
|---|---|
| `whollm_person_arrived` | `entity_name`, `room`, `confidence`, `timestamp` |
| `whollm_person_left` | `entity_name`, `last_room`, `timestamp` |
| `whollm_room_changed` | `entity_name`, `from_room`, `to_room`, `confidence`, `timestamp` |

Until then, use state triggers on `sensor.<name>_room` as shown above.
