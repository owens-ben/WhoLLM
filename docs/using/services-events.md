# Services & events

This page lists every service (called an "action" in newer Home Assistant versions) that Roomer registers, and every event it fires.

You can try any of these from **Developer Tools → Actions** (called **Services** in older versions). Switch to **YAML mode** and paste an example.

## Services

| Service | What it does |
|---|---|
| [`roomer.identify_person`](#roomeridentify_person) | Takes a camera snapshot and asks a vision model who is in it |
| [`roomer.enable_tracking`](#roomerenable_tracking-roomerdisable_tracking) | Turns on a PTZ camera's auto-tracking switch |
| [`roomer.disable_tracking`](#roomerenable_tracking-roomerdisable_tracking) | Turns off a PTZ camera's auto-tracking switch |
| [`roomer.cleanup_storage`](#roomercleanup_storage) | Trims Roomer's event log file |

!!! note "Results come back as events"
    None of these services return response data to the caller. `identify_person` and `cleanup_storage` report their results by firing an event, listed under [Events](#events).

!!! info "Only some services have a form in the UI"
    Only `roomer.cleanup_storage` has field descriptions in the UI. The other three still work, but Developer Tools shows them without a form, so use YAML mode.

### `roomer.identify_person`

Grabs a snapshot from a camera and sends it to a vision model on your Ollama server, together with the names of the people (or pets) you configured, and asks who it is. The answer appears on [`sensor.vision_last_identification`](entities.md) and in the [`roomer_vision_identification`](#roomer_vision_identification) event.

| Field | Required | Type | Default | Description |
|---|---|---|---|---|
| `camera_entity_id` | yes | entity ID | | The `camera.*` entity to take a snapshot from |
| `detection_type` | no | `person` or `animal` | `person` | Whether to look for a person (matched against your configured people) or an animal (matched against your pets) |

```yaml
service: roomer.identify_person
data:
  camera_entity_id: camera.living_room
  detection_type: person
```

How it works:

- The request goes to the **same server URL** you entered when setting up Roomer, so that must be an Ollama server.
- The vision model defaults to `moondream`. Pull it first with `ollama pull moondream`.
- The vision request times out after 60 seconds.
- Vision results do **not** currently change anyone's room sensor; they only update the vision sensor and fire the event.

### `roomer.enable_tracking` / `roomer.disable_tracking`

Turn a PTZ camera's auto-tracking on or off. Roomer simply calls `switch.turn_on` / `switch.turn_off` on an entity named `switch.<camera_name>_auto_tracking`, which is how the Reolink integration names that switch.

| Field | Required | Type | Description |
|---|---|---|---|
| `camera_name` | yes | text | The middle part of the switch's entity ID. For `switch.e1_zoom_auto_tracking`, use `e1_zoom`. |

```yaml
service: roomer.enable_tracking
data:
  camera_name: e1_zoom
```

If your camera's switch has a different name, just call `switch.turn_on` on it directly.

### `roomer.cleanup_storage`

Roomer keeps a log of its decisions in `/config/roomer_events.jsonl`. It trims this file automatically when Home Assistant starts and every 24 hours after that, removing entries older than 30 days and keeping the file under 100 MB. Call this service to clean up right away or with tighter limits.

| Field | Required | Type | Range | Description |
|---|---|---|---|---|
| `retention_days` | no | integer | 1–365 | Delete entries older than this many days |
| `max_file_size_mb` | no | number | 1–1000 | Trim the oldest entries until the file is under this size |

```yaml
service: roomer.cleanup_storage
data:
  retention_days: 7
  max_file_size_mb: 20
```

!!! note
    The values you pass here stay in effect for the automatic daily cleanup too, until Home Assistant restarts.

When it finishes, it fires [`roomer_storage_cleanup`](#roomer_storage_cleanup).

## Events

You can watch any of these live in **Developer Tools → Events**: type the event name under "Listen to events" and click **Start listening**.

| Event | Fired when |
|---|---|
| [`roomer_vision_identification`](#roomer_vision_identification) | `roomer.identify_person` finishes, whether it succeeded or failed |
| [`roomer_storage_cleanup`](#roomer_storage_cleanup) | `roomer.cleanup_storage` finishes |
| [`roomer_person_arrived`, `roomer_person_left`, `roomer_room_changed`](#arrival-departure-and-room-change-events-not-fired-yet) | Not fired yet. See below. |

### `roomer_vision_identification`

| Field | Type | Description |
|---|---|---|
| `camera` | text | The camera entity used |
| `detection_type` | text | `person` or `animal` |
| `success` | boolean | `true` if the vision model answered, `false` if something failed |
| `identified` | text | The configured person or pet name the model matched, otherwise what it said (for example `unknown` or `cat`). Present only when `success` is `true`. |
| `confidence` | text | `high`, `medium` or `low`, as the model reported it. Present only when `success` is `true`. |
| `description` | text | The model's short description of the image. Present only when `success` is `true`. |
| `raw_response` | text | The model's full reply. Present only when `success` is `true`. |
| `error` | text | What went wrong, such as `Failed to get camera snapshot` or `Vision model timeout`. Present only when `success` is `false`. |

Example: announce who was seen at the front door.

```yaml
alias: "Announce who is at the door"
trigger:
  - platform: event
    event_type: roomer_vision_identification
    event_data:
      camera: camera.front_door
      success: true
condition:
  - condition: template
    value_template: "{{ trigger.event.data.confidence in ['high', 'medium'] }}"
action:
  - service: notify.mobile_app_your_phone
    data:
      message: "{{ trigger.event.data.identified }} is at the front door"
```

### `roomer_storage_cleanup`

| Field | Type | Description |
|---|---|---|
| `deleted_by_time` | integer | Entries removed for being older than the retention period |
| `deleted_by_size` | integer | Entries removed to get under the size limit |
| `total_deleted` | integer | Total entries removed |
| `total_kept` | integer | Entries left in the file |
| `final_size_mb` | number | File size after cleanup, in MB |

### Arrival, departure and room-change events (not fired yet)

!!! warning "Not yet available"
    These events are not fired in the current version.

    The code defines the three events below, and older documentation and two of the bundled blueprints use them, but the part of Roomer that fires them is not connected yet. Automations that trigger on them **will not run**.

    Until then, trigger on the room sensor instead:

    - **Arrived home:** `sensor.<name>_room` changes `from: "away"`
    - **Left home:** `sensor.<name>_room` changes `to: "away"`
    - **Changed room:** any state change of `sensor.<name>_room`, or a `binary_sensor.<name>_in_<room>` turning `on`

    [Automations](automations.md) has working examples.

For reference, these are the fields the events are designed to carry:

| Event | Fields |
|---|---|
| `roomer_person_arrived` | `entity_name` (the person's configured name, such as `Alice`), `room`, `confidence`, `timestamp` |
| `roomer_person_left` | `entity_name`, `last_room`, `timestamp` |
| `roomer_room_changed` | `entity_name`, `from_room`, `to_room`, `confidence`, `timestamp` |

`timestamp` is an ISO-format local time string, and `confidence` is a number from 0.0 to 1.0.

## Services you may see mentioned elsewhere

!!! warning "Not yet available"
    The services named below are not registered, so you can't use them.

Older documentation, and the service descriptions bundled with the integration, mention `roomer.refresh`, `roomer.correct_room`, `roomer.get_history`, `roomer.clear_history`, `roomer.get_patterns` and `roomer.request_vision_update`. **None of these are registered** in the current version, so calling them fails with a "service not found" error.

To force an update, use Home Assistant's built-in `homeassistant.update_entity` action on one of Roomer's entities:

```yaml
service: homeassistant.update_entity
target:
  entity_id: sensor.alice_room
```

This refreshes all of Roomer's people and pets, not just the one you name.
