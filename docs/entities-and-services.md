# Entities & services

[← How it works](how-it-works.md) · [Guide home](README.md) · Next: [Automations →](automations.md)

The reference for everything WhoLLM creates and everything you can call.

## Entities

### How names are built

| You entered | Entity | Example |
|---|---|---|
| Person or pet `Alice` | `sensor.alice_room` | `sensor.alice_room` |
| Room `living_room` | `binary_sensor.alice_in_living_room` | one per person (or pet) × room |
| *(always)* | `sensor.vision_last_identification` | camera identification result |

Names are lower-cased and spaces become underscores, so `Mary Jane` becomes `sensor.mary_jane_room`. People and pets use
the same pattern; pets get a paw icon.

### Room sensor: `sensor.<name>_room`

The answer to "where is this person right now?"

**State**: a room name you configured (`office`, `living_room`, …), `away`, or `unknown`.

| Attribute | Type | Meaning |
|---|---|---|
| `confidence` | number, 0 to 1 | How much evidence supports the answer. See [how confidence works](how-it-works.md#signals-weights-and-confidence). |
| `indicators` | list of text | The active evidence, for example `motion: office_motion`. The last entry is a one-line summary of the strongest evidence. |
| `raw_response` | text | What the provider returned, such as `ROOM: office`. When the fallback was used it holds the error text instead. For `away` it says the device tracker indicated not home. |
| `entity_type` | `person` or `pet` | |

`away` comes from the device-tracker check. It needs a `device_tracker` assigned to the person under
[Person Device Ownership](configuration.md#person-device-ownership).

> [!TIP]
> Use the `confidence` attribute as a guard in automations so that a shaky guess does not switch things on.
> Example: `{{ state_attr('sensor.alice_room', 'confidence') | float(0) >= 0.6 }}`.

### Room binary sensors: `binary_sensor.<name>_in_<room>`

`on` while that person or pet is in that room. Device class is *occupancy*.

| Attribute | Meaning |
|---|---|
| `confidence` | The prediction's confidence if the person is in this room, otherwise `0.0`. |
| `current_room` | The room the person is actually in. |
| `entity_type` | `person` or `pet`. |

There is one per configured room. `away` and `unknown` are not rooms, so there is no `…_in_away` sensor (unless you
added a custom room named `away`). To react to someone leaving, trigger on `sensor.<name>_room` becoming `away`.

### Vision sensor: `sensor.vision_last_identification`

**State**: the last person or pet identified by [`whollm.identify_person`](#whollmidentify_person), or `none` before
the first run.

Attributes: `confidence` (`high`, `medium` or `low`), `description`, `camera`, `detection_type`, `timestamp`,
`raw_response`.

## Events

| Event | Fired when | Data |
|---|---|---|
| `whollm_vision_identification` | `whollm.identify_person` finishes | `camera`, `detection_type`, `identified`, `confidence`, `description`, `raw_response`, `success` (and `error` on failure) |
| `whollm_storage_cleanup` | `whollm.cleanup_storage` finishes | `deleted_by_time`, `deleted_by_size`, `total_deleted`, `total_kept`, `final_size_mb` |

Arrival, departure and room-change events exist in the code but are **not fired yet**. See
[Automations → events](automations.md#arrival-departure-and-room-change-events-not-yet-available).

## Services

### `whollm.identify_person`

Take a snapshot from a camera and ask an Ollama vision model who or what is in it.

```yaml
service: whollm.identify_person
data:
  camera_entity_id: camera.living_room
  detection_type: person        # or "animal"
```

| Field | Required | Notes |
|---|---|---|
| `camera_entity_id` | yes | Any camera entity that can return a still image. |
| `detection_type` | no | `person` (default) or `animal`. |

How it works: the snapshot is shrunk to at most 512 × 384 JPEG (if Pillow is available), then sent to the vision model
together with the names of the people and pets you configured. It gets 60 seconds to answer.

The result lands on [`sensor.vision_last_identification`](#vision-sensor-sensorvision_last_identification) and in the
`whollm_vision_identification` event. It does **not** currently change anyone's room.

Requirements: a vision model pulled in Ollama (the default is `moondream`, which is not yet selectable in the UI),
and Ollama reachable at the URL you configured. A ready-made automation is in
[`automations/vision_on_motion.yaml`](../automations/vision_on_motion.yaml).

### `whollm.enable_tracking` / `whollm.disable_tracking`

Turn a camera's auto-tracking switch on or off.

```yaml
service: whollm.enable_tracking
data:
  camera_name: e1_zoom          # controls switch.e1_zoom_auto_tracking
```

WhoLLM simply calls `switch.turn_on` or `switch.turn_off` on `switch.<camera_name>_auto_tracking`, so your camera
integration must expose a switch with exactly that entity ID.

### `whollm.cleanup_storage`

Trim the event log now.

```yaml
service: whollm.cleanup_storage
data:
  retention_days: 14            # optional, 1 to 365
  max_file_size_mb: 50          # optional, 1 to 1000
```

Removes entries older than the retention period, then the oldest entries until the file fits the size limit. Without
fields, the current limits (30 days, 100 MB) apply. Values you pass stay in effect until Home Assistant restarts.
This also runs automatically at startup and every 24 hours.

> [!NOTE]
> `identify_person`, `enable_tracking` and `disable_tracking` have no field descriptions in the UI yet. In
> **Developer Tools → Actions**, switch to **YAML mode** and paste the examples above.

### Refreshing on demand

There is no dedicated refresh service yet. Home Assistant's built-in service does the same job for any
coordinator-driven entity:

```yaml
service: homeassistant.update_entity
target:
  entity_id: sensor.alice_room
```

## Not available yet

`whollm.refresh`, `whollm.correct_room`, `whollm.get_history`, `whollm.clear_history` and `whollm.get_patterns` are
written and tested in the code but not registered with Home Assistant, so they do not appear. See
[Status & roadmap](status.md).
