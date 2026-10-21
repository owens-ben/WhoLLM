# Migrating from WhoLLM

Roomer used to be called **WhoLLM**. It's the same integration under a new name,
but its Home Assistant **domain** (the internal name Home Assistant uses for an
integration) changed from `whollm` to `roomer`. To Home Assistant, Roomer is a
brand-new integration, so nothing carries over automatically.

Set aside 15–30 minutes. You'll redo the setup and update anything that uses the
old names.

## Before you start

Write down your current settings so you can enter them again:

- your Ollama URL, model and poll interval
- your rooms, people and pets
- which entities you mapped to each room, and which devices you assigned to each
  person

Open WhoLLM's **Configure** screens and take screenshots or notes.

## Step by step

1. **Remove the old integration.** Go to **Settings → Devices & services →
   WhoLLM**, open the **⋮** menu and select **Delete**. Do this first so the old
   entities are removed and Roomer can reuse their entity IDs.
2. **Delete the old files.** Remove the `config/custom_components/whollm`
   folder. If you installed through HACS, remove WhoLLM there too, and remove
   its custom repository entry.
3. **Install Roomer** with HACS or by hand, as described in
   [Install](getting-started/install.md), then restart Home Assistant.
4. **Add Roomer** under **Settings → Devices & services → Add integration →
   Roomer**, and go through [First setup](getting-started/first-setup.md) using
   your notes. Then redo device ownership as described in
   [Map your rooms](getting-started/map-rooms.md).
5. **Update everything that uses the old names**, as listed below.

## What changed

### Services

Replace `whollm.` with `roomer.` in every service call:

| Old | New |
|---|---|
| `whollm.identify_person` | `roomer.identify_person` |
| `whollm.enable_tracking` | `roomer.enable_tracking` |
| `whollm.disable_tracking` | `roomer.disable_tracking` |
| `whollm.cleanup_storage` | `roomer.cleanup_storage` |

### Events

Rename any event triggers in your automations:

| Old | New |
|---|---|
| `whollm_room_changed` | `roomer_room_changed` |
| `whollm_person_arrived` | `roomer_person_arrived` |
| `whollm_person_left` | `roomer_person_left` |
| `whollm_vision_identification` | `roomer_vision_identification` |
| `whollm_storage_cleanup` | `roomer_storage_cleanup` |

!!! info
    In the current release, `roomer_room_changed`, `roomer_person_arrived` and
    `roomer_person_left` are defined but not fired yet. Until they are, trigger
    automations on changes to `sensor.<name>_room` or
    `binary_sensor.<name>_in_<room>` instead. See
    [Services & events](using/services-events.md) and
    [Automations](using/automations.md).

### Entities

Entity IDs still come from the names you enter, so they look the same as
before:

- `sensor.alice_room`
- `binary_sensor.alice_in_office`
- `sensor.vision_last_identification`

They are new entities, though, with new internal IDs. If an old WhoLLM entity
was still around when you added Roomer, the new one gets a `_2` suffix, like
`sensor.alice_room_2`. To fix that:

1. Go to **Settings → Devices & services → Entities** and search for the name.
2. Delete the old, unavailable entity.
3. Open the new `_2` entity and rename its entity ID to drop the suffix.

Your dashboards and automations will then work without changes.

### Logging

If you turned on debug logging in `configuration.yaml`, update the logger name:

```yaml
logger:
  logs:
    custom_components.roomer: debug   # was custom_components.whollm
```

### Also check

Search your automations, scripts, blueprints and dashboards for `whollm`.
**Settings → Automations & scenes** has a search box, and the YAML files in your
config folder can be searched with any text editor.

## Learned data starts fresh

Roomer doesn't read WhoLLM's saved files. Habits, history and the event log start
empty and build up again as Roomer runs. Roomer saves its own files in your
config folder (`roomer_events.jsonl`, for example). You can delete the old
`whollm_*` files from that folder once you're happy with the switch.
