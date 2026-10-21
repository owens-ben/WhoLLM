# Map your rooms

Roomer only looks at entities you've told it about. You give it two kinds of
mapping:

- **Room mapping** says *where* things are. Example: the office motion sensor is
  in the office.
- **Device ownership** says *whose* things are. Example: this PC belongs to Alice.

Room mapping helps Roomer pick a room. Device ownership is what lets it tell
people apart.

## Before you start

You did your first room mapping in the last screen of [First setup](first-setup.md).
To change mappings later, or to assign devices to people, use Roomer's options:

1. Go to **Settings → Devices & services → Roomer**.
2. Select **Configure**.
3. Choose a section from the menu: **General Settings**, **Room Entity
   Mapping**, **Person Device Ownership** or **Confidence Weights**.

!!! example "Screenshot needed: options-menu.png"
    The "Roomer Options" menu showing the four choices: General Settings, Room Entity Mapping, Person Device Ownership, Confidence Weights.

!!! warning "Current limitations of the options screens (alpha)"
    - **Changes apply after a reload.** After saving, go to **Settings →
      Devices & services → Roomer**, open the **⋮** menu and select **Reload**,
      or restart Home Assistant.
    - **Only the most recently saved options section takes effect.** Saving one
      section replaces anything you saved in a different section. Everything
      else goes back to what you chose in the setup wizard. The forms also show
      your setup-wizard values, not your last save.

    Until this is fixed, the most reliable approach is:

    1. Do your **room mapping in the setup wizard**.
    2. Use **Person Device Ownership** as the one options section you save.
    3. To change rooms or room mappings later, remove Roomer and add it again,
       then redo Person Device Ownership.

## How Roomer reads an entity

When you map an entity to a room, Roomer decides what **type** of clue it is
based on words in its entity ID. Roomer checks these rules in order and uses the
first one that matches:

| If the entity ID contains… | Type | Default weight |
|---|---|---|
| `motion` or `occupancy` | motion | 60% |
| `media_player` or `tv` | media | 80% |
| `light.` (it's a light) | light | 35% |
| `pc`, `computer` or `desktop` | computer | 85% |
| `camera`, `person` or `animal` | camera | 95% |
| `door`, `window` or `contact` | door | 40% |
| `temperature` or `humidity` | climate | 20% |
| `device_tracker` | presence | 70% |
| anything else | appliance | 50% |

The weight is how much the clue counts when Roomer
[combines the evidence](how-it-works.md#4-combine-the-evidence).

An entity counts as a clue only while it's **active**, which means its state is
`on`, `home`, `detected`, `playing`, `open` or `occupied`. A media player also
counts when it's `paused`, at a lower weight.

!!! tip "Name things so Roomer guesses right"
    The type is worked out once, when you save the mapping. If a name gives the
    wrong type (for example, `switch.tv_room_lamp` would count as media), rename
    the entity ID in Home Assistant first, then map it again. Use the
    person-detection **binary sensor** from your camera system (like
    `binary_sensor.kitchen_person`), not the `camera.` entity itself. A camera
    entity's state is never "active".

## Room Entity Mapping

This screen has one picker per room. It accepts binary sensors, sensors, lights,
switches, media players, device trackers and cameras. For each room, add the
things that show someone is in that room right now.

Good choices:

- motion and occupancy sensors
- the room's main lights
- TVs, speakers and consoles
- computers and their "in use" sensors
- door sensors for rooms with one main door
- camera person-detection sensors

!!! warning "Don't map phones to rooms"
    A phone's `device_tracker` says `home` wherever you are in the house. If you
    map it to a room, Roomer treats that room as active whenever anyone's phone
    is home. Assign phones under **Person Device Ownership** instead.

Remember that room-mapped entities count **equally for everyone at home**. The
office motion sensor doesn't know who moved.

!!! example "Screenshot needed: options-room-entities.png"
    The "Room Entity Mapping" options screen with several entities selected for the office and living room.

## Person Device Ownership

This screen has one picker per person and pet. It accepts device trackers,
binary sensors, switches and sensors. Add the devices that belong to that person.

Roomer uses them in two ways:

- **Device trackers decide home or away.** If all of a person's
  `device_tracker` entities say they're away, Roomer marks them `away` straight
  away, at 95% confidence. If any says `home`, Roomer works out which room
  they're in.
- **Other devices point to that person.** When an owned device is active, it
  counts as a strong clue (computer weight, 85%) for that person only, in the
  device's room. **The device must also be mapped to a room** under Room Entity
  Mapping, or Roomer won't know where it is and will ignore it.

Pets appear on this screen too, but devices assigned to pets aren't used yet.

!!! example "Screenshot needed: options-person-devices.png"
    The "Person Device Ownership" options screen with a phone tracker and a PC switch assigned to Alice, and a phone tracker and gaming PC assigned to Bob.

## Worked example

Alice and Bob live with their cat, Miso. Their rooms are `office`,
`living_room`, `bedroom` and `kitchen`.

### Room mapping

| Room | Entity | Type Roomer picks |
|---|---|---|
| office | `binary_sensor.office_motion` | motion |
| office | `light.office` | light |
| office | `switch.alice_pc` | computer |
| living_room | `binary_sensor.living_room_motion` | motion |
| living_room | `media_player.living_room_tv` | media |
| living_room | `switch.bob_gaming_pc` | computer |
| living_room | `binary_sensor.living_room_person` | camera |
| bedroom | `light.bedroom` | light |
| bedroom | `binary_sensor.bedroom_door` | door |
| kitchen | `binary_sensor.kitchen_motion` | motion |
| kitchen | `light.kitchen` | light |

### Device ownership

| Person | Devices |
|---|---|
| Alice | `device_tracker.alice_phone`, `switch.alice_pc` |
| Bob | `device_tracker.bob_phone`, `switch.bob_gaming_pc` |
| Miso | *(none)* |

Both PCs appear in **both** tables. The room mapping says where each PC is, and
the ownership mapping says whose it is.

### What Roomer can work out

| Situation | Likely result | Why |
|---|---|---|
| Alice at her desk, Bob out | Alice: `office`. Bob: `away` | Bob's phone says away. Alice's own PC is on in the office |
| Alice at her desk, Bob gaming | Alice: `office`. Bob: `living_room` | Each person's own PC gives them a clue the other doesn't get |
| Both cooking, PCs off | Both: `kitchen` | Kitchen motion and light count for both, which happens to be right |
| Alice napping in the dark bedroom, Bob in the kitchen | Both may show `kitchen` | Nothing ties either person to a room. Only anonymous sensors are active |
| Miso anywhere | A rough guess | Pets only get anonymous clues |

The fourth row shows Roomer's main limit. When the only active sensors are
shared ones, it can't tell who's who. Adding one personal signal per person,
like room-level phone tracking, fixes most of these cases. See
[What you'll need](requirements.md#how-good-are-your-signals) and
[Accuracy & limits](../using/accuracy.md).

## Check the result

Open `sensor.alice_room` in **Settings → Devices & services → Entities**, or in
**Developer tools → States**. Look at these attributes:

- `confidence`: from 0 to 1. For example, `0.82` means 82%.
- `indicators`: the clues Roomer used, such as `computer: alice_pc`.

If a device you expected isn't listed, check that it's mapped to a room and is
currently active.

!!! example "Screenshot needed: sensor-attributes-indicators.png"
    Developer tools → States filtered to sensor.alice_room, showing the confidence and indicators attributes.

## General Settings

| Setting | What it does | Default |
|---|---|---|
| **Poll Interval (seconds)** | How often Roomer guesses again (5–300) | Your setup value, or 10 |
| **LLM Request Timeout (seconds)** | How long to wait for the model before using the fallback rules (5–120) | 30 |
| **Enable Pattern Learning** | Learn habits from confident guesses | On |

## Confidence Weights

This screen has a slider (0–100%) for each type of clue: camera, computer,
media, motion, presence, appliance, light, door, the language model's reasoning
(`llm_reasoning`) and learned habits (`habit`). The labels may show technical
names like `weight_camera`. Raise a weight to make that type count for more.
Most people can leave these alone.

Next: learn about the [entities Roomer creates](../using/entities.md), or start
building [automations](../using/automations.md).
