# First setup

Roomer is set up through a short wizard with five screens. In Home Assistant,
this kind of wizard is called a **config flow**. Have your Ollama address and the
names of the people in your home ready.

## Before you start

The wizard asks for some of this and **Configure** asks for the rest, so it saves
time to look it all up first. You'll find your entities in **Settings → Devices
& services → Entities** (use the search box).

| What | Where to find it | Example |
|---|---|---|
| Ollama address | The IP address of the computer running Ollama, with port 11434 | `http://192.168.1.50:11434` |
| People and pets | Everyone you want tracked. People and pets are entered separately | People: `Alice, Bob`. Pets: `Miso` |
| Rooms | The rooms you want an answer for | `office, living_room, bedroom, kitchen` |
| Sensors in each room | Motion sensors, the room's lights, a TV or speaker, a computer, door sensors | `binary_sensor.office_motion` |
| Each person's phone | The `device_tracker` entity the Home Assistant Companion app creates for their phone. It tells Roomer who is home | `device_tracker.alice_phone` |
| Each person's own computer, if they have one | A switch or sensor that shows it's on. This is the best clue you can give without extra hardware | `switch.alice_pc` |

You don't need special room-tracking hardware. It helps, and it's covered in
[Map your rooms](map-rooms.md#devices-that-report-a-room), but without it Roomer
works from the sensors you already have and what it learns over time. See
[Accuracy & limits](../using/accuracy.md) for what to expect.

## Start the wizard

To start:

1. Go to **Settings → Devices & services**.
2. Select **Add integration** and choose **Roomer**.

## Screen 1: Roomer Setup

This screen tells Roomer where your language model server is.

| Field | What to enter | Default |
|---|---|---|
| **Provider** | `ollama` for most people. `crewai` is only for people already running a CrewAI API service. | `ollama` |
| **Provider URL** | The address of your Ollama server, including the port. | `http://localhost:11434` |
| **Poll Interval** | How often Roomer makes a new guess, from 5 to 300 seconds. | 10 seconds |

!!! example "Screenshot needed: config-flow-provider.png"
    The "Roomer Setup" screen showing the Provider dropdown, Provider URL field and Poll Interval slider.

When you select **Submit**, Roomer tries to connect to the server. If it can't,
you'll see:

> Unable to connect to the LLM provider. Check the URL and try again.

If that happens:

- Check that Ollama is running on that machine.
- If Ollama runs on a different machine from Home Assistant, don't use
  `localhost`. Use that machine's IP address instead, such as
  `http://192.168.1.50:11434`.
- Make sure Ollama accepts connections from your network. See the warning in
  [What you'll need](requirements.md#a-language-model-server-ollama).

!!! tip "Not sure about the poll interval?"
    Keep the default to start with. Each poll asks the model once per person at
    home and once per pet. If your Ollama machine is slow, choose 30 or 60
    seconds. You can change this later.

## Screen 2: Model Selection

Pick the model Roomer should use. The list shows the models already downloaded
on your Ollama server. You can also type a model name.

- **Model**: choose `llama3.2` if you followed [What you'll need](requirements.md).

The first model in the list is pre-selected, and it might not be the one you
want, so check it before you continue. If Roomer couldn't read the list, it
shows `llama3.2`.

!!! example "Screenshot needed: config-flow-model.png"
    The "Model Selection" screen with the Model dropdown open, showing installed Ollama models.

## Screen 3: Room Configuration

Choose the rooms Roomer should place people in.

| Field | What to enter |
|---|---|
| **Rooms** | Tick the rooms you have. The choices are `office`, `living_room`, `bedroom`, `kitchen`, `bathroom` and `entry`, and all six start ticked. |
| **Custom Rooms (comma-separated)** | Any other rooms, for example `garage, basement, kids room`. |

Custom names are tidied up automatically. They're made lowercase, spaces become
underscores, and symbols are removed, so `Kids Room!` becomes `kids_room`.

You need at least one room. If you untick everything and leave the custom box
empty, you'll see "Please select at least one room to track."

!!! tip "Untick rooms you don't need"
    Roomer creates an on/off sensor for every person in every room, so keep the
    list to rooms you'll actually use. More rooms also give the model more ways
    to get it wrong.

You don't need to add `away` or `unknown`. Roomer always has those.

!!! example "Screenshot needed: config-flow-rooms.png"
    The "Room Configuration" screen with a few rooms ticked and "garage" typed into the Custom Rooms field.

## Screen 4: People & Pets

Type the names of everyone you want to track.

| Field | What to enter |
|---|---|
| **People (comma-separated)** | For example `Alice, Bob` |
| **Pets (comma-separated)** | Optional. For example `Miso` |

You need at least one person or pet, otherwise you'll see "Please configure at
least one person or pet to track."

Choose names carefully, because they become entity IDs. `Alice` becomes
`sensor.alice_room`, and `Mary Jane` becomes `sensor.mary_jane_room`.

!!! example "Screenshot needed: config-flow-people.png"
    The "People & Pets" screen with "Alice, Bob" in People and "Miso" in Pets.

## Screen 5: Room Entity Mapping

Now tell Roomer which sensors and devices belong to each room. There is one
picker per room you chose. The pickers may show technical labels such as
`entities_office`, where the part after `entities_` is the room.

You can pick entities from these types: binary sensors, sensors, lights,
switches, media players, device trackers and cameras.

For each room, add things that show someone is there. Good examples are motion
sensors, the room's lights, a TV or speaker, a computer, door sensors, and
camera person-detection sensors.

!!! example "Screenshot needed: config-flow-room-entities.png"
    The "Room Entity Mapping" screen with entities selected for the office and living room pickers.

[Map your rooms](map-rooms.md) explains what to put where, with a full example.
You can leave every picker empty and select **Submit**, but Roomer will have
almost nothing to go on until you add some.

!!! tip "Do your room mapping here"
    It's best to map your rooms on this screen rather than later. Because of a
    current limitation in the settings screens (see
    [Map your rooms](map-rooms.md#before-you-start)), the mapping you make here is
    the most reliable.

## When you're done

Roomer creates an entry called **Roomer (ollama)** or **Roomer (crewai)**,
depending on the provider you picked, and makes its first guess straight away. You'll find these new entities:

| Entity | Example | What it shows |
|---|---|---|
| One room sensor per person or pet | `sensor.alice_room` | The current room, `away` or `unknown` |
| One on/off sensor per person or pet, per room | `binary_sensor.alice_in_office` | On when that person is in that room |
| Vision result | `sensor.vision_last_identification` | The last camera identification, if you use it |

!!! example "Screenshot needed: setup-complete-entities.png"
    The Roomer integration page listing its new entities, including sensor.alice_room and several binary_sensor.alice_in_* entities.

Expect early guesses to be rough. Roomer hasn't been told which devices belong to
whom yet, and it hasn't learned any habits. The next step fixes the first part.

See [Entities](../using/entities.md) for every attribute.

Next: [Map your rooms](map-rooms.md).
