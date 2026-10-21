# What you'll need

Before installing Roomer, check you have three things: a recent Home Assistant, a
language model server, and some sensors that can tell your household apart.

## Home Assistant

You need **Home Assistant 2024.1 or newer**. Any installation type works (Home
Assistant OS, Container, Supervised or Core), as long as you can add custom
integrations.

To install through HACS (the easiest way), you also need
[HACS](https://hacs.xyz/) set up. HACS is the Home Assistant Community Store, an
add-on for installing community integrations.

## A language model server (Ollama)

Roomer uses a language model to weigh up the clues, and **it needs one to run**.
During setup Roomer checks that it can reach the server, and setup won't finish
without it. (If the server goes down later, Roomer keeps working using simpler
built-in rules until it comes back. See [How it works](how-it-works.md#3-ask-the-language-model).)

The default and recommended option is [Ollama](https://ollama.com), which runs
models locally on your own hardware.

1. Install Ollama on a machine on your network. On Linux:

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

    For macOS and Windows, download it from [ollama.com](https://ollama.com/download).

2. Download the text model Roomer uses by default (about 2 GB):

    ```bash
    ollama pull llama3.2
    ```

3. Optional: download a vision model for camera identification:

    ```bash
    ollama pull moondream
    ```

    Roomer's vision feature uses `moondream` by default. You only need it if you
    plan to use the `roomer.identify_person` service (see
    [Services & events](../using/services-events.md)).

4. Check that Ollama is reachable from Home Assistant. From another machine,
    open `http://<ollama-machine-ip>:11434/api/tags` in a browser. You should see
    a short JSON list of your models.

!!! warning "If Ollama isn't on the same machine as Home Assistant"
    The default URL, `http://localhost:11434`, only works when Home Assistant
    can reach Ollama on its own machine. On Home Assistant OS, `localhost` means
    the Home Assistant system itself, so use the IP address of the computer
    running Ollama instead, for example `http://192.168.1.50:11434`. By default
    Ollama only accepts local connections, so you also have to tell it to listen
    on your network. Give that computer a fixed address in your router
    (a "DHCP reservation"), so the address in Roomer doesn't go stale.

    === "macOS (the Ollama app)"

        Open **Terminal**, run this, then quit Ollama from the menu bar and
        open it again:

        ```bash
        launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
        ```

        If macOS asks whether Ollama may accept incoming network connections,
        choose **Allow**. Also stop the Mac from sleeping (**System Settings →
        Energy**, then turn on preventing automatic sleep). A sleeping Mac stops
        answering, and Roomer falls back to its simpler built-in rules until it
        wakes up.

    === "Linux"

        Run `sudo systemctl edit ollama`, add the lines below, save, then run
        `sudo systemctl restart ollama`:

        ```ini
        [Service]
        Environment="OLLAMA_HOST=0.0.0.0:11434"
        ```

    === "Windows"

        Add a user environment variable named `OLLAMA_HOST` with the value
        `0.0.0.0:11434` (search for "environment variables" in the Start menu),
        then quit Ollama from the system tray and start it again.

    Then repeat the check in step 4 above, from another computer.

### How powerful does it need to be?

Roomer asks the model once per person at home, and once per pet, on every poll
(every 10 seconds by default). A small model like `llama3.2` runs on a normal
desktop CPU, but answers come faster on a machine with a GPU. If guesses lag or
time out, raise the poll interval or try a smaller model like `llama3.2:1b`.

### Other providers

The setup screen also offers **crewai**, which connects to a self-hosted CrewAI
API service (usually at `http://localhost:8502`). This is for advanced users who
already run one. The code also includes OpenAI and Anthropic providers, but they
aren't available in setup yet.

## Sensors that tell people apart

This is the part that decides how well Roomer works. Roomer can only tell
people apart if at least some of your sensors are tied to **one specific
person**.

### How good are your signals?

Signals ranked by how well they identify *who* is in a room:

| Strength | Signal | Example entity | What it tells Roomer |
|---|---|---|---|
| :material-signal-cellular-3: **Strong** | A personal computer used by one person | `switch.alice_pc`, `binary_sensor.alice_desktop_active` | "Alice is at her desk", as long as nobody else uses it |
| :material-signal-cellular-3: **Strong** | A person's phone or watch located to a room (Bermuda, ESPresense, room-assistant and similar) | `sensor.alice_phone_area` | "Alice is in the office". Roomer reads the room straight from the sensor (see tip below) |
| :material-signal-cellular-2: **Medium** | Phone on home Wi-Fi or the Companion app (`device_tracker` home/away) | `device_tracker.alice_phone` | "Alice is home" or "Alice is out". It rules people out but doesn't give a room |
| :material-signal-cellular-2: **Medium** | A media device mostly used by one person | `media_player.bob_bedroom_tv`, `switch.bob_gaming_pc` | "Probably Bob", in that room |
| :material-signal-cellular-1: **Anonymous** | Camera person detection (Frigate, Reolink, UniFi and similar) | `binary_sensor.kitchen_person` | "*Someone* is in the kitchen". Reliable about the room, but not about who |
| :material-signal-cellular-1: **Anonymous** | Motion and occupancy sensors | `binary_sensor.office_motion` | "Someone moved in the office" |
| :material-signal-cellular-1: **Anonymous** | Lights, doors, shared TVs | `light.office`, `binary_sensor.back_door` | "Something happened in this room" |

!!! warning "Anonymous sensors alone can't tell people apart"
    Motion, lights, doors and camera person-detection count **the same for
    everyone who's home**. If those are all you have and two or more people are
    home, Roomer can't reliably tell them apart. It falls back on the language
    model's reasoning and learned habits, which are educated guesses at best.
    With only one person home, or with one strong personal signal per person,
    results get much better. See [Accuracy & limits](../using/accuracy.md).

!!! tip "Room-level phone tracking"
    Tools like Bermuda, ESPresense and room-assistant report which room a phone,
    watch or tag is in, usually as a sensor whose state is the room or area name
    (for example, `sensor.alice_phone_area` = `Office`). Assign that sensor to
    its owner under **Person Device Ownership** and Roomer puts them in that room
    directly. No helper entity is needed, and it works for a pet's tag too. The
    room names only have to match your Roomer rooms ignoring capitals, spaces and
    dashes. See [Devices that report a room](map-rooms.md#devices-that-report-a-room).

### About camera identification

Roomer has an optional vision feature. The `roomer.identify_person` service grabs
a camera snapshot and asks a vision model who's in it. The answer appears in
`sensor.vision_last_identification`. Today it only runs when you call the
service, and **its answer isn't fed into the room guesses**. For room tracking,
camera person-detection sensors count as anonymous clues, as in the table above.

Next: [Install Roomer](install.md).
