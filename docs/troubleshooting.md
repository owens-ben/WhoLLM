# Troubleshooting

[← Dashboard](dashboard.md) · [Guide home](README.md) · Next: [Status & roadmap →](status.md)

Find your symptom, check the cause, apply the fix.

## First, turn on debug logs

Add this to `configuration.yaml` and restart:

```yaml
logger:
  default: info
  logs:
    custom_components.whollm: debug
```

Then read **Settings → System → Logs** and search for `whollm`. At debug level you see each LLM query, its raw answer
and any fallback.

## Quick triage

| Symptom | Likely cause | Jump to |
|---|---|---|
| Not in the *Add Integration* list | Wrong folder, or no restart | [↓](#whollm-is-not-in-the-add-integration-list) |
| Wizard cannot connect | Ollama is not reachable from Home Assistant | [↓](#the-wizard-says-unable-to-connect-to-the-llm-provider) |
| Everything is at low confidence | Too few mapped sensors | [↓](#confidence-is-always-low) |
| Someone is in the wrong room | Shared signals, wrong mapping | [↓](#the-wrong-person-gets-the-room) |
| `raw_response` shows an error message | The provider is failing, fallback is in use | [↓](#raw_response-shows-an-error-or-answers-feel-random) |
| Stuck on `away` | A device tracker says not home | [↓](#someone-is-stuck-on-away) |
| Settings do not stick | Reload needed, or the options-page issue | [↓](#settings-do-not-take-effect-or-disappear) |
| Habits are missing | They are not persisted yet | [↓](#habit-hints-never-appear-or-disappear-after-a-restart) |
| Camera identification does nothing | Model, snapshot or service call | [↓](#camera-identification-does-nothing) |
| Dashboard or card is empty | Entity names or setup | [↓](#dashboard-and-card-problems) |

## WhoLLM is not in the Add Integration list

- The folder must be exactly `config/custom_components/whollm/` and contain `manifest.json` directly (not one folder
  deeper).
- Restart Home Assistant after copying or installing, then hard-refresh the browser.
- Check the manifest parses: `python3 -m json.tool custom_components/whollm/manifest.json`.
- Search the logs for `whollm` import errors.

## The wizard says "Unable to connect to the LLM provider"

The URL is tested **from Home Assistant's point of view**, so `localhost` means the Home Assistant machine or container.

1. From the Home Assistant host (or the Terminal/SSH add-on), run `curl http://<ollama-host>:11434/api/tags`. It should
   print JSON.
2. If Ollama is on another machine, it must listen on the network: start it with `OLLAMA_HOST=0.0.0.0`.
3. Check firewalls and that you used the machine's IP address, not `localhost`.
4. Using the `crewai` provider? `GET <url>/` must return `{"status": "ok"}`.

## Confidence is always low

**This is usually expected.** Confidence is normalised against what your sensors can prove, and an LLM answer alone is
only about 13 %. See [how confidence works](how-it-works.md#signals-weights-and-confidence).

1. Open the sensor in **Developer Tools → States** and look at `indicators`. If it is mostly empty, none of your mapped
   entities were active, so there is little evidence.
2. Map more entities to rooms: motion sensors, lights, media players, PC power sensors, camera person detection.
3. Check each one is [guessed as the right kind](configuration.md#hint-types-are-guessed-from-the-entity-id). A TV
   light named `light.tv_backlight` is treated as media, for instance.
4. Remember only entities in an *active* state (`on`, `home`, `playing`, …) count.
5. Optionally raise the weights of signals you trust under **Configure → Confidence Weights**.

## The wrong person gets the room

Mapped entities are **shared** evidence, so one person's PC or motion in a room counts toward everybody. See
[Shared signals](how-it-works.md#shared-signals-in-multi-person-homes).

- Assign each person their own devices under **Person Device Ownership**.
- Use per-person signals (a PC only they use, phone-based presence, camera person recognition) and give them higher
  weights.
- Make sure people who are out have a `device_tracker` assigned so they are `away` and drop out of the competition.

## `raw_response` shows an error, or answers feel random

When the provider fails, WhoLLM uses a simple rule-based guess and puts the **error text** in `raw_response`.

- Is Ollama running? `ollama ps`, and `journalctl -u ollama` on Linux.
- Is the configured model still installed? `ollama list`. A missing model fails every request.
- Is it timing out? Raise **LLM request timeout** (Configure → General) or use a smaller model such as `llama3.2:1b`.
- Is the poll too fast for your hardware? See [Performance](#performance).

## Someone is stuck on `away`

`away` means **every** `device_tracker` assigned to that person reports something other than `home` (any zone such as
`work` counts as away).

- Check the tracker states in **Developer Tools → States**. Phones on Wi-Fi-only tracking often drop to `not_home`
  when they sleep. A more reliable tracker (the companion app, for example) helps.
- Remove a flaky tracker from **Person Device Ownership**, or add a second, reliable one. One tracker saying `home` is
  enough to cancel `away`.

## `unknown`

The state is `unknown` when the LLM did not return a valid room **and** no mapped entity or habit voted. Check
`raw_response` and `indicators`, and map more entities.

## Settings do not take effect, or disappear

- Options are read at startup. After saving under **Configure**, reload the integration (⋮ → **Reload**).
- **Known issue:** each *Configure* page saves only its own values and replaces the other pages' saved values, so the
  last page you saved is the one that sticks. After saving, re-open the other pages and check them. Tracked in
  [Status & roadmap](status.md#known-issues).
- The provider, URL, model, rooms, people and pets cannot be edited after setup. Remove and re-add the integration.

## Habit hints never appear, or disappear after a restart

- Habits are learned only from predictions at **70 % confidence or more**, so they need real sensor evidence.
- Learning must be on (Configure → General Settings).
- Learned habits are currently kept in memory only and start empty after each restart. See
  [Status & roadmap](status.md#known-issues).

## Camera identification does nothing

1. Pull the vision model: `ollama pull moondream` (the default; it is not selectable in the UI yet).
2. Call `whollm.identify_person` from **Developer Tools → Actions** in YAML mode, and read the log. Typical messages:
   *Failed to get camera snapshot* (the camera entity cannot return an image) or *Vision model timeout*
   (it has 60 seconds).
3. Look at `sensor.vision_last_identification` for the result.
4. The standalone dashboard's *Identify Now* button does not work yet. See [Dashboard](dashboard.md#known-limitations).

## The event log is getting big

See [What gets stored on disk](configuration.md#what-gets-stored-on-disk). Run
[`whollm.cleanup_storage`](entities-and-services.md#whollmcleanup_storage) with a smaller `retention_days` or
`max_file_size_mb`, or use a longer poll interval so there are fewer records.

## Dashboard and card problems

| Symptom | Check |
|---|---|
| Web dashboard says "Waiting for LLM Presence data…" | The page only shows sensors whose ID contains `_room` and that have an `entity_type` attribute, which `sensor.<name>_room` does. Confirm those exist in Developer Tools. |
| Web dashboard cannot sign in, or bounces back to sign-in | Did you replace `192.168.0.233:8123` in `dashboard/nginx.conf`? See [Dashboard](dashboard.md#3-the-standalone-web-dashboard). |
| Card says *Custom element doesn't exist: whollm-card* | The resource is missing. Add `/local/whollm-card.js` as a JavaScript module and hard-refresh. |
| Card says *Entity not found* | Use the exact entity ID of the room sensor, for example `sensor.alice_room`. |

## Performance

| Situation | Try |
|---|---|
| Slow predictions or timeouts | A smaller model (`llama3.2:1b`, `phi3:mini`) or a GPU. |
| High CPU load | A longer poll interval (60 s or more). |
| Many people and rooms | A dedicated Ollama machine. One request is made per person per poll, one after another. |
| Rule of thumb | Poll interval ≳ (people + pets) × time for one answer. |

## Reporting a bug

Please include:

- Home Assistant version and install type, and the WhoLLM version (`0.1.0` unless you built from `main`).
- Provider and model.
- The attributes of the affected sensor, copied from Developer Tools → States.
- Debug logs around the problem (see the top of this page).

Open an issue at <https://github.com/owens-ben/WhoLLM/issues>.
