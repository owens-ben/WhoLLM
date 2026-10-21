# Status & roadmap

[← Troubleshooting](troubleshooting.md) · [Guide home](README.md) · Next: [Development →](development.md)

WhoLLM is **alpha (version 0.1.0)**. The core works, and the repository also holds finished code that is not
connected to the running integration yet. This page says which is which, so you are not surprised.

**Legend:** ✅ works · ⚠️ works with caveats · 🚧 code exists but is not connected · 🗺️ not started

## What works today

### Detection

| Feature | | Notes |
|---|:-:|---|
| Room per person and pet using Ollama | ✅ | |
| Weighted evidence fusion and confidence | ✅ | See [how it works](how-it-works.md). |
| Away detection from `device_tracker` | ✅ | People only. |
| Rule-based fallback when the LLM fails | ✅ | |
| Per-room binary sensors | ✅ | |
| Setup wizard | ✅ | |
| Options (entity mapping, devices, weights, timing) | ⚠️ | [Pages overwrite each other](#known-issues). Reload needed. |
| Habit learning | ⚠️ | In memory only, [not saved](#known-issues). |
| Event log with retention | ✅ | `/config/whollm_events.jsonl` |
| CrewAI provider | ⚠️ | Experimental. Needs your own compatible API. |
| OpenAI and Anthropic providers | 🚧 | Written, but deliberately not offered in the wizard (no API key step). |

### Cameras

| Feature | | Notes |
|---|:-:|---|
| `whollm.identify_person` and the vision sensor | ⚠️ | Default `moondream` model only. Does not influence room detection. |
| `whollm.enable_tracking` / `disable_tracking` | ✅ | Needs a `switch.<camera>_auto_tracking`. |

### Screens and automation

| Feature | | Notes |
|---|:-:|---|
| Lovelace card | ✅ | |
| Lovelace YAML dashboard | ✅ | Needs Mushroom and card-mod. Ships demo entity names. |
| Standalone web dashboard | ⚠️ | You must edit `nginx.conf`. The vision panel is broken. |
| Blueprints: Room-Based Climate, Follow Me Music | ✅ | |
| Blueprints: Welcome Home Lights, Goodbye Routine | 🚧 | Wait for events that nothing fires yet. |

### In the repository but not connected

| Feature | | Where | What is missing |
|---|:-:|---|---|
| Arrival, departure and room-change **events** | 🚧 | `notifications.py` | Nothing creates the manager or calls it from the coordinator. |
| Services `refresh`, `correct_room`, `get_history`, `clear_history`, `get_patterns` | 🚧 | `services.py`, `services.yaml` | `async_setup_services` is never called. |
| Movement history and pattern detection | 🚧 | `history.py` | Only used by the unregistered services above. |
| Multi-zone tracking (home, work, gym, school) | 🚧 | `zones.py` | Not used by the coordinator. |
| ML predictor with uncertainty prompts | 🚧 | `ml_predictor.py`, `scripts/` | Not called by the coordinator. See [ML & training data](ml-and-training-data.md). |

Each of these has unit tests, so they are close to usable. They just need to be wired in.

## Known issues

Sorted by how likely you are to hit them.

1. **Options pages overwrite each other, and do not auto-reload.** Every page under *Configure* saves only its own
   values, which replaces whatever the other pages saved, and the integration is not reloaded afterwards. Effect:
   the last page you saved wins, and nothing applies until you reload.
   *Workaround:* reload after saving and re-check the other pages.
   *Where:* the options steps in `config_flow.py`, and `__init__.py` (no update listener).
2. **Learned habits are not saved.** They are loaded from `/config/whollm_learned_habits.json` at startup if that file
   exists, but nothing writes it except clearing patterns, so habits reset on every restart.
   *Where:* `HabitPredictor.save_learned_patterns` in `habits.py`.
3. **Shared signals in multi-person homes.** Mapped entities vote for every person; only device ownership is
   person-specific. See [shared signals](how-it-works.md#shared-signals-in-multi-person-homes). This is how the
   current design works rather than a bug, but it surprises people.
4. **The vision model cannot be chosen in the UI.** The code reads a `vision_model` setting that no setup or options
   page sets, so it is always `moondream`. *Where:* `__init__.py`.
5. **`services.yaml` is out of step with the code.** It describes five services that are not registered and has no
   entries for the three camera services that are, so those show without field help.
6. **Standalone dashboard rough edges.** `nginx.conf` contains a hard-coded LAN address you must replace; the
   `HA_URL` variable in `docker-compose.yml` is unused; the *Identify Now* button and vision panel use the old
   integration name `llm_presence`; room tiles are a fixed list. See [Dashboard](dashboard.md#known-limitations).
7. **Helper scripts still use the old name.** `check_integration.sh` and `setup_local_testing.sh` look for
   `custom_components/llm_presence`, which does not exist, so they stop immediately.
   `scripts/sync_training_data.sh` reads `llm_presence_events.jsonl`, but the integration writes
   `whollm_events.jsonl`. The ML scripts also hard-code example names (Alice, Bob, Max).
8. **The CrewAI URL is not pre-filled.** The wizard always pre-fills the Ollama address.
9. **Storage paths are fixed to `/config/`.** Home Assistant Core installs with another config directory are not
   supported.
10. **A confidence bonus never applies.** The "multiple signals agree" boost in the combiner can never trigger
    because of how the condition is written (`ConfidenceCombiner.combine` in `habits.py`).

## Roadmap

Carried over from the original README, with current state:

| Item | State |
|---|---|
| Event-driven updates instead of polling | 🗺️ Not started |
| ML model training from Home Assistant history | 🚧 Scripts and predictor exist, not connected |
| More LLM providers (OpenAI, Anthropic, llama.cpp) | 🚧 OpenAI and Anthropic code exists, not offered in the wizard |
| Better prompt engineering | 🗺️ Ongoing. Contributions welcome. |
| Dashboard card | ✅ Shipped as [`whollm-card.js`](../www/whollm-card.js) |

The 🚧 rows under [In the repository but not connected](#in-the-repository-but-not-connected) are the closest to done.

The longer-term ideas in the [ML enhancement proposal](ML_ENHANCEMENT_PROPOSAL.md) (face recognition, an image queue,
a presence database) are **not implemented**.
