# WhoLLM — Two-Track Plan

**Track A — Local:** run WhoLLM in Ben's own Home Assistant, with real rooms, people, devices, cameras and trained models.
**Track B — Repo:** whatever proves itself locally goes into `owens-ben/WhoLLM` in a generalized, configurable, tested form.

**Rule:** nothing household-specific goes into the repo. Local values live in HA config, a git-ignored overlay, or env vars. Every repo change has to work for the beta-tester profiles in `beta-testers/testers/` (minimal, family, studio and so on), not just for Ben's house.

---

## Where things stand (2026-10-21)

- `main` is at `cb8ff82` (PR #6: HA OAuth2 dashboard login and nginx fix). There are no open issues or PRs.
- Feature areas are in place: providers (Ollama, OpenAI, Anthropic, CrewAI), habits, history, zones, notifications, services, blueprints, Lovelace card, standalone dashboard, ML predictor and training scripts.
- Gaps found:
  - **Local values leaked into the repo:** the HA host IP was hardcoded in `dashboard/nginx.conf`, `dashboard/index.html` and `e2e/dashboard.spec.ts`.
  - **Hardcoded `/config` paths:** in `ml_predictor.py` (`MODEL_PATH`, feedback path), `event_logger.py`, `habits.py` and `history.py`. These should use `hass.config.path()`.
  - **Stale names:** `scripts/sync_training_data.sh` still reads `llm_presence_events.jsonl` (the old name), and `.gitignore` still lists `custom_components/llm_presence/`.
  - **Lint is not enforced:** CI runs lint with `continue-on-error`. Ruff reports 216 issues, mostly whitespace, and 13 files are unformatted.
  - **Old docs:** the CHANGELOG `[Unreleased]` section and the README roadmap don't mention multi-zone, notifications, services, blueprints, OAuth or the extra providers.
  - **Polling only:** `iot_class: local_polling`. The coordinator polls on `update_interval`, and "event-driven updates" is still open on the roadmap.

---

## Phase 0 — Separate local from generic (do this first, it unblocks both tracks)

| Repo (Track B) | Local (Track A) |
|---|---|
| Use `hass.config.path()` for every storage and model path | Keep using the default `/config` |
| Template nginx with `${HA_URL}` (envsubst in docker-compose) and remove the IP default in `index.html` | Put `.env` with your real `HA_URL` in a git-ignored `local/` folder |
| Use `homeassistant.local` in e2e tests, or read it from env | — |
| Fix `llm_presence` → `whollm` in scripts and `.gitignore` | Rename the old events file on the HA box |
| Add `local/` to `.gitignore` and document the "local overlay" pattern in DEVELOPMENT.md | `local/` holds the docker `.env`, model `.pkl` files, room and person YAML exports, and the deploy script |

**Status:** done on `claude/elegant-volta-kxqqvb`.

**Deliverable:** one PR named "generalize paths & dashboard config", plus a local `local/deploy.sh` that rsyncs `custom_components/whollm` to the HA box and restarts HA.

## Phase 1 — Make the repo healthy (Track B)

0. **Connect the unused code.** `services.py` is never imported, and it calls `get_habit_predictor` and `get_history_tracker`, which the refactor in 72f4f70 deleted. As a result `correct_room`, `get_history`, `refresh` and the other services are never registered, and 13 tests fail. `NotificationManager` is also never created, so the `whollm_person_arrived/left/room_changed` events never fire and the welcome/goodbye blueprints can't trigger.
1. Run `ruff check --fix` and `ruff format`, then remove `continue-on-error` from the lint job.
2. Fix the test setup so `pytest-homeassistant-custom-component` installs. Pin it in a `requirements_test.txt`, and add a SessionStart hook so cloud sessions can run the tests.
3. Make sure hassfest and HACS validation (`validate.yml`) pass.
4. Bring the CHANGELOG up to date and cut `v0.2.0-alpha`.

## Phase 2 — Run locally and collect data (Track A)

Monitoring: a private Slack channel, `#whollm`, receives every room change through the **WhoLLM - Activity Log** blueprint (`blueprints/activity_log.yaml`).

1. Deploy the current `main` to HA with `local/deploy.sh` and set up the real rooms, people and devices through the UI.
2. Let `event_logger` collect at least two weeks of `whollm_events.jsonl`. Record correct/incorrect feedback (`whollm_feedback.jsonl`).
3. Track accuracy per person and per room, and per signal source (LLM, heuristics, habits, ML). This is the baseline.
4. Write every rough edge down as a GitHub issue that says **local** or **generic**. These issues drive Phase 3.

## Phase 3 — Features (prove locally, then generalize)

Each feature first runs on the local HA, then lands in the repo as a PR with tests, config options and docs.

1. **Event-driven updates:** use `async_track_state_change_event` on the mapped entities, debounce, and keep polling as a fallback. Switch `iot_class` to `local_push`. This has the biggest effect on latency and LLM load.
2. **ML predictor in the integration:** make the training scripts read paths from CLI args or env, add a model-path config option, and add a `whollm.train_model` service (or document the offline flow). Blend it into the confidence scorer only when a model is present.
3. **Feedback loop:** add a `whollm.correct_location` service, plus a correction button on the card and dashboard, so corrections feed habits and ML.
4. **Prompt work:** build an eval harness from the logged events, replay them against different prompts and models, and score accuracy. The harness and the anonymized fixture format go in the repo; real logs stay local.
5. **Faster vision (optional):** add a CompreFace/Double-Take backend as an alternative to LLaVA, as described in `docs/ML_ENHANCEMENT_PROPOSAL.md`.

## Phase 4 — Release

- Polish the README (screenshots of the card and dashboard), add a HACS default-repo submission checklist, and tag `v0.3.0-beta`.

---

## Working conventions

- One feature per branch, one PR each, merged into `main` with CI green.
- Track A changes that are config only never need a commit. Track A changes that are code always become a Track B PR, otherwise local and repo drift apart.
- Before each PR, run `grep` for IPs, names and `/config` literals to catch leaks.
