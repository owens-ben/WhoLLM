# WhoLLM guide

![How WhoLLM turns weak signals into a room per person](assets/hero.svg)

WhoLLM is a Home Assistant integration that works out **who is in which room** by combining many weak signals
(motion, PCs, TVs, lights, phones, cameras) with a local LLM. This guide takes you from install to automations,
and explains what is going on under the hood when a prediction looks wrong.

## Pick your path

| I want to… | Read |
|---|---|
| **Install it and see it work** (about 15 minutes) | [Getting started](getting-started.md) |
| Tell it about my rooms, sensors and people | [Configuration](configuration.md) |
| Understand why it picked *that* room | [How it works](how-it-works.md) |
| Look up sensor names, attributes and services | [Entities & services](entities-and-services.md) |
| Turn lights on, move music, adjust climate | [Automations](automations.md) |
| See it on a dashboard | [Dashboard](dashboard.md) |
| Fix something that is not working | [Troubleshooting](troubleshooting.md) |
| Know what is finished and what is not | [Status & roadmap](status.md) |
| Work on the code | [Development](development.md) |
| Train ML models from my own data | [ML & training data](ml-and-training-data.md) |

If you are new, read the first two pages in order and skim the rest when you need them.

## Ten-second glossary

| Term | Meaning |
|---|---|
| **Provider** | The thing that answers the "which room?" question. Ollama (local) is the supported one; CrewAI is experimental. |
| **Poll** | One round of detection. Every poll asks about every person and pet. Default: every 10 seconds. |
| **Room entity** | A sensor or device you tell WhoLLM lives in a room (a motion sensor, a light, a TV…). |
| **Hint type** | What kind of signal an entity is (motion, media, light, computer, camera, …). Guessed from the entity ID. |
| **Indicator** | A piece of evidence that was active during a poll, shown in the sensor's `indicators` attribute. |
| **Confidence** | A number from 0 to 1 for how sure WhoLLM is. It is relative to what your sensors can prove, see [how it works](how-it-works.md#signals-weights-and-confidence). |
| **Fallback** | A simple rule-based guess used when the LLM cannot be reached. |
| **Habit** | A learned "this person is usually in this room around this time" hint. |

## Where things live

```
WhoLLM/
├── custom_components/whollm/   the integration itself
├── docs/                       this guide (and its images in docs/assets/)
├── blueprints/                 importable automation blueprints
├── dashboard/                  standalone web dashboard + Lovelace YAML dashboard
├── www/                        custom Lovelace card (whollm-card.js)
├── examples/  automations/     copy-paste automation YAML
├── scripts/                    ML training helpers and docs tooling
├── tests/  e2e/  beta-testers/ unit tests, dashboard browser tests, simulated households
└── README.md  CHANGELOG.md  CONTRIBUTING.md
```
