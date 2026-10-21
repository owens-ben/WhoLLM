# Roomer

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/v/release/owens-ben/Roomer?include_prereleases)](https://github.com/owens-ben/Roomer/releases)
[![Docs](https://img.shields.io/badge/docs-owens--ben.github.io%2Froomer-teal.svg)](https://owens-ben.github.io/roomer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2024.1%2B-blue.svg)](https://www.home-assistant.io/)

**A Home Assistant integration that works out *who* is in which room — not just whether a room is occupied — by combining the weak signals your home already has.**

> **Status: Alpha** — Running, but expect rough edges.

> **Formerly WhoLLM.** The rename to Roomer (domain `roomer`) is breaking — see [Migrating from WhoLLM](https://owens-ben.github.io/roomer/migrating/).

## 📖 [Read the documentation →](https://owens-ben.github.io/roomer/)

---

## How it works

Occupancy sensors tell you a room is occupied. Roomer tells you *who* is there, by combining signals the way a person would:

```
Motion in office
+ Alice's PC is on
+ Alice's phone on home WiFi
+ Bob's phone is away
─────────────────────────────
→ sensor.alice_room = office
```

Today the reasoning step uses a local LLM via [Ollama](https://ollama.com), then a confidence combiner and learned habits adjust the result. The more per-person signals you have (a personal PC, room-level phone tracking), the better it works — see [Accuracy & limits](https://owens-ben.github.io/roomer/using/accuracy/).

## Quick start

1. Run [Ollama](https://ollama.com) and pull a model: `ollama pull llama3.2`
2. In HACS → Integrations → ⋮ → **Custom repositories**, add `https://github.com/owens-ben/roomer` (category: Integration) and download **Roomer**
3. Restart Home Assistant, then **Settings → Devices & Services → Add Integration → Roomer**

Full walkthrough: [Install](https://owens-ben.github.io/roomer/getting-started/install/) · [First setup](https://owens-ben.github.io/roomer/getting-started/first-setup/) · [Map your rooms](https://owens-ben.github.io/roomer/getting-started/map-rooms/)

## Links

- [Entities](https://owens-ben.github.io/roomer/using/entities/) · [Dashboard card](https://owens-ben.github.io/roomer/using/dashboard-card/) · [Automations](https://owens-ben.github.io/roomer/using/automations/) · [Troubleshooting](https://owens-ben.github.io/roomer/using/troubleshooting/)
- [Contributing](https://owens-ben.github.io/roomer/contributing/architecture/) — architecture, dev setup, adding a provider
- [Changelog](CHANGELOG.md)

## Acknowledgments

- [Ollama](https://ollama.ai) — Making local LLMs accessible
- [Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) — Inspiration for room-level presence

## License

MIT License — see [LICENSE](LICENSE) for details.
