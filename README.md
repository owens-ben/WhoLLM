# LLM Room Presence

> 🚧 **Work in Progress** - See `develop` branch for active development

A Home Assistant custom integration that uses local LLMs to intelligently deduce room presence.

## The Problem

Traditional room presence detection fails when users are stationary:
- 🪑 Working at a desk → motion sensor times out
- 📺 Watching TV → motion sensor times out  
- 😴 Sleeping → no motion detected

Motion sensors alone can't tell if someone is still in a room.

## The Solution

Use a local LLM (Ollama) to reason about **multiple signals together**:
- 💡 Which lights are on?
- 🏃 Recent motion activity?
- 📺 Media playing?
- 📱 Phone location?

The LLM considers all signals holistically and makes an intelligent guess about which room you're in.

## Features (Planned)

- 🧠 **LLM-Powered** - Local inference with Ollama (OpenAI/Anthropic planned)
- 👥 **Multi-Person** - Track multiple household members
- 🐾 **Pet Tracking** - Track pets with pet-specific logic
- 🏠 **Bayesian Integration** - Binary sensors feed into HA's Bayesian sensors
- 🔒 **Privacy-First** - All processing happens locally

## Installation

*Coming soon - will be available via HACS*

## Development

Active development happens on the `develop` branch.

```bash
git clone https://github.com/owens-ben/llm-room-presence.git
cd llm-room-presence
git checkout develop
```

## License

MIT License - see [LICENSE](LICENSE) for details.
