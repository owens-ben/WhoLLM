"""Minimal stand-in for the Ollama HTTP API, for smoke-testing WhoLLM.

Answers /api/tags with a fixed model list and /api/generate by picking the
room named in the first active indicator in the prompt (or "unknown").
"""

from __future__ import annotations

import argparse
import re

from aiohttp import web

MODELS = ["llama3.2:latest", "llava:7b"]


async def tags(_request: web.Request) -> web.Response:
    return web.json_response({"models": [{"name": m} for m in MODELS]})


async def generate(request: web.Request) -> web.Response:
    body = await request.json()
    request.app["calls"] += 1
    prompt = body.get("prompt", "")
    match = re.search(r"ACTIVE INDICATORS:\s*\n\s*- [^\n]* in ([a-z0-9_]+)", prompt)
    room = match.group(1) if match else "unknown"
    return web.json_response({"model": body.get("model"), "response": f"{room}, 0.8, fake ollama", "done": True})


async def stats(request: web.Request) -> web.Response:
    return web.json_response({"generate_calls": request.app["calls"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()
    app = web.Application()
    app["calls"] = 0
    app.router.add_get("/api/tags", tags)
    app.router.add_post("/api/generate", generate)
    app.router.add_get("/_stats", stats)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
