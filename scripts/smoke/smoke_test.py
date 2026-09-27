"""End-to-end smoke test: boot a real Home Assistant with WhoLLM installed.

Starts a fake Ollama server and a throwaway Home Assistant instance, onboards
it, runs the WhoLLM config flow over the REST API exactly as the UI would,
flips simulated sensors and checks that the room sensors follow. Finally it
scans the Home Assistant log for errors and blocking-I/O warnings coming from
the integration.

Usage:
    python scripts/smoke/smoke_test.py [--keep]

Requires `homeassistant` to be installed in the current Python environment.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CLIENT_ID = "http://smoke.test/"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Failure(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Failure(msg)
    print(f"  ok  {msg}")


async def wait_http(session: aiohttp.ClientSession, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with session.get(url) as resp:
                if resp.status < 500:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1)
    raise Failure(f"timed out waiting for {url}")


async def onboard(session: aiohttp.ClientSession, base: str) -> str:
    """Create the owner user and return an access token."""
    async with session.post(
        f"{base}/api/onboarding/users",
        json={"client_id": CLIENT_ID, "name": "Smoke", "username": "smoke", "password": "smoke-pass", "language": "en"},
    ) as resp:
        resp.raise_for_status()
        code = (await resp.json())["auth_code"]
    async with session.post(
        f"{base}/auth/token",
        data={"grant_type": "authorization_code", "code": code, "client_id": CLIENT_ID},
    ) as resp:
        resp.raise_for_status()
        token = (await resp.json())["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    for step, body in (("core_config", {}), ("analytics", {}), ("integration", {"client_id": CLIENT_ID, "redirect_uri": CLIENT_ID})):
        async with session.post(f"{base}/api/onboarding/{step}", json=body, headers=headers) as resp:
            if resp.status not in (200, 403):  # 403: step already done
                raise Failure(f"onboarding {step} failed: {resp.status} {await resp.text()}")
    return token


async def api(session, method, url, token, **kw):
    async with session.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kw) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise Failure(f"{method} {url} -> {resp.status}: {text}")
        return await resp.json() if text else None


async def state(session, base, token, entity_id):
    async with session.get(f"{base}/api/states/{entity_id}", headers={"Authorization": f"Bearer {token}"}) as resp:
        return await resp.json() if resp.status == 200 else None


async def wait_state(session, base, token, entity_id, expected, timeout=30.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        st = await state(session, base, token, entity_id)
        last = st and st["state"]
        if last == expected:
            return st
        await asyncio.sleep(1)
    raise Failure(f"{entity_id} expected {expected!r}, last seen {last!r}")


async def run(base: str, ollama: str, log_path: Path) -> None:
    async with aiohttp.ClientSession() as session:
        print("waiting for Home Assistant to start ...")
        await wait_http(session, f"{base}/api/onboarding", timeout=240)
        token = await onboard(session, base)
        print("onboarded")

        # Wait for the simulated sensors to exist
        await wait_state(session, base, token, "binary_sensor.office_motion", "off", timeout=60)

        print("config flow:")
        flow = await api(session, "POST", f"{base}/api/config/config_entries/flow", token,
                         json={"handler": "whollm", "show_advanced_options": False})
        check(flow["step_id"] == "user", "flow starts at user step")
        flow_url = f"{base}/api/config/config_entries/flow/{flow['flow_id']}"

        flow = await api(session, "POST", flow_url, token,
                         json={"provider": "ollama", "url": ollama, "poll_interval": 10})
        check(flow.get("step_id") == "model", f"connection accepted -> model step (got {flow.get('step_id')}, errors={flow.get('errors')})")

        flow = await api(session, "POST", flow_url, token, json={"model": "llama3.2:latest"})
        check(flow.get("step_id") == "rooms", "model accepted -> rooms step")

        flow = await api(session, "POST", flow_url, token, json={"rooms": ["office", "kitchen", "bedroom"], "custom_rooms": ""})
        check(flow.get("step_id") == "persons", f"rooms accepted -> persons step (errors={flow.get('errors')})")

        flow = await api(session, "POST", flow_url, token, json={"persons": "Alice", "pets": ""})
        check(flow.get("step_id") == "room_entities", f"persons accepted -> room_entities step (errors={flow.get('errors')})")

        flow = await api(session, "POST", flow_url, token, json={
            "entities_office": ["binary_sensor.office_motion", "switch.alice_pc"],
            "entities_kitchen": ["binary_sensor.kitchen_motion"],
            "entities_bedroom": [],
        })
        check(flow.get("type") == "create_entry", f"entry created (got {flow.get('type')})")
        entry_id = flow["result"]["entry_id"]

        print("entities:")
        st = await state(session, base, token, "sensor.alice_room")
        deadline = time.monotonic() + 30
        while st is None and time.monotonic() < deadline:
            await asyncio.sleep(1)
            st = await state(session, base, token, "sensor.alice_room")
        check(st is not None, "sensor.alice_room exists")
        print(f"      initial: {st['state']} {st['attributes'].get('confidence')}")
        for room in ("office", "kitchen", "bedroom"):
            check(await state(session, base, token, f"binary_sensor.alice_in_{room}") is not None, f"binary_sensor.alice_in_{room} exists")

        print("presence follows sensors:")
        await api(session, "POST", f"{base}/api/services/input_boolean/turn_on", token, json={"entity_id": "input_boolean.office_motion_sim"})
        await api(session, "POST", f"{base}/api/services/whollm/refresh", token, json={})
        st = await wait_state(session, base, token, "sensor.alice_room", "office")
        check(True, f"office motion -> alice in office ({st['attributes'].get('confidence')})")
        await wait_state(session, base, token, "binary_sensor.alice_in_office", "on")
        check(True, "binary_sensor.alice_in_office on")

        await api(session, "POST", f"{base}/api/services/input_boolean/turn_off", token, json={"entity_id": "input_boolean.office_motion_sim"})
        await api(session, "POST", f"{base}/api/services/input_boolean/turn_on", token, json={"entity_id": "input_boolean.kitchen_motion_sim"})
        await api(session, "POST", f"{base}/api/services/whollm/refresh", token, json={})
        await wait_state(session, base, token, "sensor.alice_room", "kitchen")
        check(True, "kitchen motion -> alice in kitchen")

        print("services:")
        await api(session, "POST", f"{base}/api/services/whollm/correct_room", token,
                  json={"entity_id": "sensor.alice_room", "room": "bedroom"})
        await wait_state(session, base, token, "sensor.alice_room", "bedroom", timeout=5)
        check(True, "correct_room sets the sensor")
        hist = await api(session, "POST", f"{base}/api/services/whollm/get_history?return_response", token,
                         json={"entity_name": "Alice"})
        transitions = hist["service_response"]["transitions"]
        check(len(transitions) >= 1, f"get_history returns transitions ({len(transitions)})")
        await api(session, "POST", f"{base}/api/services/whollm/cleanup_storage?return_response", token, json={})
        check(True, "cleanup_storage runs")

        print("reload + unload:")
        await api(session, "POST", f"{base}/api/config/config_entries/entry/{entry_id}/reload", token)
        await asyncio.sleep(3)
        entries = await api(session, "GET", f"{base}/api/config/config_entries/entry?domain=whollm", token)
        check(entries[0]["state"] == "loaded", f"entry loaded after reload (state={entries[0]['state']})")

        async with session.get(f"{ollama}/_stats") as resp:
            calls = (await resp.json())["generate_calls"]
        check(calls > 0, f"integration queried the LLM ({calls} calls)")

    print("log scan:")
    log = log_path.read_text(errors="replace")
    problems = [
        line for line in log.splitlines()
        if ("whollm" in line.lower())
        and re.search(r"\b(ERROR|CRITICAL)\b|Detected blocking call|Traceback|deprecated", line)
    ]
    for line in problems[:40]:
        print(f"  !! {line}")
    check(not problems, "no errors, tracebacks, deprecations or blocking calls from whollm in the log")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="keep the temporary config dir")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="whollm-smoke-"))
    config = workdir / "config"
    shutil.copytree(HERE / "ha_config", config)
    (config / "custom_components").mkdir()
    shutil.copytree(REPO / "custom_components" / "whollm", config / "custom_components" / "whollm",
                    ignore=shutil.ignore_patterns("__pycache__"))

    ha_port, ollama_port = free_port(), free_port()
    (config / "configuration.yaml").open("a").write(f"\nhttp:\n  server_port: {ha_port}\n")
    base, ollama = f"http://127.0.0.1:{ha_port}", f"http://127.0.0.1:{ollama_port}"
    log_path = config / "home-assistant.log"

    procs = [
        subprocess.Popen([sys.executable, str(HERE / "fake_ollama.py"), "--port", str(ollama_port)]),
        subprocess.Popen(
            [sys.executable, "-m", "homeassistant", "-c", str(config), "--log-file", str(log_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        ),
    ]
    try:
        asyncio.run(run(base, ollama, log_path))
        print("\nSMOKE TEST PASSED")
        return 0
    except Failure as err:
        print(f"\nSMOKE TEST FAILED: {err}")
        print(f"log: {log_path}")
        args.keep = True
        return 1
    finally:
        for proc in reversed(procs):
            proc.send_signal(signal.SIGINT if proc is procs[-1] else signal.SIGTERM)
        for proc in procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"kept {workdir}")


if __name__ == "__main__":
    sys.exit(main())
