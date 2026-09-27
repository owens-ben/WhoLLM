"""End-to-end tests against a real Home Assistant core and a fake Ollama server.

Unlike the unit tests, nothing inside the integration is mocked here: the
config flow, coordinator, entities and services all run in a real `hass`
instance and talk HTTP to a local stand-in for Ollama.
"""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.whollm.const import DOMAIN


@pytest.fixture
def expected_lingering_timers() -> bool:
    """The integration's provider sessions may leave aiohttp timers behind."""
    return True


@pytest.fixture(scope="session", autouse=True)
def _start_pycares_shutdown_thread():
    """Start pycares' process-wide daemon thread before any test snapshots threads.

    Newer pycares lazily starts one background thread on first DNS channel teardown.
    Older pytest-homeassistant-custom-component releases (the Python 3.12 / HA 2025.1
    CI leg) flag it as a leaked thread. Starting it up front keeps it out of the diff.
    """
    try:
        import pycares
    except ImportError:
        return
    manager = getattr(pycares, "_shutdown_manager", None)
    if manager is not None:
        manager.start()


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Let HA load custom_components/whollm."""
    return


@pytest.fixture
async def fake_ollama(socket_enabled, unused_tcp_port):
    """Run a minimal Ollama-compatible HTTP server on localhost."""
    prompts: list[str] = []

    async def tags(_request: web.Request) -> web.Response:
        return web.json_response({"models": [{"name": "llama3.2"}, {"name": "llava:7b"}]})

    async def generate(request: web.Request) -> web.Response:
        body = await request.json()
        prompts.append(body.get("prompt", ""))
        # The person being located is named in the system prompt
        room = "office" if "Alice" in body.get("system", "") else "living_room"
        return web.json_response({"response": json.dumps({"room": room, "confidence": 0.9})})

    app = web.Application()
    app.router.add_get("/api/tags", tags)
    app.router.add_post("/api/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_tcp_port
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    yield {"url": f"http://127.0.0.1:{port}", "prompts": prompts}
    await runner.cleanup()


async def _run_config_flow(hass: HomeAssistant, url: str) -> config_entries.ConfigEntry:
    hass.states.async_set("binary_sensor.office_motion", "on")
    hass.states.async_set("media_player.living_room_tv", "playing")

    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
    assert result["type"] is FlowResultType.FORM and result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"provider": "ollama", "url": url, "poll_interval": 30}
    )
    assert result["step_id"] == "model", result.get("errors")

    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"model": "llama3.2"})
    assert result["step_id"] == "rooms"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"rooms": ["office", "living_room"], "custom_rooms": ""}
    )
    assert result["step_id"] == "persons"

    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"persons": "Alice", "pets": ""})
    assert result["step_id"] == "room_entities"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "entities_office": ["binary_sensor.office_motion"],
            "entities_living_room": ["media_player.living_room_tv"],
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    return result["result"]


async def test_full_setup_entities_services_and_unload(hass: HomeAssistant, fake_ollama) -> None:
    """Config flow -> entities -> services -> unload, all against real HA."""
    entry = await _run_config_flow(hass, fake_ollama["url"])
    assert entry.state is ConfigEntryState.LOADED
    assert fake_ollama["prompts"], "coordinator never queried the LLM"

    room = hass.states.get("sensor.alice_room")
    assert room is not None
    assert room.state == "office"
    assert hass.states.get("binary_sensor.alice_in_office").state == "on"

    for service in (
        "refresh",
        "correct_room",
        "get_history",
        "clear_history",
        "get_patterns",
        "identify_person",
        "enable_tracking",
        "disable_tracking",
        "cleanup_storage",
    ):
        assert hass.services.has_service(DOMAIN, service), service

    await hass.services.async_call(DOMAIN, "refresh", {}, blocking=True)

    response = await hass.services.async_call(
        DOMAIN, "get_history", {"entity_name": "Alice"}, blocking=True, return_response=True
    )
    assert isinstance(response["transitions"], list)

    response = await hass.services.async_call(DOMAIN, "cleanup_storage", {}, blocking=True, return_response=True)
    assert response["success"] is True

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.NOT_LOADED
    assert not hass.services.has_service(DOMAIN, "refresh")
    assert not hass.services.has_service(DOMAIN, "identify_person")


async def test_options_flow_opens(hass: HomeAssistant, fake_ollama) -> None:
    """The options flow (where person devices are mapped) must open on current HA."""
    entry = await _run_config_flow(hass, fake_ollama["url"])

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU

    result = await hass.config_entries.options.async_configure(result["flow_id"], {"next_step_id": "person_devices"})
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "person_devices"


async def test_options_pages_merge_and_apply_without_restart(hass: HomeAssistant, fake_ollama) -> None:
    """Saving one options page must not wipe another, and changes apply via reload."""
    entry = await _run_config_flow(hass, fake_ollama["url"])
    hass.states.async_set("light.office_lamp", "on")
    hass.states.async_set("device_tracker.alice_phone", "home")

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(result["flow_id"], {"next_step_id": "room_entities"})
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "entities_office": ["binary_sensor.office_motion", "light.office_lamp"],
            "entities_living_room": ["media_player.living_room_tv"],
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(result["flow_id"], {"next_step_id": "person_devices"})
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"devices_Alice": ["device_tracker.alice_phone"]}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    office = [e["entity_id"] for e in entry.options["room_entities"]["office"]]
    assert "light.office_lamp" in office, "room mappings were wiped by the person-devices save"
    assert entry.options["person_devices"] == {"Alice": ["device_tracker.alice_phone"]}

    # The update listener reloaded the entry, so the live coordinator sees the new mapping
    assert entry.state is ConfigEntryState.LOADED
    coordinator = hass.data[DOMAIN][entry.entry_id]
    assert coordinator.person_devices == {"Alice": ["device_tracker.alice_phone"]}
