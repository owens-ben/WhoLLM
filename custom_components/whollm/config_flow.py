"""Config flow for WhoLLM integration."""

from __future__ import annotations

import logging
import re
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_CONFIDENCE_WEIGHTS,
    CONF_LEARNING_ENABLED,
    CONF_MODEL,
    CONF_PERSON_DEVICES,
    CONF_PERSONS,
    CONF_PETS,
    CONF_POLL_INTERVAL,
    CONF_PROVIDER,
    CONF_ROOM_ENTITIES,
    CONF_ROOMS,
    CONF_TIMEOUT,
    CONF_URL,
    DEFAULT_CONFIDENCE_WEIGHTS,
    DEFAULT_LEARNING_ENABLED,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PROVIDER,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    DOMAIN,
    SUPPORTED_PROVIDERS,
    VALID_ROOMS,
)
from .providers import get_provider

_LOGGER = logging.getLogger(__name__)


def _guess_entity_hint(entity_id: str) -> str:
    """Guess the entity hint type from the entity ID."""
    entity_lower = entity_id.lower()

    if "motion" in entity_lower or "occupancy" in entity_lower:
        return "motion"
    if "media_player" in entity_lower or "tv" in entity_lower:
        return "media"
    if "light." in entity_lower:
        return "light"
    if "pc" in entity_lower or "computer" in entity_lower or "desktop" in entity_lower:
        return "computer"
    if "camera" in entity_lower or "person" in entity_lower or "animal" in entity_lower:
        return "camera"
    if "door" in entity_lower or "window" in entity_lower or "contact" in entity_lower:
        return "door"
    if "temperature" in entity_lower or "humidity" in entity_lower:
        return "climate"
    if "device_tracker" in entity_lower:
        return "presence"
    return "appliance"


class LLMPresenceConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for WhoLLM."""

    VERSION = 2  # Bumped for new room_entities config

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: dict[str, Any] = {}
        self._available_models: list[str] = []

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial step - provider configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Test connection to provider
            provider = get_provider(
                provider_type=user_input[CONF_PROVIDER],
                url=user_input[CONF_URL],
                model=user_input.get(CONF_MODEL, DEFAULT_MODEL),
            )

            if await provider.test_connection():
                self._data.update(user_input)
                self._available_models = await provider.get_available_models()
                return await self.async_step_model()
            else:
                errors["base"] = "cannot_connect"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_PROVIDER, default=DEFAULT_PROVIDER): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=SUPPORTED_PROVIDERS,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(CONF_URL, default=DEFAULT_URL): str,
                    vol.Optional(CONF_POLL_INTERVAL, default=DEFAULT_POLL_INTERVAL): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=300,
                            step=5,
                            unit_of_measurement="seconds",
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_model(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle model selection step."""
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_rooms()

        # Build model options
        model_options = self._available_models if self._available_models else [DEFAULT_MODEL]

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MODEL, default=model_options[0] if model_options else DEFAULT_MODEL
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                            custom_value=True,
                        )
                    ),
                }
            ),
        )

    async def async_step_rooms(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle room configuration step."""

        errors: dict[str, str] = {}

        if user_input is not None:
            rooms = user_input.get(CONF_ROOMS, [])
            custom_rooms = user_input.get("custom_rooms", "")

            # Add custom rooms with sanitization
            if custom_rooms:
                for room in custom_rooms.split(","):
                    room = room.strip().lower().replace(" ", "_")
                    # Remove any characters that aren't alphanumeric, underscore, or hyphen
                    room = re.sub(r"[^\w-]", "", room)
                    if room and room not in rooms:
                        rooms.append(room)

            # Sanitize all room names (including predefined ones)
            sanitized = []
            for room in rooms:
                room = room.strip().lower().replace(" ", "_")
                room = re.sub(r"[^\w-]", "", room)
                if room and room not in sanitized:
                    sanitized.append(room)
            rooms = sanitized

            if not rooms:
                errors["base"] = "at_least_one_room"
            else:
                self._data[CONF_ROOMS] = rooms
                return await self.async_step_persons()

        return self.async_show_form(
            step_id="rooms",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_ROOMS, default=VALID_ROOMS[:6]): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=VALID_ROOMS[:-2],  # Exclude 'away' and 'unknown'
                            multiple=True,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                    vol.Optional("custom_rooms", default=""): str,
                }
            ),
            description_placeholders={
                "custom_rooms_hint": "Add custom rooms (comma-separated, e.g., garage, basement)",
            },
            errors=errors,
        )

    async def async_step_persons(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle person/pet configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Parse persons and pets from input
            persons = []
            pets = []

            persons_input = user_input.get(CONF_PERSONS, "")
            if persons_input:
                for name in persons_input.split(","):
                    name = name.strip()
                    if name:
                        persons.append({"name": name})

            pets_input = user_input.get(CONF_PETS, "")
            if pets_input:
                for name in pets_input.split(","):
                    name = name.strip()
                    if name:
                        pets.append({"name": name})

            # Validate that at least one person or pet is configured
            if not persons and not pets:
                errors["base"] = "at_least_one_entity"
            else:
                self._data[CONF_PERSONS] = persons
                self._data[CONF_PETS] = pets
                return await self.async_step_room_entities()

        return self.async_show_form(
            step_id="persons",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_PERSONS, default=""): str,
                    vol.Optional(CONF_PETS, default=""): str,
                }
            ),
            description_placeholders={
                "persons_hint": "Comma-separated names (e.g., Alice, Bob)",
                "pets_hint": "Comma-separated names (e.g., Fido, Whiskers)",
            },
            errors=errors,
        )

    async def async_step_room_entities(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle room-entity mapping configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Process room entity mappings, validating entities exist
            room_entities = {}
            invalid_entities = []
            for room in self._data.get(CONF_ROOMS, []):
                key = f"entities_{room}"
                entities = user_input.get(key, [])
                if entities:
                    valid = []
                    for e in entities:
                        if self.hass.states.get(e) is not None:
                            valid.append({"entity_id": e, "hint_type": _guess_entity_hint(e)})
                        else:
                            invalid_entities.append(e)
                    if valid:
                        room_entities[room] = valid

            if invalid_entities:
                _LOGGER.warning(
                    "Skipped non-existent entities during config: %s",
                    ", ".join(invalid_entities),
                )

            self._data[CONF_ROOM_ENTITIES] = room_entities

            # Initialize empty person devices (can be configured in options later)
            self._data[CONF_PERSON_DEVICES] = {}
            self._data[CONF_LEARNING_ENABLED] = True

            # Create the config entry
            return self.async_create_entry(
                title=f"WhoLLM ({self._data[CONF_PROVIDER]})",
                data=self._data,
            )

        # Build dynamic schema based on selected rooms
        rooms = self._data.get(CONF_ROOMS, [])
        schema_dict = {}

        for room in rooms:
            # Create entity selector for each room
            schema_dict[vol.Optional(f"entities_{room}", default=[])] = selector.EntitySelector(
                selector.EntitySelectorConfig(
                    multiple=True,
                    filter=selector.EntityFilterSelectorConfig(
                        domain=[
                            "binary_sensor",
                            "sensor",
                            "light",
                            "switch",
                            "media_player",
                            "device_tracker",
                            "camera",
                        ]
                    ),
                )
            )

        return self.async_show_form(
            step_id="room_entities",
            data_schema=vol.Schema(schema_dict),
            description_placeholders={
                "room_hint": "Select entities that indicate activity in each room. This helps WhoLLM understand which sensors/devices belong to which room.",
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return LLMPresenceOptionsFlow(config_entry)


class LLMPresenceOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for WhoLLM."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options - show menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["general", "room_entities", "person_devices", "confidence_weights"],
        )

    async def async_step_general(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """General options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="general",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_POLL_INTERVAL,
                        default=self.config_entry.data.get(CONF_POLL_INTERVAL, DEFAULT_POLL_INTERVAL),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=300,
                            step=5,
                            unit_of_measurement="seconds",
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=self.config_entry.options.get(
                            CONF_TIMEOUT, self.config_entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=5,
                            max=120,
                            step=5,
                            unit_of_measurement="seconds",
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_LEARNING_ENABLED,
                        default=self.config_entry.data.get(CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED),
                    ): bool,
                }
            ),
        )

    async def async_step_room_entities(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure room-entity mappings."""
        if user_input is not None:
            # Process and save room entity mappings, validating entities exist
            room_entities = self.config_entry.data.get(CONF_ROOM_ENTITIES, {}).copy()
            rooms = self.config_entry.data.get(CONF_ROOMS, [])
            invalid_entities = []

            for room in rooms:
                key = f"entities_{room}"
                entities = user_input.get(key, [])
                if entities:
                    valid = []
                    for e in entities:
                        if self.hass.states.get(e) is not None:
                            valid.append({"entity_id": e, "hint_type": _guess_entity_hint(e)})
                        else:
                            invalid_entities.append(e)
                    if valid:
                        room_entities[room] = valid
                    elif room in room_entities:
                        del room_entities[room]
                elif room in room_entities:
                    del room_entities[room]

            if invalid_entities:
                _LOGGER.warning(
                    "Skipped non-existent entities during options update: %s",
                    ", ".join(invalid_entities),
                )

            return self.async_create_entry(title="", data={CONF_ROOM_ENTITIES: room_entities})

        # Build dynamic schema
        rooms = self.config_entry.data.get(CONF_ROOMS, [])
        current_mappings = self.config_entry.data.get(CONF_ROOM_ENTITIES, {})
        schema_dict = {}

        for room in rooms:
            current_entities = [e["entity_id"] for e in current_mappings.get(room, [])]
            schema_dict[vol.Optional(f"entities_{room}", default=current_entities)] = selector.EntitySelector(
                selector.EntitySelectorConfig(
                    multiple=True,
                    filter=selector.EntityFilterSelectorConfig(
                        domain=[
                            "binary_sensor",
                            "sensor",
                            "light",
                            "switch",
                            "media_player",
                            "device_tracker",
                            "camera",
                        ]
                    ),
                )
            )

        return self.async_show_form(
            step_id="room_entities",
            data_schema=vol.Schema(schema_dict),
        )

    async def async_step_person_devices(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure person-device ownership mappings."""
        if user_input is not None:
            # Process and save person device mappings
            person_devices = {}
            persons = self.config_entry.data.get(CONF_PERSONS, [])
            pets = self.config_entry.data.get(CONF_PETS, [])

            for person in persons + pets:
                name = person.get("name", "")
                key = f"devices_{name}"
                devices = user_input.get(key, [])
                if devices:
                    person_devices[name] = devices

            return self.async_create_entry(title="", data={CONF_PERSON_DEVICES: person_devices})

        # Build dynamic schema
        persons = self.config_entry.data.get(CONF_PERSONS, [])
        pets = self.config_entry.data.get(CONF_PETS, [])
        current_mappings = self.config_entry.data.get(CONF_PERSON_DEVICES, {})
        schema_dict = {}

        for entity in persons + pets:
            name = entity.get("name", "")
            current_devices = current_mappings.get(name, [])
            schema_dict[vol.Optional(f"devices_{name}", default=current_devices)] = selector.EntitySelector(
                selector.EntitySelectorConfig(
                    multiple=True,
                    filter=selector.EntityFilterSelectorConfig(
                        domain=["device_tracker", "binary_sensor", "switch", "sensor"]
                    ),
                )
            )

        return self.async_show_form(
            step_id="person_devices",
            data_schema=vol.Schema(schema_dict),
            description_placeholders={
                "hint": "Assign devices owned by each person (their phone, PC, etc). If that device is active, it strongly suggests that person is nearby.",
            },
        )

    async def async_step_confidence_weights(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure confidence weights for different signal types."""
        if user_input is not None:
            weights = {}
            for key, value in user_input.items():
                if key.startswith("weight_"):
                    hint_type = key.replace("weight_", "")
                    weights[hint_type] = value / 100  # Convert from percentage

            return self.async_create_entry(title="", data={CONF_CONFIDENCE_WEIGHTS: weights})

        current_weights = self.config_entry.data.get(CONF_CONFIDENCE_WEIGHTS, DEFAULT_CONFIDENCE_WEIGHTS)
        schema_dict = {}

        weight_descriptions = {
            "camera": "Camera AI Detection",
            "computer": "PC/Computer Active",
            "media": "Media Playing",
            "motion": "Motion Sensor",
            "presence": "BLE/WiFi Presence",
            "appliance": "Appliance In Use",
            "light": "Light On",
            "door": "Door/Window Sensor",
            "llm_reasoning": "LLM Reasoning",
            "habit": "Learned Patterns",
        }

        for hint_type, _description in weight_descriptions.items():
            default_value = int(current_weights.get(hint_type, 0.5) * 100)
            schema_dict[vol.Optional(f"weight_{hint_type}", default=default_value)] = selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=100,
                    step=5,
                    unit_of_measurement="%",
                    mode=selector.NumberSelectorMode.SLIDER,
                )
            )

        return self.async_show_form(
            step_id="confidence_weights",
            data_schema=vol.Schema(schema_dict),
            description_placeholders={
                "hint": "Adjust how much each signal type contributes to presence detection. Higher = more influence.",
            },
        )

