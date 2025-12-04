"""Config flow for LLM Room Presence integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_MODEL,
    CONF_PERSONS,
    CONF_PETS,
    CONF_POLL_INTERVAL,
    CONF_PROVIDER,
    CONF_ROOMS,
    CONF_URL,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PROVIDER,
    DEFAULT_URL,
    DOMAIN,
    SUPPORTED_PROVIDERS,
    VALID_ROOMS,
)
from .providers import get_provider

_LOGGER = logging.getLogger(__name__)


class LLMPresenceConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for LLM Room Presence."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: dict[str, Any] = {}
        self._available_models: list[str] = []

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
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

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
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
                    vol.Required(CONF_MODEL, default=model_options[0] if model_options else DEFAULT_MODEL): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                            custom_value=True,
                        )
                    ),
                }
            ),
        )

    async def async_step_rooms(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle room configuration step."""
        if user_input is not None:
            self._data[CONF_ROOMS] = user_input.get(CONF_ROOMS, VALID_ROOMS)
            return await self.async_step_persons()

        return self.async_show_form(
            step_id="rooms",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_ROOMS, default=VALID_ROOMS): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=VALID_ROOMS,
                            multiple=True,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
        )

    async def async_step_persons(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle person/pet configuration step."""
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
            
            self._data[CONF_PERSONS] = persons
            self._data[CONF_PETS] = pets
            
            # Create the config entry
            return self.async_create_entry(
                title=f"LLM Presence ({self._data[CONF_PROVIDER]})",
                data=self._data,
            )

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
                "pets_hint": "Comma-separated names (e.g., Max, Luna)",
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
    """Handle options flow for LLM Presence."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
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
                }
            ),
        )

