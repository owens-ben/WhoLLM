"""Data update coordinator for LLM Presence."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

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
    VALID_ROOMS,
)
from .providers import get_provider

_LOGGER = logging.getLogger(__name__)


class LLMPresenceCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage LLM presence detection."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        self.entry = entry
        self.provider_type = entry.data.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        self.provider_url = entry.data.get(CONF_URL, DEFAULT_URL)
        self.model = entry.data.get(CONF_MODEL, DEFAULT_MODEL)
        self.persons = entry.data.get(CONF_PERSONS, [])
        self.pets = entry.data.get(CONF_PETS, [])
        self.rooms = entry.data.get(CONF_ROOMS, VALID_ROOMS)
        
        poll_interval = entry.data.get(CONF_POLL_INTERVAL, DEFAULT_POLL_INTERVAL)
        
        # Initialize the LLM provider
        self.provider = get_provider(
            provider_type=self.provider_type,
            url=self.provider_url,
            model=self.model,
        )
        
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=poll_interval),
        )
        
        _LOGGER.info(
            "LLM Presence coordinator initialized: provider=%s, url=%s, model=%s",
            self.provider_type,
            self.provider_url,
            self.model,
        )

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from LLM provider."""
        try:
            # Gather sensor context from Home Assistant
            context = await self._gather_sensor_context()
            
            # Query LLM for each person/pet
            results: dict[str, Any] = {
                "persons": {},
                "pets": {},
                "last_context": context,
            }
            
            for person in self.persons:
                person_name = person.get("name", "unknown")
                guess = await self.provider.deduce_presence(
                    hass=self.hass,
                    context=context,
                    entity_name=person_name,
                    entity_type="person",
                    rooms=self.rooms,
                )
                results["persons"][person_name] = guess
            
            for pet in self.pets:
                pet_name = pet.get("name", "unknown")
                guess = await self.provider.deduce_presence(
                    hass=self.hass,
                    context=context,
                    entity_name=pet_name,
                    entity_type="pet",
                    rooms=self.rooms,
                )
                results["pets"][pet_name] = guess
            
            return results
            
        except Exception as err:
            _LOGGER.error("Error fetching LLM presence data: %s", err)
            raise UpdateFailed(f"Error communicating with LLM: {err}") from err

    async def _gather_sensor_context(self) -> dict[str, Any]:
        """Gather current state of all relevant sensors."""
        context: dict[str, Any] = {
            "lights": {},
            "motion": {},
            "media": {},
            "device_trackers": {},
        }
        
        # Get all entity states
        states = self.hass.states.async_all()
        
        for state in states:
            entity_id = state.entity_id
            
            # Collect lights
            if entity_id.startswith("light."):
                context["lights"][entity_id] = {
                    "state": state.state,
                    "last_changed": state.last_changed.isoformat() if state.last_changed else None,
                }
            
            # Collect motion sensors
            elif entity_id.startswith("binary_sensor.") and "motion" in entity_id.lower():
                context["motion"][entity_id] = {
                    "state": state.state,
                    "last_changed": state.last_changed.isoformat() if state.last_changed else None,
                }
            
            # Collect media players
            elif entity_id.startswith("media_player."):
                context["media"][entity_id] = {
                    "state": state.state,
                    "last_changed": state.last_changed.isoformat() if state.last_changed else None,
                }
            
            # Collect device trackers and persons
            elif entity_id.startswith(("device_tracker.", "person.")):
                context["device_trackers"][entity_id] = {
                    "state": state.state,
                    "last_changed": state.last_changed.isoformat() if state.last_changed else None,
                }
        
        return context


