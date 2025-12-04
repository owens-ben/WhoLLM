"""Sensor platform for LLM Room Presence."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import ATTR_CONFIDENCE, ATTR_INDICATORS, ATTR_RAW_RESPONSE, DOMAIN
from .coordinator import LLMPresenceCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up LLM Presence sensors from a config entry."""
    coordinator: LLMPresenceCoordinator = hass.data[DOMAIN][entry.entry_id]
    
    entities: list[SensorEntity] = []
    
    # Create a sensor for each person
    for person in coordinator.persons:
        entities.append(
            LLMPresenceSensor(
                coordinator=coordinator,
                entity_name=person.get("name", "unknown"),
                entity_type="person",
            )
        )
    
    # Create a sensor for each pet
    for pet in coordinator.pets:
        entities.append(
            LLMPresenceSensor(
                coordinator=coordinator,
                entity_name=pet.get("name", "unknown"),
                entity_type="pet",
            )
        )
    
    async_add_entities(entities)


class LLMPresenceSensor(CoordinatorEntity[LLMPresenceCoordinator], SensorEntity):
    """Sensor showing current room for a person or pet."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: LLMPresenceCoordinator,
        entity_name: str,
        entity_type: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entity_name = entity_name
        self._entity_type = entity_type
        
        # Set unique ID and name
        self._attr_unique_id = f"{DOMAIN}_{entity_type}_{entity_name.lower().replace(' ', '_')}_room"
        self._attr_name = f"{entity_name} Room"
        
        # Set icon based on type
        self._attr_icon = "mdi:account" if entity_type == "person" else "mdi:paw"

    @property
    def native_value(self) -> str | None:
        """Return the current room."""
        if not self.coordinator.data:
            return None
        
        data_key = "persons" if self._entity_type == "person" else "pets"
        entity_data = self.coordinator.data.get(data_key, {}).get(self._entity_name)
        
        if entity_data:
            return entity_data.room
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        if not self.coordinator.data:
            return {}
        
        data_key = "persons" if self._entity_type == "person" else "pets"
        entity_data = self.coordinator.data.get(data_key, {}).get(self._entity_name)
        
        if entity_data:
            return {
                ATTR_CONFIDENCE: entity_data.confidence,
                ATTR_RAW_RESPONSE: entity_data.raw_response,
                ATTR_INDICATORS: entity_data.indicators,
                "entity_type": self._entity_type,
            }
        return {}

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()

