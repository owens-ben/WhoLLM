"""Sensor platform for LLM Room Presence."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback, Event
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
    
    # Create vision identification sensor
    entities.append(LLMVisionSensor(hass, coordinator))
    
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


class LLMVisionSensor(SensorEntity):
    """Sensor showing last vision identification result."""

    _attr_has_entity_name = True
    _attr_unique_id = f"{DOMAIN}_vision_last_identification"
    _attr_name = "Vision Last Identification"
    _attr_icon = "mdi:camera-iris"

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: LLMPresenceCoordinator,
    ) -> None:
        """Initialize the vision sensor."""
        self.hass = hass
        self._coordinator = coordinator
        self._identified: str = "none"
        self._confidence: str = "none"
        self._description: str = ""
        self._camera: str = ""
        self._detection_type: str = ""
        self._timestamp: str = ""
        self._raw_response: str = ""
        
        # Listen for vision identification events
        self._unsub = None

    async def async_added_to_hass(self) -> None:
        """Register event listener when added to hass."""
        @callback
        def handle_vision_event(event: Event) -> None:
            """Handle vision identification event."""
            data = event.data
            self._identified = data.get("identified", "unknown")
            self._confidence = data.get("confidence", "low")
            self._description = data.get("description", "")
            self._camera = data.get("camera", "")
            self._detection_type = data.get("detection_type", "")
            self._timestamp = datetime.now().isoformat()
            self._raw_response = data.get("raw_response", "")
            
            _LOGGER.info(
                "Vision sensor updated: %s (%s) from %s",
                self._identified,
                self._confidence,
                self._camera
            )
            self.async_write_ha_state()
        
        self._unsub = self.hass.bus.async_listen(
            f"{DOMAIN}_vision_identification",
            handle_vision_event
        )

    async def async_will_remove_from_hass(self) -> None:
        """Unregister event listener when removed."""
        if self._unsub:
            self._unsub()

    @property
    def native_value(self) -> str:
        """Return the last identified person/pet."""
        return self._identified

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        return {
            "confidence": self._confidence,
            "description": self._description,
            "camera": self._camera,
            "detection_type": self._detection_type,
            "timestamp": self._timestamp,
            "raw_response": self._raw_response,
        }


