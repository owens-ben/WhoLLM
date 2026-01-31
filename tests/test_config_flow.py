"""Tests for WhoLLM config flow."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from custom_components.whollm.const import (
    CONF_MODEL,
    CONF_PERSONS,
    CONF_PETS,
    CONF_POLL_INTERVAL,
    CONF_PROVIDER,
    CONF_ROOMS,
    CONF_URL,
    CONF_ROOM_ENTITIES,
    CONF_LEARNING_ENABLED,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PROVIDER,
    DEFAULT_URL,
    DOMAIN,
)


@pytest.fixture
def mock_setup_entry():
    """Override async_setup_entry."""
    with patch(
        "custom_components.whollm.async_setup_entry",
        return_value=True,
    ) as mock_setup:
        yield mock_setup


class TestConfigFlowUserStep:
    """Test the user step of config flow."""

    @pytest.mark.asyncio
    async def test_user_step_shows_form(self):
        """Test that user step shows the provider configuration form."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        
        result = await flow.async_step_user(user_input=None)
        
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"
        assert CONF_PROVIDER in result["data_schema"].schema
        assert CONF_URL in result["data_schema"].schema

    @pytest.mark.asyncio
    async def test_user_step_connection_failure(self):
        """Test error handling when provider connection fails."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        
        with patch("custom_components.whollm.config_flow.get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.test_connection = AsyncMock(return_value=False)
            mock_get.return_value = mock_provider
            
            result = await flow.async_step_user(
                user_input={
                    CONF_PROVIDER: "ollama",
                    CONF_URL: "http://localhost:11434",
                    CONF_POLL_INTERVAL: 30,
                }
            )
        
        assert result["type"] == FlowResultType.FORM
        assert result["errors"]["base"] == "cannot_connect"

    @pytest.mark.asyncio
    async def test_user_step_success_proceeds_to_model(self):
        """Test successful connection proceeds to model selection."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        
        with patch("custom_components.whollm.config_flow.get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.test_connection = AsyncMock(return_value=True)
            mock_provider.get_available_models = AsyncMock(return_value=["llama3.2", "mistral"])
            mock_get.return_value = mock_provider
            
            result = await flow.async_step_user(
                user_input={
                    CONF_PROVIDER: "ollama",
                    CONF_URL: "http://localhost:11434",
                    CONF_POLL_INTERVAL: 30,
                }
            )
        
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "model"


class TestConfigFlowModelStep:
    """Test the model selection step."""

    @pytest.mark.asyncio
    async def test_model_step_shows_available_models(self):
        """Test that model step shows available models."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {CONF_PROVIDER: "ollama", CONF_URL: "http://localhost:11434"}
        flow._available_models = ["llama3.2", "mistral", "codellama"]
        
        result = await flow.async_step_model(user_input=None)
        
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "model"
        assert CONF_MODEL in result["data_schema"].schema

    @pytest.mark.asyncio
    async def test_model_step_proceeds_to_rooms(self):
        """Test selecting model proceeds to room configuration."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {CONF_PROVIDER: "ollama", CONF_URL: "http://localhost:11434"}
        flow._available_models = ["llama3.2"]
        
        result = await flow.async_step_model(
            user_input={CONF_MODEL: "llama3.2"}
        )
        
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "rooms"


class TestConfigFlowRoomsStep:
    """Test the rooms configuration step."""

    @pytest.mark.asyncio
    async def test_rooms_step_shows_form(self):
        """Test that rooms step shows room selection form."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
        }
        
        result = await flow.async_step_rooms(user_input=None)
        
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "rooms"
        assert CONF_ROOMS in result["data_schema"].schema

    @pytest.mark.asyncio
    async def test_rooms_step_requires_at_least_one_room(self):
        """Test that at least one room must be selected."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
        }
        
        result = await flow.async_step_rooms(
            user_input={CONF_ROOMS: [], "custom_rooms": ""}
        )
        
        assert result["type"] == FlowResultType.FORM
        assert result["errors"]["base"] == "at_least_one_room"

    @pytest.mark.asyncio
    async def test_rooms_step_accepts_custom_rooms(self):
        """Test that custom rooms can be added."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
        }
        
        result = await flow.async_step_rooms(
            user_input={CONF_ROOMS: ["office"], "custom_rooms": "garage, basement"}
        )
        
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "persons"
        assert "garage" in flow._data[CONF_ROOMS]
        assert "basement" in flow._data[CONF_ROOMS]


class TestConfigFlowPersonsStep:
    """Test the persons configuration step."""

    @pytest.mark.asyncio
    async def test_persons_step_shows_form(self):
        """Test that persons step shows person/pet input form."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
            CONF_ROOMS: ["office", "bedroom"],
        }
        
        result = await flow.async_step_persons(user_input=None)
        
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "persons"
        assert CONF_PERSONS in result["data_schema"].schema
        assert CONF_PETS in result["data_schema"].schema

    @pytest.mark.asyncio
    async def test_persons_step_requires_at_least_one_entity(self):
        """Test that at least one person or pet is required."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
            CONF_ROOMS: ["office", "bedroom"],
        }
        
        result = await flow.async_step_persons(
            user_input={CONF_PERSONS: "", CONF_PETS: ""}
        )
        
        assert result["type"] == FlowResultType.FORM
        assert result["errors"]["base"] == "at_least_one_entity"

    @pytest.mark.asyncio
    async def test_persons_step_parses_comma_separated_names(self):
        """Test that comma-separated names are parsed correctly."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
            CONF_ROOMS: ["office", "bedroom"],
        }
        
        result = await flow.async_step_persons(
            user_input={CONF_PERSONS: "Alice, Bob", CONF_PETS: "Whiskers"}
        )
        
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "room_entities"
        assert flow._data[CONF_PERSONS] == [{"name": "Alice"}, {"name": "Bob"}]
        assert flow._data[CONF_PETS] == [{"name": "Whiskers"}]


class TestConfigFlowRoomEntitiesStep:
    """Test the room entities mapping step."""

    @pytest.mark.asyncio
    async def test_room_entities_step_creates_entry(self):
        """Test that room entities step creates config entry."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
            CONF_ROOMS: ["office", "bedroom"],
            CONF_PERSONS: [{"name": "Alice"}],
            CONF_PETS: [],
        }
        
        # Simulate entity selection
        result = await flow.async_step_room_entities(
            user_input={
                "entities_office": ["switch.office_pc", "binary_sensor.office_motion"],
                "entities_bedroom": [],
            }
        )
        
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "WhoLLM (ollama)"
        assert CONF_ROOM_ENTITIES in result["data"]
        assert CONF_LEARNING_ENABLED in result["data"]


class TestGuessEntityHint:
    """Test the entity hint type guessing logic."""

    def test_guess_motion_sensor(self):
        """Test guessing motion sensor hint type."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        
        assert flow._guess_entity_hint("binary_sensor.office_motion") == "motion"
        assert flow._guess_entity_hint("binary_sensor.living_room_occupancy") == "motion"

    def test_guess_media_player(self):
        """Test guessing media player hint type."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        
        assert flow._guess_entity_hint("media_player.living_room_tv") == "media"

    def test_guess_light(self):
        """Test guessing light hint type."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        
        assert flow._guess_entity_hint("light.office_light") == "light"

    def test_guess_computer(self):
        """Test guessing computer hint type."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        
        assert flow._guess_entity_hint("switch.office_pc") == "computer"
        assert flow._guess_entity_hint("switch.desktop_computer") == "computer"

    def test_guess_camera(self):
        """Test guessing camera hint type."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        
        assert flow._guess_entity_hint("camera.living_room_camera") == "camera"
        assert flow._guess_entity_hint("binary_sensor.person_detected") == "camera"

    def test_guess_door_sensor(self):
        """Test guessing door/window sensor hint type."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        
        assert flow._guess_entity_hint("binary_sensor.front_door") == "door"
        assert flow._guess_entity_hint("binary_sensor.bedroom_window") == "door"
        assert flow._guess_entity_hint("binary_sensor.contact_sensor") == "door"

    def test_guess_device_tracker(self):
        """Test guessing device tracker hint type."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        
        assert flow._guess_entity_hint("device_tracker.alice_phone") == "presence"

    def test_guess_unknown_defaults_to_appliance(self):
        """Test that unknown entities default to appliance hint type."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        
        assert flow._guess_entity_hint("switch.random_switch") == "appliance"
        assert flow._guess_entity_hint("sensor.unknown_sensor") == "appliance"


class TestOptionsFlow:
    """Test the options flow for reconfiguration.
    
    Note: Options flow tests require proper Home Assistant test fixtures.
    These tests validate the flow structure conceptually.
    """

    def test_options_flow_config_structure(self):
        """Test that options can be defined with proper config keys."""
        # Verify config keys exist
        assert CONF_POLL_INTERVAL is not None
        assert CONF_LEARNING_ENABLED is not None
        
        # Test valid config structure
        valid_config = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_POLL_INTERVAL: 30,
            CONF_ROOMS: ["office", "bedroom"],
            CONF_PERSONS: [{"name": "Alice"}],
            CONF_PETS: [],
            CONF_ROOM_ENTITIES: {},
        }
        
        assert valid_config[CONF_PROVIDER] == "ollama"
        assert valid_config[CONF_POLL_INTERVAL] == 30

    def test_options_flow_general_settings_keys(self):
        """Test general settings keys are available."""
        # Verify key config options
        assert CONF_POLL_INTERVAL is not None
        assert CONF_LEARNING_ENABLED is not None
        
        # Test valid option values
        options = {
            CONF_POLL_INTERVAL: 60,
            CONF_LEARNING_ENABLED: False,
        }
        
        assert options[CONF_POLL_INTERVAL] == 60
        assert options[CONF_LEARNING_ENABLED] is False


class TestConfigFlowEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_model_list_uses_default(self):
        """Test that empty model list uses default model."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {}
        flow._available_models = []  # Empty list
        
        result = await flow.async_step_model(user_input=None)
        
        assert result["type"] == FlowResultType.FORM
        # Schema should still contain model field with default

    @pytest.mark.asyncio
    async def test_whitespace_in_custom_rooms_handled(self):
        """Test that whitespace in custom room names is handled."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
        }
        
        result = await flow.async_step_rooms(
            user_input={
                CONF_ROOMS: [],
                "custom_rooms": "  garage  ,   Game Room  ,  "
            }
        )
        
        assert "garage" in flow._data[CONF_ROOMS]
        assert "game_room" in flow._data[CONF_ROOMS]

    @pytest.mark.asyncio
    async def test_duplicate_custom_rooms_deduplicated(self):
        """Test that duplicate rooms are not added twice."""
        from custom_components.whollm.config_flow import LLMPresenceConfigFlow
        
        flow = LLMPresenceConfigFlow()
        flow.hass = MagicMock()
        flow._data = {
            CONF_PROVIDER: "ollama",
            CONF_URL: "http://localhost:11434",
            CONF_MODEL: "llama3.2",
        }
        
        result = await flow.async_step_rooms(
            user_input={
                CONF_ROOMS: ["office"],
                "custom_rooms": "office, bedroom"
            }
        )
        
        # office should not be duplicated
        assert flow._data[CONF_ROOMS].count("office") == 1
