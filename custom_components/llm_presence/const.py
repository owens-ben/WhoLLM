"""Constants for LLM Room Presence integration."""

DOMAIN = "llm_presence"

# Configuration keys
CONF_PROVIDER = "provider"
CONF_URL = "url"
CONF_MODEL = "model"
CONF_POLL_INTERVAL = "poll_interval"
CONF_PERSONS = "persons"
CONF_PETS = "pets"
CONF_ROOMS = "rooms"

# Provider types (Ollama only for now, others stubbed)
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"  # Stubbed
PROVIDER_ANTHROPIC = "anthropic"  # Stubbed
PROVIDER_LOCAL = "local"  # Stubbed

SUPPORTED_PROVIDERS = [PROVIDER_OLLAMA]  # Only Ollama enabled for now
ALL_PROVIDERS = [PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_LOCAL]

# Defaults
DEFAULT_PROVIDER = PROVIDER_OLLAMA
DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_POLL_INTERVAL = 10  # seconds

# Entity types
ENTITY_TYPE_PERSON = "person"
ENTITY_TYPE_PET = "pet"

# Valid room states
VALID_ROOMS = ["office", "living_room", "bedroom", "kitchen", "bathroom", "away", "unknown"]

# Attributes
ATTR_CONFIDENCE = "confidence"
ATTR_LAST_SEEN = "last_seen"
ATTR_INDICATORS = "indicators"
ATTR_RAW_RESPONSE = "raw_response"


