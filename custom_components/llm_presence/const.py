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

# Vision configuration keys
CONF_VISION_ENABLED = "vision_enabled"
CONF_VISION_MODEL = "vision_model"
CONF_VISION_CAMERAS = "vision_cameras"  # List of camera entity IDs to use
CONF_VISION_ON_DETECTION = "vision_on_detection"  # Trigger vision on AI detection
CONF_AUTO_TRACKING = "auto_tracking"  # Enable/disable camera tracking control

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
DEFAULT_VISION_MODEL = "moondream"  # Faster than llava:7b (~5-15s vs ~50-60s)
DEFAULT_VISION_ENABLED = False

# Entity types
ENTITY_TYPE_PERSON = "person"
ENTITY_TYPE_PET = "pet"

# Valid room states
VALID_ROOMS = ["office", "living_room", "bedroom", "kitchen", "bathroom", "entry", "away", "unknown"]

# Attributes
ATTR_CONFIDENCE = "confidence"
ATTR_LAST_SEEN = "last_seen"
ATTR_INDICATORS = "indicators"
ATTR_RAW_RESPONSE = "raw_response"
ATTR_VISION_IDENTIFIED = "vision_identified"
ATTR_VISION_CONFIDENCE = "vision_confidence"

# Storage management
CONF_RETENTION_DAYS = "retention_days"
CONF_MAX_FILE_SIZE_MB = "max_file_size_mb"
DEFAULT_RETENTION_DAYS = 30
DEFAULT_MAX_FILE_SIZE_MB = 100


