"""Constants for WhoLLM integration."""

DOMAIN = "whollm"

# Configuration keys
CONF_PROVIDER = "provider"
CONF_URL = "url"
CONF_MODEL = "model"
CONF_POLL_INTERVAL = "poll_interval"
CONF_PERSONS = "persons"
CONF_PETS = "pets"
CONF_ROOMS = "rooms"

# Room-entity mapping configuration
# Maps rooms to lists of entities that indicate activity in that room
CONF_ROOM_ENTITIES = "room_entities"  # Dict[room_name, List[entity_id]]

# Entity type hints - what kind of signal does this entity provide?
ENTITY_HINT_MOTION = "motion"  # Motion sensors
ENTITY_HINT_MEDIA = "media"  # Media players (TV, speakers)
ENTITY_HINT_LIGHT = "light"  # Lights
ENTITY_HINT_COMPUTER = "computer"  # PC/workstation indicators
ENTITY_HINT_CAMERA = "camera"  # Camera with AI detection
ENTITY_HINT_CLIMATE = "climate"  # Temperature/humidity sensors
ENTITY_HINT_DOOR = "door"  # Door/window sensors
ENTITY_HINT_APPLIANCE = "appliance"  # Other appliances
ENTITY_HINT_PRESENCE = "presence"  # BLE/WiFi presence sensors

ENTITY_HINTS = [
    ENTITY_HINT_MOTION,
    ENTITY_HINT_MEDIA,
    ENTITY_HINT_LIGHT,
    ENTITY_HINT_COMPUTER,
    ENTITY_HINT_CAMERA,
    ENTITY_HINT_CLIMATE,
    ENTITY_HINT_DOOR,
    ENTITY_HINT_APPLIANCE,
    ENTITY_HINT_PRESENCE,
]

# Confidence weights for different entity types (can be overridden in options)
CONF_CONFIDENCE_WEIGHTS = "confidence_weights"
DEFAULT_CONFIDENCE_WEIGHTS = {
    ENTITY_HINT_CAMERA: 0.95,  # Camera AI detection is very reliable
    ENTITY_HINT_COMPUTER: 0.85,  # PC actively in use - very strong for office
    ENTITY_HINT_MEDIA: 0.80,  # Media playing is strong
    ENTITY_HINT_MOTION: 0.60,  # Motion sensor
    ENTITY_HINT_PRESENCE: 0.70,  # BLE/WiFi presence
    ENTITY_HINT_APPLIANCE: 0.50,  # Appliance in use
    ENTITY_HINT_LIGHT: 0.25,  # Light on - weak indicator alone
    ENTITY_HINT_DOOR: 0.40,  # Door recently opened
    ENTITY_HINT_CLIMATE: 0.20,  # Climate change - very weak
    "llm_reasoning": 0.50,  # LLM text reasoning
    "habit": 0.35,  # Habit-based prediction
}

# Person-device ownership mapping
# Maps person names to their "owned" entities (e.g., their PC, their phone tracker)
CONF_PERSON_DEVICES = "person_devices"  # Dict[person_name, List[entity_id]]

# Vision configuration keys
CONF_VISION_ENABLED = "vision_enabled"
CONF_VISION_MODEL = "vision_model"
CONF_VISION_CAMERAS = "vision_cameras"  # List of camera entity IDs to use
CONF_VISION_ON_DETECTION = "vision_on_detection"  # Trigger vision on AI detection
CONF_AUTO_TRACKING = "auto_tracking"  # Enable/disable camera tracking control

# Provider types
PROVIDER_OLLAMA = "ollama"
PROVIDER_CREWAI = "crewai"  # Uses CrewAI API with Claude/Ollama
PROVIDER_OPENAI = "openai"  # Stubbed
PROVIDER_ANTHROPIC = "anthropic"  # Stubbed
PROVIDER_LOCAL = "local"  # Stubbed

SUPPORTED_PROVIDERS = [PROVIDER_OLLAMA, PROVIDER_CREWAI]
ALL_PROVIDERS = [PROVIDER_OLLAMA, PROVIDER_CREWAI, PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_LOCAL]

# CrewAI defaults
DEFAULT_CREWAI_URL = "http://localhost:8502"
DEFAULT_CREWAI_MODEL = "sonnet"  # "haiku", "sonnet", or "opus"

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

# Valid room states (defaults, users can add custom rooms)
VALID_ROOMS = ["office", "living_room", "bedroom", "kitchen", "bathroom", "entry", "away", "unknown"]

# Special room values
ROOM_AWAY = "away"
ROOM_UNKNOWN = "unknown"

# Device tracker states
DEVICE_TRACKER_HOME = "home"
DEVICE_TRACKER_NOT_HOME = "not_home"
DEVICE_TRACKER_UNAVAILABLE = "unavailable"

# Confidence for device tracker based away detection
DEVICE_TRACKER_AWAY_CONFIDENCE = 0.95

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

# Learning configuration
CONF_LEARNING_ENABLED = "learning_enabled"
DEFAULT_LEARNING_ENABLED = True
