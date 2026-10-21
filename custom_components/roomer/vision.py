"""Vision-based person identification using Ollama vision models."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Known people/pets for identification
DEFAULT_KNOWN_ENTITIES = {
    "persons": ["Alice", "Bob"],
    "pets": ["Max"],
}


class VisionIdentifier:
    """Use Ollama vision models to identify people/pets in camera images."""

    def __init__(
        self,
        ollama_url: str,
        vision_model: str = "moondream",  # Default to faster model
        known_persons: list[str] | None = None,
        known_pets: list[str] | None = None,
    ) -> None:
        """Initialize the vision identifier.

        Vision model options:
        - moondream: Fastest (~5-15s), good accuracy, 1.7GB
        - llava:7b: Slower (~50-60s), better accuracy, 4.7GB
        """
        self.ollama_url = ollama_url
        self.vision_model = vision_model
        self.known_persons = known_persons or DEFAULT_KNOWN_ENTITIES["persons"]
        self.known_pets = known_pets or DEFAULT_KNOWN_ENTITIES["pets"]
        _LOGGER.info("VisionIdentifier initialized with model: %s", vision_model)

    async def identify_from_camera(
        self,
        hass: HomeAssistant,
        camera_entity_id: str,
        detection_type: str = "person",  # "person" or "animal"
    ) -> dict[str, Any]:
        """Capture image from camera and identify who/what is in it.

        Args:
            hass: Home Assistant instance
            camera_entity_id: Entity ID of the camera (e.g., camera.e1_zoom_fluent)
            detection_type: Whether we're looking for a person or animal

        Returns:
            Dict with identification results
        """
        try:
            # Get camera snapshot
            image_data = await self._get_camera_snapshot(hass, camera_entity_id)
            if not image_data:
                return {"success": False, "error": "Failed to get camera snapshot"}

            # Compress image for faster processing
            compressed_image = await self._compress_image(image_data)

            # Send to Ollama for identification
            result = await self._identify_with_ollama(compressed_image, detection_type)

            return result

        except Exception as err:
            _LOGGER.error("Error in vision identification: %s", err)
            return {"success": False, "error": str(err)}

    async def _get_camera_snapshot(
        self,
        hass: HomeAssistant,
        camera_entity_id: str,
    ) -> bytes | None:
        """Get a snapshot from the camera."""
        try:
            # Use Home Assistant's camera component to get image
            from homeassistant.components.camera import async_get_image

            image = await async_get_image(hass, camera_entity_id)
            return image.content

        except Exception as err:
            _LOGGER.error("Failed to get camera snapshot from %s: %s", camera_entity_id, err)
            return None

    async def _compress_image(
        self,
        image_data: bytes,
        max_size: tuple[int, int] = (512, 384),  # Smaller for faster processing
        quality: int = 75,  # Balance between quality and speed
    ) -> bytes:
        """Compress image for faster LLM processing."""
        try:
            from PIL import Image

            # Open image
            img = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize if larger than max_size
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Save as JPEG with compression
            output = BytesIO()
            img.save(output, format="JPEG", quality=quality, optimize=True)

            compressed = output.getvalue()
            _LOGGER.debug("Compressed image from %d bytes to %d bytes", len(image_data), len(compressed))
            return compressed

        except ImportError:
            _LOGGER.warning("PIL not available, using original image")
            return image_data
        except Exception as err:
            _LOGGER.warning("Failed to compress image: %s", err)
            return image_data

    async def _identify_with_ollama(
        self,
        image_data: bytes,
        detection_type: str,
    ) -> dict[str, Any]:
        """Send image to Ollama vision model for identification."""

        # Encode image as base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Build prompt based on detection type
        if detection_type == "person":
            known_list = ", ".join(self.known_persons)
            prompt = f"""Look at this camera image and identify the person you see.

Known household members: {known_list}

Based on physical appearance (hair, clothing, body type, etc.), who do you see?

Respond in this exact format:
PERSON: [name or "unknown"]
CONFIDENCE: [high/medium/low]
DESCRIPTION: [brief description of what you see]

If you cannot clearly see a person or identify them, respond with PERSON: unknown"""
        else:
            known_list = ", ".join(self.known_pets)
            prompt = f"""Look at this camera image and identify the animal/pet you see.

Known household pets: {known_list}

What animal do you see and can you identify which pet it is?

Respond in this exact format:
ANIMAL: [pet name or type like "cat", "dog", or "unknown"]
CONFIDENCE: [high/medium/low]
DESCRIPTION: [brief description of what you see]"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 150,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=60),  # Vision takes longer
                ) as response:
                    if response.status != 200:
                        return {"success": False, "error": f"Ollama API error: {response.status}"}

                    result = await response.json()
                    raw_response = result.get("response", "").strip()

                    # Parse the response
                    parsed = self._parse_identification_response(raw_response, detection_type)
                    parsed["raw_response"] = raw_response
                    parsed["success"] = True

                    _LOGGER.info("Vision identification result: %s", parsed)

                    return parsed

        except TimeoutError:
            return {"success": False, "error": "Vision model timeout"}
        except Exception as err:
            return {"success": False, "error": str(err)}

    def _parse_identification_response(
        self,
        response: str,
        detection_type: str,
    ) -> dict[str, Any]:
        """Parse the LLM's identification response."""
        result = {
            "identified": "unknown",
            "confidence": "low",
            "description": "",
        }

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if detection_type == "person" and line.upper().startswith("PERSON:"):
                result["identified"] = line.split(":", 1)[1].strip()
            elif detection_type != "person" and line.upper().startswith("ANIMAL:"):
                result["identified"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                result["confidence"] = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("DESCRIPTION:"):
                result["description"] = line.split(":", 1)[1].strip()

        # Normalize the identified name
        identified = result["identified"].lower()

        # Check if it matches any known person
        for person in self.known_persons:
            if person.lower() in identified:
                result["identified"] = person
                break

        # Check if it matches any known pet
        for pet in self.known_pets:
            if pet.lower() in identified:
                result["identified"] = pet
                break

        return result


class CameraTrackingController:
    """Control camera PTZ tracking based on presence detection."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the tracking controller."""
        self.hass = hass
        self._tracking_enabled: dict[str, bool] = {}

    async def enable_tracking(self, camera_name: str) -> bool:
        """Enable auto-tracking on a camera."""
        entity_id = f"switch.{camera_name}_auto_tracking"
        try:
            await self.hass.services.async_call(
                "switch",
                "turn_on",
                {"entity_id": entity_id},
            )
            self._tracking_enabled[camera_name] = True
            _LOGGER.info("Enabled tracking on %s", camera_name)
            return True
        except Exception as err:
            _LOGGER.error("Failed to enable tracking on %s: %s", camera_name, err)
            return False

    async def disable_tracking(self, camera_name: str) -> bool:
        """Disable auto-tracking on a camera."""
        entity_id = f"switch.{camera_name}_auto_tracking"
        try:
            await self.hass.services.async_call(
                "switch",
                "turn_off",
                {"entity_id": entity_id},
            )
            self._tracking_enabled[camera_name] = False
            _LOGGER.info("Disabled tracking on %s", camera_name)
            return True
        except Exception as err:
            _LOGGER.error("Failed to disable tracking on %s: %s", camera_name, err)
            return False

    async def toggle_tracking_on_detection(
        self,
        camera_name: str,
        person_detected: bool,
    ) -> None:
        """Toggle tracking based on person detection.

        Enable tracking when person detected, disable after timeout.
        """
        if person_detected:
            await self.enable_tracking(camera_name)
        else:
            # Could add a delay here before disabling
            await self.disable_tracking(camera_name)
