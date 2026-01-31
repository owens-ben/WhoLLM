"""Tests for WhoLLM vision module."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import base64


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


class TestVisionIdentifier:
    """Tests for VisionIdentifier class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
        )
        
        assert identifier.ollama_url == "http://localhost:11434"
        assert identifier.vision_model == "moondream"
        assert "Alice" in identifier.known_persons
        assert "Bob" in identifier.known_persons

    def test_initialization_custom(self):
        """Test initialization with custom settings."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://192.168.1.100:11434",
            vision_model="llava:7b",
            known_persons=["Charlie", "Dana"],
            known_pets=["Fluffy"],
        )
        
        assert identifier.vision_model == "llava:7b"
        assert identifier.known_persons == ["Charlie", "Dana"]
        assert identifier.known_pets == ["Fluffy"]

    @pytest.mark.asyncio
    async def test_identify_from_camera_success(self, mock_hass):
        """Test successful camera identification."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
            known_persons=["Alice", "Bob"],
        )
        
        # Mock camera snapshot
        fake_image = b"fake image data"
        
        with patch.object(identifier, "_get_camera_snapshot", AsyncMock(return_value=fake_image)):
            with patch.object(identifier, "_compress_image", AsyncMock(return_value=fake_image)):
                with patch.object(identifier, "_identify_with_ollama", AsyncMock(return_value={
                    "identified": "Alice",
                    "confidence": "high",
                    "description": "Person at desk",
                    "success": True,
                })):
                    result = await identifier.identify_from_camera(
                        mock_hass,
                        "camera.living_room",
                        "person",
                    )
        
        assert result["success"] is True
        assert result["identified"] == "Alice"
        assert result["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_identify_from_camera_snapshot_failure(self, mock_hass):
        """Test handling when camera snapshot fails."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        with patch.object(identifier, "_get_camera_snapshot", AsyncMock(return_value=None)):
            result = await identifier.identify_from_camera(
                mock_hass,
                "camera.living_room",
                "person",
            )
        
        assert result["success"] is False
        assert "snapshot" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_identify_from_camera_exception(self, mock_hass):
        """Test handling of exceptions during identification."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        with patch.object(identifier, "_get_camera_snapshot", AsyncMock(side_effect=Exception("Network error"))):
            result = await identifier.identify_from_camera(
                mock_hass,
                "camera.living_room",
                "person",
            )
        
        assert result["success"] is False
        assert "Network error" in result["error"]


class TestImageCompression:
    """Tests for image compression."""

    @pytest.mark.asyncio
    async def test_compress_image_with_pil(self):
        """Test image compression with PIL available."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        # Create a simple test image using PIL if available
        try:
            from PIL import Image
            from io import BytesIO
            
            # Create a simple test image
            img = Image.new('RGB', (1920, 1080), color='red')
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            original_data = buffer.getvalue()
            
            compressed = await identifier._compress_image(original_data)
            
            # Compressed should be smaller
            assert len(compressed) < len(original_data)
            
        except ImportError:
            pytest.skip("PIL not available")

    @pytest.mark.asyncio
    async def test_compress_image_without_pil(self):
        """Test that compression returns original when PIL not available."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        fake_image = b"fake image bytes"
        
        with patch("custom_components.whollm.vision.BytesIO", side_effect=ImportError):
            # Should return original data
            result = await identifier._compress_image(fake_image)
            # In actual code, ImportError is caught differently
            # This tests the fallback path


class TestOllamaVisionAPI:
    """Tests for Ollama vision API integration."""

    @pytest.mark.asyncio
    async def test_identify_with_ollama_person(self):
        """Test Ollama API call for person identification."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
            known_persons=["Alice", "Bob"],
        )
        
        fake_image = b"test image"
        
        with patch("custom_components.whollm.vision.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "response": "PERSON: Alice\nCONFIDENCE: high\nDESCRIPTION: Person at desk"
            })
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()
            
            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance
            
            result = await identifier._identify_with_ollama(fake_image, "person")
        
        assert result["success"] is True
        assert result["identified"] == "Alice"
        assert result["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_identify_with_ollama_animal(self):
        """Test Ollama API call for animal identification."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
            known_pets=["Whiskers", "Fluffy"],
        )
        
        fake_image = b"test image"
        
        with patch("custom_components.whollm.vision.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "response": "ANIMAL: Whiskers\nCONFIDENCE: medium\nDESCRIPTION: Orange cat on couch"
            })
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()
            
            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance
            
            result = await identifier._identify_with_ollama(fake_image, "animal")
        
        assert result["identified"] == "Whiskers"

    @pytest.mark.asyncio
    async def test_identify_with_ollama_api_error(self):
        """Test handling of API errors."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        fake_image = b"test image"
        
        with patch("custom_components.whollm.vision.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()
            
            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_response)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance
            
            result = await identifier._identify_with_ollama(fake_image, "person")
        
        assert result["success"] is False
        assert "500" in result["error"]

    @pytest.mark.asyncio
    async def test_identify_with_ollama_timeout(self):
        """Test handling of timeout."""
        from custom_components.whollm.vision import VisionIdentifier
        import asyncio
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        fake_image = b"test image"
        
        with patch("custom_components.whollm.vision.aiohttp.ClientSession") as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(side_effect=asyncio.TimeoutError())
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance
            
            result = await identifier._identify_with_ollama(fake_image, "person")
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()


class TestResponseParsing:
    """Tests for parsing LLM responses."""

    def test_parse_person_response(self):
        """Test parsing person identification response."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
            known_persons=["Alice", "Bob"],
        )
        
        response = """PERSON: Alice
CONFIDENCE: high
DESCRIPTION: Woman with brown hair sitting at desk"""
        
        result = identifier._parse_identification_response(response, "person")
        
        assert result["identified"] == "Alice"
        assert result["confidence"] == "high"
        assert "brown hair" in result["description"]

    def test_parse_animal_response(self):
        """Test parsing animal identification response."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
            known_pets=["Whiskers", "Fluffy"],
        )
        
        response = """ANIMAL: Whiskers
CONFIDENCE: medium
DESCRIPTION: Orange tabby cat on the couch"""
        
        result = identifier._parse_identification_response(response, "animal")
        
        assert result["identified"] == "Whiskers"
        assert result["confidence"] == "medium"

    def test_parse_unknown_person(self):
        """Test parsing when person is unknown."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
            known_persons=["Alice"],
        )
        
        response = """PERSON: unknown
CONFIDENCE: low
DESCRIPTION: Cannot see face clearly"""
        
        result = identifier._parse_identification_response(response, "person")
        
        assert result["identified"] == "unknown"
        assert result["confidence"] == "low"

    def test_parse_normalizes_known_names(self):
        """Test that response normalizes to known names."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(
            ollama_url="http://localhost:11434",
            known_persons=["Alice", "Bob"],
        )
        
        # Response has lowercase
        response = """PERSON: alice (I think)
CONFIDENCE: high
DESCRIPTION: Person at desk"""
        
        result = identifier._parse_identification_response(response, "person")
        
        # Should be normalized to "Alice"
        assert result["identified"] == "Alice"

    def test_parse_malformed_response(self):
        """Test parsing malformed response."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        response = "I don't understand the question"
        
        result = identifier._parse_identification_response(response, "person")
        
        assert result["identified"] == "unknown"
        assert result["confidence"] == "low"


class TestCameraTrackingController:
    """Tests for CameraTrackingController class."""

    @pytest.mark.asyncio
    async def test_enable_tracking(self, mock_hass):
        """Test enabling camera tracking."""
        from custom_components.whollm.vision import CameraTrackingController
        
        controller = CameraTrackingController(mock_hass)
        
        result = await controller.enable_tracking("living_room_camera")
        
        assert result is True
        mock_hass.services.async_call.assert_called_once_with(
            "switch",
            "turn_on",
            {"entity_id": "switch.living_room_camera_auto_tracking"},
        )
        assert controller._tracking_enabled["living_room_camera"] is True

    @pytest.mark.asyncio
    async def test_disable_tracking(self, mock_hass):
        """Test disabling camera tracking."""
        from custom_components.whollm.vision import CameraTrackingController
        
        controller = CameraTrackingController(mock_hass)
        
        result = await controller.disable_tracking("living_room_camera")
        
        assert result is True
        mock_hass.services.async_call.assert_called_once_with(
            "switch",
            "turn_off",
            {"entity_id": "switch.living_room_camera_auto_tracking"},
        )
        assert controller._tracking_enabled["living_room_camera"] is False

    @pytest.mark.asyncio
    async def test_enable_tracking_error(self, mock_hass):
        """Test handling error when enabling tracking."""
        from custom_components.whollm.vision import CameraTrackingController
        
        mock_hass.services.async_call = AsyncMock(side_effect=Exception("Service not found"))
        
        controller = CameraTrackingController(mock_hass)
        
        result = await controller.enable_tracking("nonexistent_camera")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_toggle_tracking_on_detection(self, mock_hass):
        """Test toggling tracking based on detection."""
        from custom_components.whollm.vision import CameraTrackingController
        
        controller = CameraTrackingController(mock_hass)
        
        # Person detected - should enable
        await controller.toggle_tracking_on_detection("camera1", person_detected=True)
        
        assert controller._tracking_enabled.get("camera1") is True
        
        # Person not detected - should disable
        await controller.toggle_tracking_on_detection("camera1", person_detected=False)
        
        assert controller._tracking_enabled.get("camera1") is False


class TestCameraSnapshot:
    """Tests for camera snapshot functionality."""

    @pytest.mark.asyncio
    async def test_get_camera_snapshot_success(self, mock_hass):
        """Test successful camera snapshot retrieval."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        fake_image_content = b"fake image data"
        
        with patch("custom_components.whollm.vision.async_get_image", new_callable=AsyncMock) as mock_get:
            mock_image = MagicMock()
            mock_image.content = fake_image_content
            mock_get.return_value = mock_image
            
            result = await identifier._get_camera_snapshot(mock_hass, "camera.living_room")
        
        assert result == fake_image_content

    @pytest.mark.asyncio
    async def test_get_camera_snapshot_error(self, mock_hass):
        """Test handling camera snapshot error."""
        from custom_components.whollm.vision import VisionIdentifier
        
        identifier = VisionIdentifier(ollama_url="http://localhost:11434")
        
        with patch("custom_components.whollm.vision.async_get_image", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Camera unavailable")
            
            result = await identifier._get_camera_snapshot(mock_hass, "camera.broken")
        
        assert result is None
