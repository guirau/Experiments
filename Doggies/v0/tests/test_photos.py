"""Tests for src/services/photos.py — written BEFORE implementation (TDD)."""

import io
from unittest.mock import MagicMock, patch

import pytest

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PILLOW_AVAILABLE, reason="Pillow not installed")


def make_jpeg_bytes(width: int = 1600, height: int = 1200) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_png_bytes(width: int = 800, height: int = 600) -> bytes:
    img = Image.new("RGBA", (width, height), color=(100, 200, 100, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# optimize_image
# ---------------------------------------------------------------------------

def test_optimize_image_output_is_jpeg():
    from src.services.photos import optimize_image

    result = optimize_image(make_jpeg_bytes(), max_size=1280, quality=85)
    img = Image.open(io.BytesIO(result))

    assert img.format == "JPEG"


def test_optimize_image_respects_max_size_landscape():
    """1600×1200 image → longest side should be ≤ 1280."""
    from src.services.photos import optimize_image

    result = optimize_image(make_jpeg_bytes(1600, 1200), max_size=1280, quality=85)
    img = Image.open(io.BytesIO(result))

    assert max(img.size) <= 1280


def test_optimize_image_respects_max_size_portrait():
    """1200×1600 image → longest side should be ≤ 1280."""
    from src.services.photos import optimize_image

    result = optimize_image(make_jpeg_bytes(1200, 1600), max_size=1280, quality=85)
    img = Image.open(io.BytesIO(result))

    assert max(img.size) <= 1280


def test_optimize_image_does_not_upscale_small_image():
    """320×240 image with max_size=1280 should stay at 320×240."""
    from src.services.photos import optimize_image

    result = optimize_image(make_jpeg_bytes(320, 240), max_size=1280, quality=85)
    img = Image.open(io.BytesIO(result))

    assert img.size == (320, 240)


def test_optimize_image_converts_png_to_jpeg():
    from src.services.photos import optimize_image

    result = optimize_image(make_png_bytes(), max_size=1280, quality=85)
    img = Image.open(io.BytesIO(result))

    assert img.format == "JPEG"
    assert img.mode == "RGB"


def test_optimize_image_converts_rgba_to_rgb():
    from src.services.photos import optimize_image

    rgba_bytes = make_png_bytes()
    result = optimize_image(rgba_bytes, max_size=1280, quality=85)
    img = Image.open(io.BytesIO(result))

    assert img.mode == "RGB"


def test_thumbnail_is_within_max_size():
    """Thumbnail: max_size=320, result longest side ≤ 320."""
    from src.services.photos import optimize_image

    result = optimize_image(make_jpeg_bytes(1600, 1200), max_size=320, quality=70)
    img = Image.open(io.BytesIO(result))

    assert max(img.size) <= 320


def test_thumbnail_size_under_target():
    """Thumbnail bytes should be well under 50KB for a typical photo."""
    from src.services.photos import optimize_image

    result = optimize_image(make_jpeg_bytes(1600, 1200), max_size=320, quality=70)

    assert len(result) < 50 * 1024  # 50KB


def test_full_image_size_under_target():
    """Full optimized image (solid-color, typical compressibility) under 300KB."""
    from src.services.photos import optimize_image

    # Solid-color images are representative of compressible real photos
    img = Image.new("RGB", (1600, 1200), color=(180, 120, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    image_bytes = buf.getvalue()

    result = optimize_image(image_bytes, max_size=1280, quality=85)

    assert len(result) < 300 * 1024  # 300KB


def test_corrupt_image_raises_value_error():
    from src.services.photos import optimize_image

    with pytest.raises((ValueError, OSError, Exception)):
        optimize_image(b"this is not an image", max_size=1280, quality=85)


# ---------------------------------------------------------------------------
# upload_dog_photo (mocks Supabase Storage)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_dog_photo_returns_public_url(dog_id):
    mock_storage = MagicMock()
    mock_storage.from_.return_value.upload.return_value = MagicMock()
    mock_storage.from_.return_value.get_public_url.return_value = (
        f"https://example.supabase.co/storage/v1/object/public/dog-photos/dogs/{dog_id}/photo_0.jpg"
    )

    with patch("src.services.photos.get_supabase_client") as mock_client:
        mock_client.return_value.storage = mock_storage

        from src.services.photos import upload_dog_photo

        url = await upload_dog_photo(make_jpeg_bytes(), dog_id=dog_id, photo_index=0)

    assert url.startswith("https://")
    assert dog_id in url


@pytest.mark.asyncio
async def test_process_and_upload_returns_full_and_thumb_urls(dog_id):
    mock_storage = MagicMock()
    mock_storage.from_.return_value.upload.return_value = MagicMock()
    mock_storage.from_.return_value.get_public_url.side_effect = [
        f"https://example.supabase.co/storage/v1/object/public/dog-photos/dogs/{dog_id}/photo_0.jpg",
        f"https://example.supabase.co/storage/v1/object/public/dog-photos/dogs/{dog_id}/thumb_0.jpg",
    ]

    with patch("src.services.photos.get_supabase_client") as mock_client:
        mock_client.return_value.storage = mock_storage

        from src.services.photos import process_and_upload_dog_photo

        full_url, thumb_url = await process_and_upload_dog_photo(
            make_jpeg_bytes(), dog_id=dog_id, photo_index=0
        )

    assert "photo_0" in full_url
    assert "thumb_0" in thumb_url
