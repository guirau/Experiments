"""Photo optimization and Supabase Storage upload.

Upload flow:
  1. optimize_image()       — resize, convert to JPEG, strip EXIF, compress
  2. upload_dog_photo()     — push bytes to Supabase Storage, return public URL
  3. process_and_upload_dog_photo() — full pipeline for one photo: full + thumb
"""

import io
import logging

from src.db.client import get_supabase_client

logger = logging.getLogger(__name__)

BUCKET = "dog-photos"


def optimize_image(image_bytes: bytes, max_size: int, quality: int) -> bytes:
    """Resize to max_size on the longest side, convert to RGB JPEG, strip EXIF.

    Args:
        image_bytes: Raw input image bytes (any format PIL can read).
        max_size: Maximum pixels on the longest side.
        quality: JPEG compression quality (1-95).

    Returns:
        Optimized JPEG bytes.

    Raises:
        OSError / ValueError: If image_bytes cannot be decoded.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB (handles RGBA, P-mode palette, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Strip EXIF by re-encoding without metadata
    buf_strip = io.BytesIO()
    img.save(buf_strip, format="JPEG", quality=95)
    clean = Image.open(buf_strip)

    # Resize only if the image exceeds max_size on its longest side
    if max(clean.size) > max_size:
        clean.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


async def upload_dog_photo(
    image_bytes: bytes,
    dog_id: str,
    photo_index: int,
    is_thumbnail: bool = False,
) -> str:
    """Upload image bytes to Supabase Storage and return the public URL.

    Path pattern:
        dogs/{dog_id}/photo_{n}.jpg  (full)
        dogs/{dog_id}/thumb_{n}.jpg  (thumbnail)
    """
    prefix = "thumb" if is_thumbnail else "photo"
    path = f"dogs/{dog_id}/{prefix}_{photo_index}.jpg"

    client = get_supabase_client()
    storage = client.storage

    storage.from_(BUCKET).upload(
        path=path,
        file=image_bytes,
        file_options={"content-type": "image/jpeg", "upsert": "true"},
    )

    public_url: str = storage.from_(BUCKET).get_public_url(path)
    logger.info("Uploaded %s → %s", path, public_url)
    return public_url


async def process_and_upload_dog_photo(
    image_bytes: bytes,
    dog_id: str,
    photo_index: int,
) -> tuple[str, str]:
    """Optimize, generate thumbnail, upload both, return (full_url, thumb_url)."""
    full_bytes = optimize_image(image_bytes, max_size=1280, quality=85)
    thumb_bytes = optimize_image(image_bytes, max_size=320, quality=70)

    full_url = await upload_dog_photo(full_bytes, dog_id=dog_id, photo_index=photo_index, is_thumbnail=False)
    thumb_url = await upload_dog_photo(thumb_bytes, dog_id=dog_id, photo_index=photo_index, is_thumbnail=True)

    return full_url, thumb_url
