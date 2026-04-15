"""Download sample dog photos and link them to the dogs in the database.

How the linking works
---------------------
Each dog row in the `dogs` table has two array columns:
  • photos[]      — full-size public URLs  (max 1280px, JPEG, <300KB)
  • thumbnails[]  — small public URLs      (max 320px, JPEG, <50KB)

Photos are stored in the Supabase Storage bucket `dog-photos` at:
  dogs/{dog_id}/photo_{n}.jpg
  dogs/{dog_id}/thumb_{n}.jpg

This script:
  1. Loads all dogs from the database.
  2. Downloads real dog photos from the Dog CEO public API (https://dog.ceo).
  3. Runs each through the existing optimize_image pipeline.
  4. Uploads full + thumbnail to Supabase Storage.
  5. Writes the resulting public URLs back to the dog record.

Usage
-----
  python scripts/seed_photos.py              # 2 photos per dog, skip dogs that already have photos
  python scripts/seed_photos.py --force      # re-upload even if photos already exist
  python scripts/seed_photos.py --count 3   # 3 photos per dog
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.dogs import list_available_dogs, update_dog
from src.models.schemas import DogUpdate
from src.services.photos import process_and_upload_dog_photo

logger = logging.getLogger(__name__)

DOG_IMAGE_API = "https://dog.ceo/api/breeds/image/random"


async def fetch_random_dog_image(client: httpx.AsyncClient) -> bytes:
    """Download one random dog photo from dog.ceo. Returns raw image bytes."""
    # Step 1: get a URL for a random dog photo
    api_resp = await client.get(DOG_IMAGE_API, timeout=15)
    api_resp.raise_for_status()
    image_url: str = api_resp.json()["message"]

    # Step 2: download the image itself
    img_resp = await client.get(image_url, timeout=30, follow_redirects=True)
    img_resp.raise_for_status()
    return img_resp.content


async def seed_photos(photos_per_dog: int = 2, force: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dogs = await list_available_dogs()
    if not dogs:
        logger.warning("No available dogs found — run seed_dogs.py first.")
        return

    logger.info("Found %d dogs. Uploading %d photo(s) each...\n", len(dogs), photos_per_dog)

    async with httpx.AsyncClient() as http:
        for dog in dogs:
            if dog.photos and not force:
                logger.info("⏭  %s already has %d photo(s) — skipping (use --force to re-upload)", dog.name, len(dog.photos))
                continue

            new_photos: list[str] = []
            new_thumbnails: list[str] = []

            for i in range(photos_per_dog):
                try:
                    logger.info("   Downloading photo %d/%d for %s...", i + 1, photos_per_dog, dog.name)
                    image_bytes = await fetch_random_dog_image(http)

                    full_url, thumb_url = await process_and_upload_dog_photo(
                        image_bytes,
                        dog_id=str(dog.id),
                        photo_index=i,
                    )
                    new_photos.append(full_url)
                    new_thumbnails.append(thumb_url)
                    logger.info("   ✓ Uploaded: %s", full_url)

                except Exception as exc:
                    logger.error("   ✗ Failed photo %d for %s: %s", i + 1, dog.name, exc)

            if new_photos:
                await update_dog(str(dog.id), DogUpdate(photos=new_photos, thumbnails=new_thumbnails))
                logger.info("✅ %s → %d photo(s) saved to DB\n", dog.name, len(new_photos))
            else:
                logger.warning("⚠  %s — no photos uploaded\n", dog.name)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed dog photos from dog.ceo API")
    parser.add_argument("--count", type=int, default=2, metavar="N", help="Photos per dog (default: 2)")
    parser.add_argument("--force", action="store_true", help="Re-upload even if photos already exist")
    args = parser.parse_args()

    asyncio.run(seed_photos(photos_per_dog=args.count, force=args.force))
