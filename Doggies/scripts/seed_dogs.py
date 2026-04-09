"""Populate the database with sample dogs from the PRD for development/demo purposes.

Usage:
    python scripts/seed_dogs.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.dogs import create_dog
from src.models.schemas import DogCreate

logger = logging.getLogger(__name__)

SAMPLE_DOGS: list[dict] = [
    {
        "name": "Mango",
        "breed": "Thai Ridgeback mix",
        "age_estimate": "~2 years",
        "size": "medium",
        "gender": "male",
        "temperament": ["calm", "loyal", "gentle"],
        "medical_notes": "Vaccinated, neutered",
        "story": (
            "Found wandering near Thong Sala pier, extremely sweet and well-behaved. "
            "Mango is the kind of dog who'll sit by your feet all day while you work "
            "and then join you for an evening walk. Zero drama, all love."
        ),
    },
    {
        "name": "Coconut",
        "breed": "Mixed breed",
        "age_estimate": "~6 months",
        "size": "small",
        "gender": "female",
        "temperament": ["playful", "curious", "energetic"],
        "medical_notes": "First vaccines done, will need booster",
        "story": (
            "Rescued as a tiny puppy from a construction site, full of joy. "
            "Coconut has never met a stranger — she'll zoom around your garden, "
            "steal your socks, and then fall asleep in your lap within minutes."
        ),
    },
    {
        "name": "Luna",
        "breed": "Lab mix",
        "age_estimate": "~3 years",
        "size": "large",
        "gender": "female",
        "temperament": ["friendly", "good_with_kids", "social"],
        "medical_notes": "Fully vaccinated, spayed",
        "story": (
            "Abandoned when her family left the island, she still trusts everyone. "
            "Luna is the dog who'll make your kids' childhood memorable. "
            "She's gentle, patient, and absolutely loves a good cuddle session."
        ),
    },
    {
        "name": "Diesel",
        "breed": "Thai Bangkaew mix",
        "age_estimate": "~4 years",
        "size": "large",
        "gender": "male",
        "temperament": ["protective", "loyal", "smart"],
        "medical_notes": "Vaccinated, neutered",
        "story": (
            "Street dog who chose the shelter — literally walked in one day and never left. "
            "Diesel guards it like it's his home, because it is. He's not aggressive, "
            "just deeply bonded and will be the same with your family."
        ),
    },
    {
        "name": "Noodle",
        "breed": "Mixed breed",
        "age_estimate": "~1 year",
        "size": "small",
        "gender": "female",
        "temperament": ["shy", "sweet", "cuddly"],
        "medical_notes": "Recovered from injury, fully healthy now, vaccinated",
        "story": (
            "Found injured and scared, slowly becoming the most affectionate dog at the shelter. "
            "Noodle needs a patient home with a calm environment. "
            "Once she trusts you, she'll be your shadow forever."
        ),
    },
    {
        "name": "Sunny",
        "breed": "Golden mix",
        "age_estimate": "~2 years",
        "size": "medium",
        "gender": "male",
        "temperament": ["happy", "energetic", "loves_water"],
        "medical_notes": "Vaccinated, neutered",
        "story": (
            "Found on the beach, lives for swimming and fetch. "
            "Sunny is pure serotonin. If you have access to the ocean or a pool "
            "and an active lifestyle, you've already found your match."
        ),
    },
    {
        "name": "Ghost",
        "breed": "White mixed breed",
        "age_estimate": "~5 years",
        "size": "medium",
        "gender": "male",
        "temperament": ["calm", "independent", "quiet"],
        "medical_notes": "Vaccinated, neutered",
        "story": (
            "Senior boy who's been at the shelter longest — dignified, gentle, and wise. "
            "Ghost doesn't need constant attention. He'll be your calm presence, "
            "happy to nap in the sun while you work and join you for a quiet evening stroll."
        ),
    },
    {
        "name": "Pepper",
        "breed": "Mixed breed",
        "age_estimate": "~8 months",
        "size": "small",
        "gender": "female",
        "temperament": ["bold", "clever", "mischievous"],
        "medical_notes": "Vaccinated",
        "story": (
            "Tiny but fearless — escapes every enclosure and steals everyone's food. "
            "Pepper is a project, but the best kind. She's insanely smart and will keep you "
            "on your toes. If you want a dog with a personality bigger than her body, here she is."
        ),
    },
]


async def seed() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Seeding %d dogs...", len(SAMPLE_DOGS))

    for dog_dict in SAMPLE_DOGS:
        try:
            dog = await create_dog(DogCreate(**dog_dict))
            logger.info("Created: %s (id=%s)", dog.name, dog.id)
        except Exception as exc:
            logger.error("Failed to create %s: %s", dog_dict["name"], exc)

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(seed())
