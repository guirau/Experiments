"""Shared pytest fixtures for all test modules."""

import io
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.models.schemas import (
    Booking,
    ConversationMemory,
    Dog,
    DogCreate,
    User,
    UserCreate,
)


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------

@pytest.fixture
def dog_id() -> str:
    return str(uuid4())


@pytest.fixture
def user_id() -> str:
    return str(uuid4())


# ---------------------------------------------------------------------------
# Sample data dicts (as returned by Supabase .data)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dog_data(dog_id: str) -> dict:
    return {
        "id": dog_id,
        "name": "Mango",
        "breed": "Thai Ridgeback mix",
        "age_estimate": "~2 years",
        "size": "medium",
        "gender": "male",
        "temperament": ["calm", "loyal", "gentle"],
        "medical_notes": "Vaccinated, neutered",
        "story": "Found wandering near Thong Sala pier, extremely sweet",
        "photos": [],
        "thumbnails": [],
        "status": "available",
        "intake_date": None,
        "instagram_post_text": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_user_data(user_id: str) -> dict:
    return {
        "id": user_id,
        "telegram_id": 123456789,
        "telegram_username": "emma_nomad",
        "name": "Emma",
        "language": "en",
        "living_situation": "villa with garden",
        "location": "Koh Phangan, 6 months",
        "experience_with_dogs": "experienced",
        "lifestyle_notes": "works from home, no kids",
        "preferences": {"size": "medium", "energy": "calm"},
        "funnel_stage": "exploring",
        "liked_dog_ids": [],
        "intent": "adopt",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_conversation_data(user_id: str) -> dict:
    return {
        "id": str(uuid4()),
        "user_id": user_id,
        "summary": "User Emma is looking for a calm medium dog. Likes Mango.",
        "extracted_facts": {"preferred_size": "medium", "lifestyle": "work from home"},
        "embedding": [0.1] * 1536,
        "messages_count": 5,
        "created_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Pydantic model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dog(sample_dog_data: dict) -> Dog:
    return Dog(**sample_dog_data)


@pytest.fixture
def sample_user(sample_user_data: dict) -> User:
    return User(**sample_user_data)


@pytest.fixture
def sample_dog_create() -> DogCreate:
    return DogCreate(
        name="Mango",
        breed="Thai Ridgeback mix",
        age_estimate="~2 years",
        size="medium",
        gender="male",
        temperament=["calm", "loyal", "gentle"],
        medical_notes="Vaccinated, neutered",
        story="Found wandering near Thong Sala pier",
    )


@pytest.fixture
def sample_user_create() -> UserCreate:
    return UserCreate(
        telegram_id=123456789,
        telegram_username="emma_nomad",
        name="Emma",
    )


# ---------------------------------------------------------------------------
# Supabase mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_supabase() -> MagicMock:
    """
    Mock Supabase client supporting chainable query builder pattern:
        client.table("x").select("*").execute()
        client.table("x").insert({}).execute()
        client.table("x").update({}).eq("id", x).execute()
        client.table("x").delete().eq("id", x).execute()
        client.rpc("fn", {}).execute()
    """
    return MagicMock()


def make_supabase_response(data: list) -> MagicMock:
    """Build a mock APIResponse with a .data attribute."""
    response = MagicMock()
    response.data = data
    return response


# ---------------------------------------------------------------------------
# Image fixture (Pillow)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image_bytes() -> bytes:
    """Return a minimal valid JPEG byte string for photo tests."""
    try:
        from PIL import Image

        img = Image.new("RGB", (1600, 1200), color=(200, 150, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except ImportError:
        pytest.skip("Pillow not installed")
