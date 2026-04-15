from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dog
# ---------------------------------------------------------------------------

class Dog(BaseModel):
    id: UUID
    name: str
    breed: str
    age_estimate: str
    size: str  # small | medium | large
    gender: str  # male | female | unknown
    temperament: list[str] = []
    medical_notes: str | None = None
    story: str | None = None
    photos: list[str] = []
    thumbnails: list[str] = []
    status: str = "available"  # available | reserved | adopted
    intake_date: datetime | None = None
    instagram_post_text: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DogCreate(BaseModel):
    name: str
    breed: str
    age_estimate: str
    size: str
    gender: str
    temperament: list[str] = []
    medical_notes: str | None = None
    story: str | None = None
    status: str = "available"


class DogUpdate(BaseModel):
    name: str | None = None
    breed: str | None = None
    age_estimate: str | None = None
    size: str | None = None
    gender: str | None = None
    temperament: list[str] | None = None
    medical_notes: str | None = None
    story: str | None = None
    photos: list[str] | None = None
    thumbnails: list[str] | None = None
    status: str | None = None
    instagram_post_text: str | None = None


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User(BaseModel):
    id: UUID
    telegram_id: int
    telegram_username: str | None = None
    name: str | None = None
    language: str | None = None
    living_situation: str | None = None
    location: str | None = None
    experience_with_dogs: str | None = None
    lifestyle_notes: str | None = None
    preferences: dict = Field(default_factory=dict)
    funnel_stage: str = "curious"  # curious | exploring | interested | ready | adopted | donor
    liked_dog_ids: list[str] = []
    intent: str = "unknown"  # adopt | donate | both | unknown
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UserCreate(BaseModel):
    telegram_id: int
    telegram_username: str | None = None
    name: str | None = None


class UserUpdate(BaseModel):
    telegram_username: str | None = None
    name: str | None = None
    language: str | None = None
    living_situation: str | None = None
    location: str | None = None
    experience_with_dogs: str | None = None
    lifestyle_notes: str | None = None
    preferences: dict | None = None
    funnel_stage: str | None = None
    liked_dog_ids: list[str] | None = None
    intent: str | None = None


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

class ConversationMemory(BaseModel):
    id: UUID
    user_id: UUID
    summary: str
    extracted_facts: dict = Field(default_factory=dict)
    embedding: list[float] | None = None
    messages_count: int = 0
    created_at: datetime | None = None


class ConversationMemoryCreate(BaseModel):
    user_id: UUID
    summary: str
    extracted_facts: dict = Field(default_factory=dict)
    embedding: list[float] | None = None
    messages_count: int = 0


# ---------------------------------------------------------------------------
# Booking
# ---------------------------------------------------------------------------

class Booking(BaseModel):
    id: UUID
    user_id: UUID
    dog_id: UUID
    scheduled_at: datetime
    google_calendar_event_id: str | None = None
    status: str = "scheduled"  # scheduled | completed | cancelled | no_show
    admin_notified: bool = False
    created_at: datetime | None = None


class BookingCreate(BaseModel):
    user_id: UUID
    dog_id: UUID
    scheduled_at: datetime
    google_calendar_event_id: str | None = None
