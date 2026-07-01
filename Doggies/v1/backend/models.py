from datetime import datetime
from pydantic import BaseModel


class DogSummary(BaseModel):
    id: str
    name: str


class DogListResponse(BaseModel):
    dogs: list[DogSummary]


class DogDetail(BaseModel):
    id: str
    name: str
    breed: str | None = None
    age_estimate: str | None = None
    size: str | None = None
    gender: str | None = None
    temperament: list[str] | None = None
    medical_notes: str | None = None
    story: str | None = None
    photos: list[str] | None = None
    status: str | None = None
    intake_date: datetime | None = None
