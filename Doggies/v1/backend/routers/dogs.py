import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException
from postgrest.exceptions import APIError
from database import get_supabase_client
from models import DogDetail, DogListResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Fields returned by GET /api/dogs/{id} — excludes unused fields per PRD §4
DOG_DETAIL_FIELDS = (
    "id, name, breed, age_estimate, size, gender, "
    "temperament, medical_notes, story, photos, status, intake_date"
)


@router.get("/dogs")
def list_dogs() -> DogListResponse:
    """GET /api/dogs — returns { dogs: [{ id, name }] }"""
    client = get_supabase_client()
    try:
        response = client.table("dogs").select("id, name").execute()
    except APIError as exc:
        logger.error("Supabase error in list_dogs: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    return DogListResponse(dogs=response.data)


@router.get("/dogs/{dog_id}")
def get_dog(dog_id: UUID) -> DogDetail:
    """GET /api/dogs/{id} — returns full dog object or 404"""
    client = get_supabase_client()
    try:
        response = (
            client.table("dogs")
            .select(DOG_DETAIL_FIELDS)
            .eq("id", str(dog_id))
            .execute()
        )
    except APIError as exc:
        logger.error("Supabase error in get_dog(%s): %s", dog_id, exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    if not response.data:
        raise HTTPException(status_code=404, detail=f"Dog {dog_id} not found")
    return DogDetail(**response.data[0])
