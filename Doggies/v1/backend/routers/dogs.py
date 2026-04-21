import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from database import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter()

# Fields returned by GET /api/dogs/{id} — excludes unused fields per PRD §4
DOG_DETAIL_FIELDS = (
    "id, name, breed, age_estimate, size, gender, "
    "temperament, medical_notes, story, photos, status, intake_date"
)


@router.get("/dogs")
def list_dogs() -> dict[str, list[dict[str, Any]]]:
    """GET /api/dogs — returns { dogs: [{ id, name }] }"""
    client = get_supabase_client()
    response = client.table("dogs").select("id, name").execute()
    return {"dogs": response.data}


@router.get("/dogs/{dog_id}")
def get_dog(dog_id: str) -> dict[str, Any]:
    """GET /api/dogs/{id} — returns full dog object or 404"""
    client = get_supabase_client()
    response = (
        client.table("dogs")
        .select(DOG_DETAIL_FIELDS)
        .eq("id", dog_id)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail=f"Dog {dog_id} not found")
    return response.data[0]
