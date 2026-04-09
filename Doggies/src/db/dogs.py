import logging

from src.db.client import get_supabase_client
from src.models.schemas import Dog, DogCreate, DogUpdate

logger = logging.getLogger(__name__)


async def create_dog(dog_data: DogCreate) -> Dog:
    client = get_supabase_client()
    payload = dog_data.model_dump(exclude_none=True)
    response = client.table("dogs").insert(payload).execute()
    return Dog(**response.data[0])


async def get_dog_by_id(dog_id: str) -> Dog | None:
    client = get_supabase_client()
    response = client.table("dogs").select("*").eq("id", dog_id).execute()
    if not response.data:
        return None
    return Dog(**response.data[0])


async def search_dogs(
    size: str | None = None,
    temperament: list[str] | None = None,
    status: str = "available",
) -> list[Dog]:
    client = get_supabase_client()
    query = client.table("dogs").select("*").eq("status", status)

    if size:
        query = query.eq("size", size)

    if temperament:
        query = query.contains("temperament", temperament)

    response = query.execute()
    return [Dog(**row) for row in response.data]


async def list_available_dogs() -> list[Dog]:
    return await search_dogs(status="available")


async def get_dog_by_name(name: str) -> Dog | None:
    client = get_supabase_client()
    response = client.table("dogs").select("*").ilike("name", name).execute()
    if not response.data:
        return None
    return Dog(**response.data[0])


async def update_dog(dog_id: str, update_data: DogUpdate) -> Dog:
    client = get_supabase_client()
    payload = update_data.model_dump(exclude_none=True)
    response = client.table("dogs").update(payload).eq("id", dog_id).execute()
    return Dog(**response.data[0])
