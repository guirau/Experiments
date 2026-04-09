import logging

from src.db.client import get_supabase_client
from src.models.schemas import User, UserCreate, UserUpdate

logger = logging.getLogger(__name__)


async def get_user_by_telegram_id(telegram_id: int) -> User | None:
    client = get_supabase_client()
    response = client.table("users").select("*").eq("telegram_id", telegram_id).execute()
    if not response.data:
        return None
    return User(**response.data[0])


async def get_user_by_id(user_id: str) -> User | None:
    client = get_supabase_client()
    response = client.table("users").select("*").eq("id", user_id).execute()
    if not response.data:
        return None
    return User(**response.data[0])


async def get_or_create_user(user_data: UserCreate) -> User:
    existing = await get_user_by_telegram_id(user_data.telegram_id)
    if existing:
        return existing

    client = get_supabase_client()
    payload = user_data.model_dump(exclude_none=True)
    response = client.table("users").insert(payload).execute()
    return User(**response.data[0])


async def update_user(user_id: str, update_data: UserUpdate) -> User:
    client = get_supabase_client()
    payload = update_data.model_dump(exclude_none=True)
    response = client.table("users").update(payload).eq("id", user_id).execute()
    return User(**response.data[0])


async def add_liked_dog(user_id: str, dog_id: str) -> User:
    user = await get_user_by_id(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")

    current_liked = [str(d) for d in user.liked_dog_ids]
    if dog_id in current_liked:
        return user

    updated_liked = current_liked + [dog_id]
    return await update_user(user_id, UserUpdate(liked_dog_ids=updated_liked))
