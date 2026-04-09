import logging

from src.db.client import get_supabase_client
from src.models.schemas import ConversationMemory, ConversationMemoryCreate

logger = logging.getLogger(__name__)


async def save_conversation(memory_data: ConversationMemoryCreate) -> ConversationMemory:
    client = get_supabase_client()
    payload = memory_data.model_dump(exclude_none=True)
    # Convert UUID to string for Supabase
    payload["user_id"] = str(payload["user_id"])
    response = client.table("conversations").insert(payload).execute()
    return ConversationMemory(**response.data[0])


async def get_recent_conversations(
    user_id: str,
    limit: int = 10,
) -> list[ConversationMemory]:
    client = get_supabase_client()
    response = (
        client.table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [ConversationMemory(**row) for row in response.data]


async def search_similar_conversations(
    user_id: str,
    query_embedding: list[float],
    match_count: int = 5,
) -> list[ConversationMemory]:
    client = get_supabase_client()
    response = client.rpc(
        "match_conversations",
        {
            "query_embedding": query_embedding,
            "match_user_id": user_id,
            "match_count": match_count,
        },
    ).execute()

    return [ConversationMemory(**row) for row in response.data]
