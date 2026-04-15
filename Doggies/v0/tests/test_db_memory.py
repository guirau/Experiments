"""Tests for src/db/memory.py — written BEFORE implementation (TDD)."""

from unittest.mock import patch
from uuid import uuid4

import pytest

from src.models.schemas import ConversationMemory, ConversationMemoryCreate
from tests.conftest import make_supabase_response


# ---------------------------------------------------------------------------
# save_conversation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_conversation_returns_memory_model(
    mock_supabase, sample_conversation_data, user_id
):
    mock_supabase.table.return_value.insert.return_value.execute.return_value = (
        make_supabase_response([sample_conversation_data])
    )

    with patch("src.db.memory.get_supabase_client", return_value=mock_supabase):
        from src.db.memory import save_conversation

        memory = await save_conversation(
            ConversationMemoryCreate(
                user_id=user_id,
                summary="User Emma is looking for a calm medium dog.",
                extracted_facts={"preferred_size": "medium"},
                embedding=[0.1] * 1536,
                messages_count=5,
            )
        )

    assert isinstance(memory, ConversationMemory)
    assert memory.summary == sample_conversation_data["summary"]


@pytest.mark.asyncio
async def test_save_conversation_calls_correct_table(
    mock_supabase, sample_conversation_data, user_id
):
    mock_supabase.table.return_value.insert.return_value.execute.return_value = (
        make_supabase_response([sample_conversation_data])
    )

    with patch("src.db.memory.get_supabase_client", return_value=mock_supabase):
        from src.db.memory import save_conversation

        await save_conversation(
            ConversationMemoryCreate(
                user_id=user_id,
                summary="Test summary",
                messages_count=1,
            )
        )

    mock_supabase.table.assert_called_with("conversations")


# ---------------------------------------------------------------------------
# get_recent_conversations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_recent_conversations_returns_list(
    mock_supabase, sample_conversation_data, user_id
):
    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = (
        make_supabase_response([sample_conversation_data])
    )

    with patch("src.db.memory.get_supabase_client", return_value=mock_supabase):
        from src.db.memory import get_recent_conversations

        memories = await get_recent_conversations(user_id, limit=5)

    assert isinstance(memories, list)
    assert len(memories) == 1
    assert isinstance(memories[0], ConversationMemory)


@pytest.mark.asyncio
async def test_get_recent_conversations_returns_empty_for_new_user(mock_supabase, user_id):
    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = (
        make_supabase_response([])
    )

    with patch("src.db.memory.get_supabase_client", return_value=mock_supabase):
        from src.db.memory import get_recent_conversations

        memories = await get_recent_conversations(user_id)

    assert memories == []


# ---------------------------------------------------------------------------
# search_similar_conversations (vector similarity)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_similar_conversations_returns_list(
    mock_supabase, sample_conversation_data, user_id
):
    rpc_result = {**sample_conversation_data, "similarity": 0.92}
    mock_supabase.rpc.return_value.execute.return_value = make_supabase_response([rpc_result])

    with patch("src.db.memory.get_supabase_client", return_value=mock_supabase):
        from src.db.memory import search_similar_conversations

        results = await search_similar_conversations(
            user_id=user_id,
            query_embedding=[0.1] * 1536,
            match_count=5,
        )

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], ConversationMemory)


@pytest.mark.asyncio
async def test_search_similar_conversations_calls_rpc(mock_supabase, user_id):
    mock_supabase.rpc.return_value.execute.return_value = make_supabase_response([])

    with patch("src.db.memory.get_supabase_client", return_value=mock_supabase):
        from src.db.memory import search_similar_conversations

        await search_similar_conversations(
            user_id=user_id,
            query_embedding=[0.0] * 1536,
            match_count=3,
        )

    mock_supabase.rpc.assert_called_once_with(
        "match_conversations",
        {
            "query_embedding": [0.0] * 1536,
            "match_user_id": user_id,
            "match_count": 3,
        },
    )


@pytest.mark.asyncio
async def test_search_similar_conversations_returns_empty_when_no_match(mock_supabase, user_id):
    mock_supabase.rpc.return_value.execute.return_value = make_supabase_response([])

    with patch("src.db.memory.get_supabase_client", return_value=mock_supabase):
        from src.db.memory import search_similar_conversations

        results = await search_similar_conversations(
            user_id=user_id,
            query_embedding=[0.0] * 1536,
        )

    assert results == []
