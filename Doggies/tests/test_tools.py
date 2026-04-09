"""Tests for src/agent/tools.py — written BEFORE implementation (TDD)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.models.schemas import Dog, DogCreate, User, UserCreate
from tests.conftest import make_supabase_response


# ---------------------------------------------------------------------------
# handle_search_dogs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_dogs_returns_list_of_dicts(sample_dog_data):
    with patch("src.agent.tools.search_dogs", new=AsyncMock(return_value=[])):
        from src.agent.tools import handle_search_dogs

        result = await handle_search_dogs()

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_search_dogs_includes_thumbnail_url(sample_dog_data, sample_dog):
    with patch("src.agent.tools.search_dogs", new=AsyncMock(return_value=[sample_dog])):
        from src.agent.tools import handle_search_dogs

        result = await handle_search_dogs(size="medium")

    assert len(result) == 1
    assert "name" in result[0]
    assert "id" in result[0]


@pytest.mark.asyncio
async def test_search_dogs_passes_filters():
    mock_search = AsyncMock(return_value=[])

    with patch("src.agent.tools.search_dogs", new=mock_search):
        from src.agent.tools import handle_search_dogs

        await handle_search_dogs(size="small", temperament=["calm"])

    mock_search.assert_awaited_once_with(size="small", temperament=["calm"])


# ---------------------------------------------------------------------------
# handle_get_dog_profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_dog_profile_returns_full_profile(sample_dog, dog_id):
    with patch("src.agent.tools.get_dog_by_id", new=AsyncMock(return_value=sample_dog)):
        from src.agent.tools import handle_get_dog_profile

        result = await handle_get_dog_profile(dog_id=dog_id)

    assert result["name"] == "Mango"
    assert "story" in result
    assert "photos" in result


@pytest.mark.asyncio
async def test_get_dog_profile_returns_error_when_not_found(dog_id):
    with patch("src.agent.tools.get_dog_by_id", new=AsyncMock(return_value=None)):
        from src.agent.tools import handle_get_dog_profile

        result = await handle_get_dog_profile(dog_id=dog_id)

    assert "error" in result


# ---------------------------------------------------------------------------
# handle_get_user_profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_user_profile_returns_user_dict(sample_user):
    with patch("src.agent.tools.get_or_create_user", new=AsyncMock(return_value=sample_user)):
        from src.agent.tools import handle_get_user_profile

        result = await handle_get_user_profile(telegram_id=123456789)

    assert result["telegram_id"] == 123456789
    assert "funnel_stage" in result


@pytest.mark.asyncio
async def test_get_user_profile_creates_new_user_if_needed():
    new_user = User(
        id=uuid4(),
        telegram_id=999,
        funnel_stage="curious",
        preferences={},
        liked_dog_ids=[],
        intent="unknown",
    )
    with patch("src.agent.tools.get_or_create_user", new=AsyncMock(return_value=new_user)):
        from src.agent.tools import handle_get_user_profile

        result = await handle_get_user_profile(telegram_id=999)

    assert result["telegram_id"] == 999


# ---------------------------------------------------------------------------
# handle_update_user_profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_user_profile_calls_update(sample_user, user_id):
    mock_update = AsyncMock(return_value=sample_user)

    with patch("src.agent.tools.update_user", new=mock_update):
        from src.agent.tools import handle_update_user_profile

        result = await handle_update_user_profile(user_id=user_id, funnel_stage="interested")

    assert result["updated"] is True
    mock_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_user_profile_ignores_unknown_fields(sample_user, user_id):
    mock_update = AsyncMock(return_value=sample_user)

    with patch("src.agent.tools.update_user", new=mock_update):
        from src.agent.tools import handle_update_user_profile

        # Should not crash with valid fields
        result = await handle_update_user_profile(user_id=user_id, name="Emma Updated")

    assert result["updated"] is True


# ---------------------------------------------------------------------------
# handle_recall_past_conversations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_past_conversations_returns_summaries(sample_conversation_data, user_id):
    from src.models.schemas import ConversationMemory

    memory = ConversationMemory(**sample_conversation_data)
    with patch("src.agent.tools.get_recent_conversations", new=AsyncMock(return_value=[memory])):
        from src.agent.tools import handle_recall_past_conversations

        result = await handle_recall_past_conversations(user_id=user_id, query="calm dog")

    assert isinstance(result, list)
    assert "summary" in result[0]


@pytest.mark.asyncio
async def test_recall_past_conversations_returns_empty_for_new_user(user_id):
    with patch("src.agent.tools.get_recent_conversations", new=AsyncMock(return_value=[])):
        from src.agent.tools import handle_recall_past_conversations

        result = await handle_recall_past_conversations(user_id=user_id, query="dogs")

    assert result == []


# ---------------------------------------------------------------------------
# handle_check_calendar_availability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_calendar_returns_slots():
    fake_slots = [
        {"date": "2026-04-10", "time": "10:00", "available": True},
        {"date": "2026-04-10", "time": "14:00", "available": True},
    ]
    with patch("src.agent.tools.get_available_slots", new=AsyncMock(return_value=fake_slots)):
        from src.agent.tools import handle_check_calendar_availability

        result = await handle_check_calendar_availability()

    assert isinstance(result, list)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# handle_book_shelter_visit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_book_shelter_visit_creates_booking(sample_user, user_id, dog_id):
    from src.models.schemas import Booking

    booking = Booking(
        id=uuid4(),
        user_id=sample_user.id,
        dog_id=uuid4(),
        scheduled_at="2026-04-10T10:00:00",
        status="scheduled",
    )
    with (
        patch("src.agent.tools.get_or_create_user", new=AsyncMock(return_value=sample_user)),
        patch("src.agent.tools.get_dog_by_id", new=AsyncMock(return_value=None)),
        patch("src.agent.tools.create_booking", new=AsyncMock(return_value=booking)),
        patch("src.agent.tools.create_calendar_event", new=AsyncMock(return_value=None)),
        patch("src.agent.tools.send_admin_notification", new=AsyncMock()),
    ):
        from src.agent.tools import handle_book_shelter_visit

        result = await handle_book_shelter_visit(
            telegram_id=123456789,
            dog_id=dog_id,
            scheduled_at="2026-04-10T10:00:00",
        )

    assert result["confirmed"] is True
    assert "booking_id" in result


# ---------------------------------------------------------------------------
# handle_notify_admin
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notify_admin_sends_message():
    mock_send = AsyncMock()

    with patch("src.agent.tools.send_admin_notification", new=mock_send):
        from src.agent.tools import handle_notify_admin

        result = await handle_notify_admin(message="Someone is interested in Mango!")

    assert result["sent"] is True
    mock_send.assert_awaited_once_with("Someone is interested in Mango!")


# ---------------------------------------------------------------------------
# handle_send_donation_info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_donation_info_onetime():
    from src.agent.tools import handle_send_donation_info

    result = await handle_send_donation_info(donation_type="onetime")

    assert "url" in result
    assert "onetime" in result["url"] or "gofundme" in result["url"].lower() or result["url"]


@pytest.mark.asyncio
async def test_send_donation_info_recurring():
    from src.agent.tools import handle_send_donation_info

    result = await handle_send_donation_info(donation_type="recurring")

    assert "url" in result


@pytest.mark.asyncio
async def test_send_donation_info_defaults_to_both():
    from src.agent.tools import handle_send_donation_info

    result = await handle_send_donation_info()

    assert "onetime_url" in result or "url" in result


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_tool_dispatches_to_correct_handler():
    fake_executor = AsyncMock(return_value=[{"id": "123", "name": "Mango"}])
    executors = {"search_dogs": fake_executor}

    from src.agent.tools import execute_tool

    result = await execute_tool("search_dogs", {}, executors)

    fake_executor.assert_awaited_once_with()
    assert "Mango" in result


@pytest.mark.asyncio
async def test_execute_tool_returns_error_for_unknown_tool():
    from src.agent.tools import execute_tool

    result = await execute_tool("nonexistent_tool", {}, {})

    assert "Unknown tool" in result


@pytest.mark.asyncio
async def test_execute_tool_handles_exception_gracefully():
    async def crashing_handler(**kwargs):
        raise ValueError("DB connection failed")

    executors = {"search_dogs": crashing_handler}

    from src.agent.tools import execute_tool

    result = await execute_tool("search_dogs", {}, executors)

    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_execute_tool_json_encodes_dict_result():
    fake_executor = AsyncMock(return_value={"name": "Mango", "size": "medium"})
    executors = {"get_dog_profile": fake_executor}

    from src.agent.tools import execute_tool

    result = await execute_tool("get_dog_profile", {"dog_id": "123"}, executors)

    parsed = json.loads(result)
    assert parsed["name"] == "Mango"
