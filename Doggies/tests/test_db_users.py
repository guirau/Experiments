"""Tests for src/db/users.py — written BEFORE implementation (TDD)."""

from unittest.mock import patch
from uuid import uuid4

import pytest

from src.models.schemas import User, UserCreate, UserUpdate
from tests.conftest import make_supabase_response


# ---------------------------------------------------------------------------
# get_or_create_user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_or_create_user_creates_new_user(mock_supabase, sample_user_data, sample_user_create):
    # First call (get) returns empty, second call (insert) returns new user
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([])
    )
    mock_supabase.table.return_value.insert.return_value.execute.return_value = (
        make_supabase_response([sample_user_data])
    )

    with patch("src.db.users.get_supabase_client", return_value=mock_supabase):
        from src.db.users import get_or_create_user

        user = await get_or_create_user(sample_user_create)

    assert isinstance(user, User)
    assert user.telegram_id == 123456789


@pytest.mark.asyncio
async def test_get_or_create_user_returns_existing_user(mock_supabase, sample_user_data, sample_user_create):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([sample_user_data])
    )

    with patch("src.db.users.get_supabase_client", return_value=mock_supabase):
        from src.db.users import get_or_create_user

        user = await get_or_create_user(sample_user_create)

    assert isinstance(user, User)
    assert user.name == "Emma"
    # Should NOT have called insert
    mock_supabase.table.return_value.insert.assert_not_called()


# ---------------------------------------------------------------------------
# get_user_by_telegram_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_user_by_telegram_id_returns_user(mock_supabase, sample_user_data):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([sample_user_data])
    )

    with patch("src.db.users.get_supabase_client", return_value=mock_supabase):
        from src.db.users import get_user_by_telegram_id

        user = await get_user_by_telegram_id(123456789)

    assert isinstance(user, User)
    assert user.telegram_id == 123456789


@pytest.mark.asyncio
async def test_get_user_by_telegram_id_returns_none_when_not_found(mock_supabase):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([])
    )

    with patch("src.db.users.get_supabase_client", return_value=mock_supabase):
        from src.db.users import get_user_by_telegram_id

        user = await get_user_by_telegram_id(999999999)

    assert user is None


# ---------------------------------------------------------------------------
# update_user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_user_returns_updated_model(mock_supabase, sample_user_data, user_id):
    updated = {**sample_user_data, "funnel_stage": "interested", "intent": "adopt"}
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([updated])
    )

    with patch("src.db.users.get_supabase_client", return_value=mock_supabase):
        from src.db.users import update_user

        user = await update_user(user_id, UserUpdate(funnel_stage="interested", intent="adopt"))

    assert user.funnel_stage == "interested"
    assert user.intent == "adopt"


@pytest.mark.asyncio
async def test_update_user_only_sends_non_none_fields(mock_supabase, sample_user_data, user_id):
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([sample_user_data])
    )

    with patch("src.db.users.get_supabase_client", return_value=mock_supabase):
        from src.db.users import update_user

        await update_user(user_id, UserUpdate(name="Emma Updated"))

    # Only 'name' should be in the update payload
    call_args = mock_supabase.table.return_value.update.call_args
    payload = call_args[0][0]
    assert "name" in payload
    assert "funnel_stage" not in payload


# ---------------------------------------------------------------------------
# add_liked_dog
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_liked_dog_appends_to_liked_list(mock_supabase, sample_user_data, user_id):
    new_dog_id = str(uuid4())
    updated = {**sample_user_data, "liked_dog_ids": [new_dog_id]}
    # First call: get current user
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([sample_user_data])
    )
    # Second call: update
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([updated])
    )

    with patch("src.db.users.get_supabase_client", return_value=mock_supabase):
        from src.db.users import add_liked_dog

        user = await add_liked_dog(user_id, new_dog_id)

    assert new_dog_id in [str(d) for d in user.liked_dog_ids]


@pytest.mark.asyncio
async def test_add_liked_dog_does_not_duplicate(mock_supabase, sample_user_data, user_id):
    existing_dog_id = str(uuid4())
    user_with_like = {**sample_user_data, "liked_dog_ids": [existing_dog_id]}
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([user_with_like])
    )
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([user_with_like])
    )

    with patch("src.db.users.get_supabase_client", return_value=mock_supabase):
        from src.db.users import add_liked_dog

        user = await add_liked_dog(user_id, existing_dog_id)

    # liked_dog_ids should still have exactly one entry
    assert len(user.liked_dog_ids) == 1
