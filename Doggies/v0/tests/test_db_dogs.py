"""Tests for src/db/dogs.py — written BEFORE implementation (TDD)."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.models.schemas import Dog, DogCreate, DogUpdate
from tests.conftest import make_supabase_response


# ---------------------------------------------------------------------------
# create_dog
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_dog_returns_dog_model(mock_supabase, sample_dog_data, sample_dog_create):
    # Arrange
    mock_supabase.table.return_value.insert.return_value.execute.return_value = (
        make_supabase_response([sample_dog_data])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import create_dog

        # Act
        dog = await create_dog(sample_dog_create)

    # Assert
    assert isinstance(dog, Dog)
    assert dog.name == "Mango"
    assert dog.size == "medium"
    assert dog.status == "available"


@pytest.mark.asyncio
async def test_create_dog_calls_correct_table(mock_supabase, sample_dog_data, sample_dog_create):
    mock_supabase.table.return_value.insert.return_value.execute.return_value = (
        make_supabase_response([sample_dog_data])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import create_dog

        await create_dog(sample_dog_create)

    mock_supabase.table.assert_called_with("dogs")


# ---------------------------------------------------------------------------
# get_dog_by_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_dog_by_id_returns_dog(mock_supabase, sample_dog_data, dog_id):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([sample_dog_data])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import get_dog_by_id

        dog = await get_dog_by_id(dog_id)

    assert isinstance(dog, Dog)
    assert str(dog.id) == dog_id


@pytest.mark.asyncio
async def test_get_dog_by_id_returns_none_when_not_found(mock_supabase, dog_id):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import get_dog_by_id

        dog = await get_dog_by_id(dog_id)

    assert dog is None


# ---------------------------------------------------------------------------
# search_dogs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_dogs_returns_list(mock_supabase, sample_dog_data):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([sample_dog_data])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import search_dogs

        dogs = await search_dogs()

    assert isinstance(dogs, list)
    assert len(dogs) == 1
    assert isinstance(dogs[0], Dog)


@pytest.mark.asyncio
async def test_search_dogs_filters_by_size(mock_supabase, sample_dog_data):
    query_mock = MagicMock()
    query_mock.execute.return_value = make_supabase_response([sample_dog_data])
    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value = query_mock

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import search_dogs

        dogs = await search_dogs(size="medium")

    assert all(d.size == "medium" for d in dogs)


@pytest.mark.asyncio
async def test_search_dogs_returns_empty_list_when_none_found(mock_supabase):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import search_dogs

        dogs = await search_dogs()

    assert dogs == []


# ---------------------------------------------------------------------------
# update_dog
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_dog_status(mock_supabase, sample_dog_data, dog_id):
    updated = {**sample_dog_data, "status": "reserved"}
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([updated])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import update_dog

        dog = await update_dog(dog_id, DogUpdate(status="reserved"))

    assert dog.status == "reserved"


@pytest.mark.asyncio
async def test_update_dog_photos(mock_supabase, sample_dog_data, dog_id):
    photos = ["https://example.com/photo1.jpg"]
    thumbnails = ["https://example.com/thumb1.jpg"]
    updated = {**sample_dog_data, "photos": photos, "thumbnails": thumbnails}
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([updated])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import update_dog

        dog = await update_dog(dog_id, DogUpdate(photos=photos, thumbnails=thumbnails))

    assert dog.photos == photos
    assert dog.thumbnails == thumbnails


# ---------------------------------------------------------------------------
# list_available_dogs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_available_dogs_only_returns_available(mock_supabase, sample_dog_data):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        make_supabase_response([sample_dog_data])
    )

    with patch("src.db.dogs.get_supabase_client", return_value=mock_supabase):
        from src.db.dogs import list_available_dogs

        dogs = await list_available_dogs()

    assert all(d.status == "available" for d in dogs)
