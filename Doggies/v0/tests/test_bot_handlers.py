"""Tests for src/bot/telegram.py — written BEFORE implementation (TDD)."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.agent.core import AgentResponse


def make_update(text: str | None = None, telegram_id: int = 123456789) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = telegram_id
    update.effective_user.username = "test_user"
    update.message.text = text
    update.message.photo = None
    update.message.caption = None
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = telegram_id
    return update


def make_photo_update(
    caption: str = "Mango",
    telegram_id: int = 123456789,
    file_id: str = "file_abc123",
) -> MagicMock:
    photo_size = MagicMock()
    photo_size.file_id = file_id
    photo_size.width = 1600
    photo_size.height = 1200

    update = MagicMock()
    update.effective_user.id = telegram_id
    update.message.text = None
    update.message.photo = [photo_size]
    update.message.caption = caption
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = telegram_id
    return update


def make_context(image_bytes: bytes = b"fake_image") -> MagicMock:
    mock_file = AsyncMock()
    mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(image_bytes))

    context = MagicMock()
    context.bot.get_file = AsyncMock(return_value=mock_file)
    context.bot.send_media_group = AsyncMock()
    return context


# ---------------------------------------------------------------------------
# handle_message — text messages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_message_calls_run_agent():
    mock_run = AsyncMock(return_value=AgentResponse(text="Hello! Let me help you find a dog."))
    update = make_update(text="I want to adopt a dog")
    context = make_context()

    with patch("src.bot.bot.run_agent", new=mock_run):
        from src.bot.bot import handle_message

        await handle_message(update, context)

    mock_run.assert_awaited_once_with("I want to adopt a dog", 123456789)


@pytest.mark.asyncio
async def test_text_response_sent_to_user():
    mock_run = AsyncMock(return_value=AgentResponse(text="Here's Mango, a calm medium dog!"))
    update = make_update(text="Show me dogs")
    context = make_context()

    with patch("src.bot.bot.run_agent", new=mock_run):
        from src.bot.bot import handle_message

        await handle_message(update, context)

    update.message.reply_text.assert_awaited_once_with("Here's Mango, a calm medium dog!")


@pytest.mark.asyncio
async def test_photos_sent_as_media_group_when_present():
    photos = [
        "https://example.com/dogs/123/photo_0.jpg",
        "https://example.com/dogs/123/photo_1.jpg",
    ]
    mock_run = AsyncMock(return_value=AgentResponse(text="Here are photos of Mango!", photos=photos))
    update = make_update(text="Tell me about Mango")
    context = make_context()

    with patch("src.bot.bot.run_agent", new=mock_run):
        from src.bot.bot import handle_message

        await handle_message(update, context)

    context.bot.send_media_group.assert_awaited_once()
    call_kwargs = context.bot.send_media_group.call_args
    assert call_kwargs is not None


@pytest.mark.asyncio
async def test_no_media_group_when_no_photos():
    mock_run = AsyncMock(return_value=AgentResponse(text="Let me tell you about our dogs.", photos=[]))
    update = make_update(text="Hello")
    context = make_context()

    with patch("src.bot.bot.run_agent", new=mock_run):
        from src.bot.bot import handle_message

        await handle_message(update, context)

    context.bot.send_media_group.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_error_sends_friendly_message():
    mock_run = AsyncMock(side_effect=RuntimeError("DB connection failed"))
    update = make_update(text="Hello")
    context = make_context()

    with patch("src.bot.bot.run_agent", new=mock_run):
        from src.bot.bot import handle_message

        await handle_message(update, context)

    # Should not raise — user gets a friendly error message
    update.message.reply_text.assert_awaited_once()
    reply_text = update.message.reply_text.call_args[0][0]
    assert isinstance(reply_text, str)
    assert len(reply_text) > 0


@pytest.mark.asyncio
async def test_empty_text_message_ignored():
    update = make_update(text=None)
    context = make_context()

    with patch("src.bot.bot.run_agent", new=AsyncMock()) as mock_run:
        from src.bot.bot import handle_message

        await handle_message(update, context)

    mock_run.assert_not_awaited()


# ---------------------------------------------------------------------------
# handle_photo — admin photo upload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_admin_photo_rejected():
    update = make_photo_update(caption="Mango", telegram_id=999999)
    context = make_context()

    with patch("src.bot.bot.settings") as mock_settings:
        mock_settings.TELEGRAM_ADMIN_CHAT_ID = 123456789

        from src.bot.bot import handle_photo

        await handle_photo(update, context)

    update.message.reply_text.assert_awaited_once()
    reply = update.message.reply_text.call_args[0][0]
    assert "admin" in reply.lower() or "not" in reply.lower()


@pytest.mark.asyncio
async def test_admin_photo_without_caption_rejected():
    update = make_photo_update(caption="", telegram_id=123456789)
    context = make_context()

    with patch("src.bot.bot.settings") as mock_settings:
        mock_settings.TELEGRAM_ADMIN_CHAT_ID = 123456789

        from src.bot.bot import handle_photo

        await handle_photo(update, context)

    update.message.reply_text.assert_awaited_once()
    reply = update.message.reply_text.call_args[0][0]
    assert "caption" in reply.lower() or "name" in reply.lower()


@pytest.mark.asyncio
async def test_admin_photo_processes_and_replies(dog_id, sample_dog):
    update = make_photo_update(caption="Mango", telegram_id=123456789)
    context = make_context(image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # fake JPEG

    full_url = f"https://example.com/dogs/{dog_id}/photo_0.jpg"
    thumb_url = f"https://example.com/dogs/{dog_id}/thumb_0.jpg"

    with (
        patch("src.bot.bot.settings") as mock_settings,
        patch("src.bot.bot.get_dog_by_name", new=AsyncMock(return_value=sample_dog)),
        patch("src.bot.bot.process_and_upload_dog_photo", new=AsyncMock(return_value=(full_url, thumb_url))),
        patch("src.bot.bot.update_dog", new=AsyncMock(return_value=sample_dog)),
    ):
        mock_settings.TELEGRAM_ADMIN_CHAT_ID = 123456789

        from src.bot.bot import handle_photo

        await handle_photo(update, context)

    update.message.reply_text.assert_awaited_once()


# ---------------------------------------------------------------------------
# send_photo_album helper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_photo_album_batches_by_10():
    """11 photos should result in 2 send_media_group calls (10 + 1)."""
    photos = [f"https://example.com/photo_{i}.jpg" for i in range(11)]
    mock_bot = MagicMock()
    mock_bot.send_media_group = AsyncMock()

    from src.bot.bot import send_photo_album

    await send_photo_album(chat_id=123, photo_urls=photos, bot=mock_bot)

    assert mock_bot.send_media_group.call_count == 2


@pytest.mark.asyncio
async def test_send_photo_album_single_batch():
    photos = ["https://example.com/photo_0.jpg", "https://example.com/photo_1.jpg"]
    mock_bot = MagicMock()
    mock_bot.send_media_group = AsyncMock()

    from src.bot.bot import send_photo_album

    await send_photo_album(chat_id=123, photo_urls=photos, bot=mock_bot)

    mock_bot.send_media_group.assert_awaited_once()
