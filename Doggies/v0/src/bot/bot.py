"""Telegram bot setup and message handlers."""

import logging

from telegram import InputMediaPhoto, Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from src.agent.core import run_agent
from src.config import settings
from src.db.dogs import get_dog_by_name, update_dog
from src.models.schemas import DogUpdate
from src.services.photos import process_and_upload_dog_photo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text message handler
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route text messages from any user through the agent."""
    if not update.message or not update.message.text:
        return

    telegram_id = update.effective_user.id

    try:
        response = await run_agent(update.message.text, telegram_id)
        await update.message.reply_text(response.text)

        if response.photos:
            await send_photo_album(
                chat_id=update.effective_chat.id,
                photo_urls=response.photos,
                bot=context.bot,
            )
    except Exception as exc:
        logger.error("Agent error for user %s: %s", telegram_id, exc, exc_info=True)
        await update.message.reply_text(
            "Sorry, something went wrong on my end. Please try again in a moment! 🙏"
        )


# ---------------------------------------------------------------------------
# Photo handler (admin dog intake)
# ---------------------------------------------------------------------------

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — admin uploads dog photos for intake."""
    if not update.message or not update.message.photo:
        return

    telegram_id = update.effective_user.id

    if telegram_id != settings.TELEGRAM_ADMIN_CHAT_ID:
        await update.message.reply_text(
            "Photo upload is for the shelter admin only."
        )
        return

    caption = (update.message.caption or "").strip()
    if not caption:
        await update.message.reply_text(
            "Please add the dog's name as a caption so I know which dog this photo is for."
        )
        return

    try:
        # Download highest-resolution photo from Telegram
        photo = update.message.photo[-1]
        tg_file = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(await tg_file.download_as_bytearray())

        # Find the dog by name
        dog = await get_dog_by_name(caption)
        if not dog:
            await update.message.reply_text(
                f"I couldn't find a dog named '{caption}'. "
                "Check the name and try again, or add the dog first."
            )
            return

        # How many photos already exist?
        photo_index = len(dog.photos)

        # Process + upload
        full_url, thumb_url = await process_and_upload_dog_photo(
            image_bytes, dog_id=str(dog.id), photo_index=photo_index
        )

        # Append to dog record (immutable update — new lists)
        updated_photos = list(dog.photos) + [full_url]
        updated_thumbnails = list(dog.thumbnails) + [thumb_url]
        await update_dog(str(dog.id), DogUpdate(photos=updated_photos, thumbnails=updated_thumbnails))

        await update.message.reply_text(
            f"Photo #{photo_index + 1} for {dog.name} uploaded successfully! ✅"
        )

    except Exception as exc:
        logger.error("Photo processing failed for user %s: %s", telegram_id, exc, exc_info=True)
        await update.message.reply_text(
            "Failed to process the photo. Please try again."
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def send_photo_album(chat_id: int, photo_urls: list[str], bot) -> None:
    """Send photo URLs as Telegram media groups (max 10 per group)."""
    for batch_start in range(0, len(photo_urls), 10):
        batch = photo_urls[batch_start : batch_start + 10]
        media = [InputMediaPhoto(media=url) for url in batch]
        try:
            await bot.send_media_group(chat_id=chat_id, media=media)
        except Exception as exc:
            logger.warning("Failed to send photo album batch: %s", exc)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the bot with long-polling (development mode)."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting Doggies bot (polling)...")
    app = create_app()
    app.run_polling(drop_pending_updates=True)


def create_app() -> Application:
    """Build and return the configured Telegram Application."""
    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    return app


if __name__ == "__main__":
    main()
