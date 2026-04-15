"""Admin notifications via Telegram direct message."""

import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


async def send_admin_notification(message: str) -> None:
    """Send a direct Telegram message to the shelter admin."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_ADMIN_CHAT_ID:
        logger.warning("Admin notification skipped — TELEGRAM_BOT_TOKEN or TELEGRAM_ADMIN_CHAT_ID not set")
        return

    url = f"{TELEGRAM_API}/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_ADMIN_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Admin notification sent (chat_id=%s)", settings.TELEGRAM_ADMIN_CHAT_ID)
    except Exception as exc:
        logger.error("Failed to send admin notification: %s", exc)
