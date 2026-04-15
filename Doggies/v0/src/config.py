import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic
    ANTHROPIC_API_KEY: str = ""

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # Telegram
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_ADMIN_CHAT_ID: int = 0

    # Google Calendar
    GOOGLE_CALENDAR_ID: str = ""
    # Path to a service account JSON file, relative to the project root.
    # e.g. credentials/google_calendar.json
    GOOGLE_SERVICE_ACCOUNT_FILE: str = "credentials/google_calendar.json"

    # Donation links
    DONATION_ONETIME_URL: str = "https://gofundme.com"
    DONATION_RECURRING_URL: str = "https://patreon.com"

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
