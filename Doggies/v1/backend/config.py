import os
from dotenv import load_dotenv

# Load .env from the backend directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


class Settings:
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""

    def __init__(self) -> None:
        self.SUPABASE_URL = _require("SUPABASE_URL")
        self.SUPABASE_ANON_KEY = _require("SUPABASE_ANON_KEY")


settings = Settings()
