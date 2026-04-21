import logging
import threading
from supabase import Client, create_client
from config import settings

logger = logging.getLogger(__name__)

_client: Client | None = None
_lock = threading.Lock()


def get_supabase_client() -> Client:
    global _client
    with _lock:
        if _client is None:
            _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
            logger.info("Supabase client initialized")
    return _client
