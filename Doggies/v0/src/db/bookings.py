import logging

from src.db.client import get_supabase_client
from src.models.schemas import Booking, BookingCreate

logger = logging.getLogger(__name__)


async def create_booking(booking_data: BookingCreate) -> Booking:
    client = get_supabase_client()
    payload = {
        "user_id": str(booking_data.user_id),
        "dog_id": str(booking_data.dog_id),
        "scheduled_at": booking_data.scheduled_at.isoformat()
        if hasattr(booking_data.scheduled_at, "isoformat")
        else booking_data.scheduled_at,
        "google_calendar_event_id": booking_data.google_calendar_event_id,
        "status": "scheduled",
        "admin_notified": False,
    }
    response = client.table("bookings").insert(payload).execute()
    return Booking(**response.data[0])


async def get_booking_by_id(booking_id: str) -> Booking | None:
    client = get_supabase_client()
    response = client.table("bookings").select("*").eq("id", booking_id).execute()
    if not response.data:
        return None
    return Booking(**response.data[0])


async def update_booking_status(booking_id: str, status: str) -> Booking:
    client = get_supabase_client()
    response = (
        client.table("bookings")
        .update({"status": status})
        .eq("id", booking_id)
        .execute()
    )
    return Booking(**response.data[0])
