"""Google Calendar integration via the official Python SDK.

Authentication
--------------
Uses a **service account** JSON key file stored at the path set by
GOOGLE_SERVICE_ACCOUNT_FILE (default: credentials/google_calendar.json).

This file is project-local and gitignored, so each project can use a
different Google account simply by pointing to different credentials.

Setup (one time per project)
-----------------------------
1. Go to https://console.cloud.google.com
2. Create a project (or pick an existing one).
3. Enable the Google Calendar API.
4. IAM & Admin → Service Accounts → Create service account.
5. Keys → Add Key → JSON → download the file.
6. Save it as  credentials/google_calendar.json  in this project.
7. In Google Calendar, share your shelter calendar with the service
   account's email address (give it "Make changes to events" permission).
8. Copy the Calendar ID from Calendar Settings → Integrate calendar.
9. Set GOOGLE_CALENDAR_ID in .env.

Fallback
--------
If the credentials file is missing or any API call fails, the service
falls back to returning generated placeholder slots so the bot keeps
working during development without a real calendar.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
BANGKOK_TZ = ZoneInfo("Asia/Bangkok")

_MORNING_SLOT = "10:00"
_AFTERNOON_SLOT = "14:00"


def _get_service():
    """Return an authenticated Google Calendar API service, or None on failure.

    This is a synchronous function — always call it via asyncio.to_thread().
    """
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        creds_path = Path(settings.GOOGLE_SERVICE_ACCOUNT_FILE)
        if not creds_path.exists():
            logger.error(
                "Google Calendar: service account file not found at '%s'. "
                "Events will NOT be created in Google Calendar.",
                creds_path.resolve(),
            )
            return None

        creds = Credentials.from_service_account_file(str(creds_path), scopes=SCOPES)
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        logger.debug("Google Calendar service built successfully.")
        return service
    except Exception as exc:
        logger.error("Google Calendar: could not build service — %s", exc)
        return None


async def get_available_slots(days_ahead: int = 7) -> list[dict]:
    """Return available shelter visit slots for the next `days_ahead` days.

    Queries Google Calendar's freebusy endpoint to find gaps in the schedule,
    then proposes morning (10:00) and afternoon (14:00) slots for each day
    that isn't fully booked.

    Falls back to generated slots if the Calendar API is unavailable.
    """
    if not settings.GOOGLE_CALENDAR_ID:
        logger.warning(
            "Google Calendar: GOOGLE_CALENDAR_ID not set — returning generated slots."
        )
        return _generate_slots(days_ahead)

    service = await asyncio.to_thread(_get_service)
    if service is None:
        return _generate_slots(days_ahead)

    try:
        now = datetime.now(tz=timezone.utc)
        time_max = now + timedelta(days=days_ahead)

        body = {
            "timeMin": now.isoformat(),
            "timeMax": time_max.isoformat(),
            "items": [{"id": settings.GOOGLE_CALENDAR_ID}],
        }

        result = await asyncio.to_thread(
            lambda: service.freebusy().query(body=body).execute()
        )
        busy_periods = (
            result["calendars"]
            .get(settings.GOOGLE_CALENDAR_ID, {})
            .get("busy", [])
        )
        logger.info(
            "Google Calendar: freebusy returned %d busy period(s).", len(busy_periods)
        )
        return _slots_avoiding_busy(days_ahead, busy_periods)

    except Exception as exc:
        logger.error(
            "Google Calendar: freebusy query failed — %s. Returning generated slots.", exc
        )
        return _generate_slots(days_ahead)


async def create_calendar_event(
    summary: str,
    start_time: datetime,
    duration_minutes: int = 60,
    description: str = "",
) -> str | None:
    """Create a Google Calendar event and return its event ID, or None on failure."""
    if not settings.GOOGLE_CALENDAR_ID:
        logger.error(
            "Google Calendar: GOOGLE_CALENDAR_ID not set — event '%s' was NOT created.", summary
        )
        return None

    service = await asyncio.to_thread(_get_service)
    if service is None:
        logger.error(
            "Google Calendar: service unavailable — event '%s' was NOT created.", summary
        )
        return None

    # Ensure start_time is in Bangkok timezone for the calendar entry
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=BANGKOK_TZ)
    else:
        start_time = start_time.astimezone(BANGKOK_TZ)

    end_time = start_time + timedelta(minutes=duration_minutes)

    event_body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_time.isoformat(), "timeZone": "Asia/Bangkok"},
        "end": {"dateTime": end_time.isoformat(), "timeZone": "Asia/Bangkok"},
    }

    try:
        event = await asyncio.to_thread(
            lambda: service.events()
            .insert(calendarId=settings.GOOGLE_CALENDAR_ID, body=event_body)
            .execute()
        )
        event_id: str = event["id"]
        logger.info("Google Calendar: event created — id=%s title='%s'", event_id, summary)
        return event_id

    except Exception as exc:
        logger.error(
            "Google Calendar: failed to create event '%s' — %s", summary, exc
        )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slots_avoiding_busy(days_ahead: int, busy_periods: list[dict]) -> list[dict]:
    """Generate morning + afternoon slots (Bangkok time), skipping busy periods."""
    busy_ranges = [
        (
            datetime.fromisoformat(p["start"]).astimezone(timezone.utc),
            datetime.fromisoformat(p["end"]).astimezone(timezone.utc),
        )
        for p in busy_periods
    ]

    slots = []
    today = datetime.now(tz=BANGKOK_TZ).date()

    for offset in range(1, days_ahead + 1):
        date = today + timedelta(days=offset)
        if date.weekday() == 6:  # closed Sundays
            continue

        for time_str in (_MORNING_SLOT, _AFTERNOON_SLOT):
            hour, minute = map(int, time_str.split(":"))
            # Slot in Bangkok time
            slot_dt = datetime(date.year, date.month, date.day, hour, minute, tzinfo=BANGKOK_TZ)
            slot_end = slot_dt + timedelta(hours=1)
            slot_dt_utc = slot_dt.astimezone(timezone.utc)
            slot_end_utc = slot_end.astimezone(timezone.utc)

            overlaps = any(
                start < slot_end_utc and end > slot_dt_utc
                for start, end in busy_ranges
            )
            if not overlaps:
                slots.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "time": time_str,
                    "label": f"{date.strftime('%A, %B %d')} at {time_str} (Bangkok time)",
                    "iso": slot_dt.isoformat(),
                })

    return slots


def _generate_slots(days_ahead: int) -> list[dict]:
    """Fallback: return placeholder slots in Bangkok time."""
    slots = []
    today = datetime.now(tz=BANGKOK_TZ).date()
    for offset in range(1, days_ahead + 1):
        date = today + timedelta(days=offset)
        if date.weekday() == 6:
            continue
        for time_str in (_MORNING_SLOT, _AFTERNOON_SLOT):
            hour, minute = map(int, time_str.split(":"))
            slot_dt = datetime(date.year, date.month, date.day, hour, minute, tzinfo=BANGKOK_TZ)
            slots.append({
                "date": date.strftime("%Y-%m-%d"),
                "time": time_str,
                "label": f"{date.strftime('%A, %B %d')} at {time_str} (Bangkok time)",
                "iso": slot_dt.isoformat(),
            })
    return slots
