"""Diagnose Google Calendar integration step by step.

Run from the project root:
    python scripts/check_calendar.py
"""

import asyncio
import sys
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings


def step(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print("─" * 60)


def ok(msg: str) -> None:
    print(f"  ✅  {msg}")


def fail(msg: str) -> None:
    print(f"  ❌  {msg}")


def info(msg: str) -> None:
    print(f"  ℹ️   {msg}")


# ---------------------------------------------------------------------------
# Step 1: Check env vars
# ---------------------------------------------------------------------------
step("Step 1 — Environment variables")

calendar_id = settings.GOOGLE_CALENDAR_ID
creds_file = settings.GOOGLE_SERVICE_ACCOUNT_FILE

if calendar_id:
    ok(f"GOOGLE_CALENDAR_ID = {calendar_id}")
else:
    fail("GOOGLE_CALENDAR_ID is empty — set it in .env")

if creds_file:
    ok(f"GOOGLE_SERVICE_ACCOUNT_FILE = {creds_file}")
else:
    fail("GOOGLE_SERVICE_ACCOUNT_FILE is empty")

# ---------------------------------------------------------------------------
# Step 2: Check credentials file
# ---------------------------------------------------------------------------
step("Step 2 — Service account file")

creds_path = Path(creds_file)
if creds_path.exists():
    ok(f"File found at: {creds_path.resolve()}")
    try:
        import json
        data = json.loads(creds_path.read_text())
        sa_email = data.get("client_email", "?")
        project = data.get("project_id", "?")
        ok(f"Service account email: {sa_email}")
        ok(f"GCP project: {project}")
        info(
            f"👉 Make sure this email is added as an editor to your Google Calendar:\n"
            f"     Calendar → Settings → Share with specific people → {sa_email}"
        )
    except Exception as e:
        fail(f"Could not parse credentials file: {e}")
else:
    fail(
        f"File NOT found at: {creds_path.resolve()}\n"
        "  Create it by following: CLAUDE.md → Setup → Google Calendar"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3: Build the service
# ---------------------------------------------------------------------------
step("Step 3 — Build Google Calendar service")

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    creds = Credentials.from_service_account_file(str(creds_path), scopes=SCOPES)
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    ok("Service built successfully")
except Exception as e:
    fail(f"Could not build service: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: List calendars the service account can see
# ---------------------------------------------------------------------------
step("Step 4 — Calendars accessible to this service account")

try:
    result = service.calendarList().list().execute()
    items = result.get("items", [])
    if items:
        ok(f"Found {len(items)} accessible calendar(s):")
        for cal in items:
            print(f"       • {cal['summary']}  →  {cal['id']}")
    else:
        fail(
            "No calendars found. The service account has no calendars shared with it.\n"
            "  → Share your calendar with the service account email above."
        )
except Exception as e:
    fail(f"calendarList failed: {e}")

# ---------------------------------------------------------------------------
# Step 5: Freebusy query
# ---------------------------------------------------------------------------
step("Step 5 — Freebusy query for configured GOOGLE_CALENDAR_ID")

if not calendar_id:
    fail("Skipped — GOOGLE_CALENDAR_ID is not set")
else:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(tz=timezone.utc)
    body = {
        "timeMin": now.isoformat(),
        "timeMax": (now + timedelta(days=7)).isoformat(),
        "items": [{"id": calendar_id}],
    }
    try:
        result = service.freebusy().query(body=body).execute()
        cal_data = result["calendars"].get(calendar_id)
        if cal_data is None:
            fail(
                f"Calendar ID '{calendar_id}' not found in freebusy response.\n"
                "  → Check that GOOGLE_CALENDAR_ID in .env matches exactly."
            )
        elif "errors" in cal_data:
            fail(f"Freebusy returned errors: {cal_data['errors']}")
        else:
            busy = cal_data.get("busy", [])
            ok(f"Freebusy succeeded — {len(busy)} busy period(s) in next 7 days")
    except Exception as e:
        fail(f"Freebusy query failed: {e}")

# ---------------------------------------------------------------------------
# Step 6: Create a test event
# ---------------------------------------------------------------------------
step("Step 6 — Create a test event (will be deleted immediately)")

if not calendar_id:
    fail("Skipped — GOOGLE_CALENDAR_ID is not set")
else:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    BANGKOK = ZoneInfo("Asia/Bangkok")
    start = datetime.now(tz=BANGKOK) + timedelta(hours=1)
    end = start + timedelta(minutes=30)
    event_body = {
        "summary": "🐕 Doggies bot — calendar test (safe to delete)",
        "start": {"dateTime": start.isoformat(), "timeZone": "Asia/Bangkok"},
        "end": {"dateTime": end.isoformat(), "timeZone": "Asia/Bangkok"},
    }
    try:
        event = service.events().insert(calendarId=calendar_id, body=event_body).execute()
        event_id = event["id"]
        ok(f"Test event created! id={event_id}")
        info("Check your Google Calendar — you should see '🐕 Doggies bot — calendar test'")

        # Clean up
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        ok("Test event deleted (cleanup done)")
    except Exception as e:
        fail(f"Event creation failed: {e}")

print(f"\n{'═' * 60}")
print("  Diagnosis complete")
print("═" * 60)
