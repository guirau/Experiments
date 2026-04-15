"""Targeted calendar access check — bypasses calendarList (unreliable for service accounts)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]
creds_path = Path(settings.GOOGLE_SERVICE_ACCOUNT_FILE)
creds = Credentials.from_service_account_file(str(creds_path), scopes=SCOPES)
service = build("calendar", "v3", credentials=creds, cache_discovery=False)

cal_id = settings.GOOGLE_CALENDAR_ID
print(f"Testing direct access to calendar: {cal_id}\n")

# 1. Get calendar metadata directly
try:
    cal = service.calendars().get(calendarId=cal_id).execute()
    print(f"✅ Calendar metadata accessible: '{cal.get('summary')}'")
except Exception as e:
    print(f"❌ Cannot read calendar metadata: {e}")
    print("\nPossible causes:")
    print("  • Calendar not shared with the service account — or shared as 'See only free/busy' instead of 'Make changes to events'")
    print("  • Google Workspace org policy blocking external service account access")
    sys.exit(1)

# 2. Insert test event
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
BANGKOK = ZoneInfo("Asia/Bangkok")
today = datetime.now(tz=BANGKOK)
start = today.replace(hour=10, minute=0, second=0, microsecond=0)
end = start + timedelta(minutes=30)
event_body = {
    "summary": "🐕 Doggies bot — test event (safe to delete)",
    "start": {"dateTime": start.isoformat(), "timeZone": "Asia/Bangkok"},
    "end": {"dateTime": end.isoformat(), "timeZone": "Asia/Bangkok"},
}
try:
    event = service.events().insert(calendarId=cal_id, body=event_body).execute()
    print(f"✅ Test event created: {event['id']}")
    print("   → Check your Google Calendar now. You should see the test event.")
except Exception as e:
    print(f"❌ Could not create event: {e}")
    print("\nThe calendar is readable but not writable.")
    print("  → In Google Calendar, re-share with 'Make changes to events' (not just 'See all event details')")
