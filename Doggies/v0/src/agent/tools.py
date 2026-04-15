"""Tool definitions for Claude API and handler implementations."""

import json
import logging
from typing import Any, Callable, Coroutine

from src.config import settings
from src.db.bookings import create_booking
from src.db.dogs import get_dog_by_id, search_dogs
from src.db.memory import get_recent_conversations
from src.db.users import get_or_create_user, update_user
from src.models.schemas import BookingCreate, UserCreate, UserUpdate
from src.services.calendar import create_calendar_event, get_available_slots
from src.services.notifications import send_admin_notification

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (schema passed to Claude API)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "search_dogs",
        "description": (
            "Search for available dogs by traits. Returns a list of dogs with thumbnail photos. "
            "Use when the user describes what they're looking for in a dog."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "size": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "description": "Preferred dog size",
                },
                "temperament": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Desired temperament traits, e.g. ['calm', 'good_with_kids']",
                },
            },
        },
    },
    {
        "name": "get_dog_profile",
        "description": (
            "Get the full profile and full-size photos for a specific dog. "
            "Use when the user wants to know more about a particular dog."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dog_id": {"type": "string", "description": "UUID of the dog"},
            },
            "required": ["dog_id"],
        },
    },
    {
        "name": "get_user_profile",
        "description": (
            "Load the current user's profile — preferences, funnel stage, liked dogs, etc. "
            "ALWAYS call this at the start of every conversation."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "update_user_profile",
        "description": (
            "Save new information learned about the user during conversation. "
            "Call this whenever you learn something new (lifestyle, preferences, experience, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "language": {"type": "string", "description": "ISO language code, e.g. 'en', 'th', 'de'"},
                "living_situation": {"type": "string", "description": "e.g. 'villa with garden', 'apartment'"},
                "location": {"type": "string", "description": "Where they are and for how long"},
                "experience_with_dogs": {"type": "string", "description": "e.g. 'first-time', 'experienced'"},
                "lifestyle_notes": {"type": "string", "description": "Work situation, activity level, family"},
                "preferences": {
                    "type": "object",
                    "description": "Structured preferences: size, energy, gender, etc.",
                },
                "funnel_stage": {
                    "type": "string",
                    "enum": ["curious", "exploring", "interested", "ready", "adopted", "donor"],
                },
                "intent": {
                    "type": "string",
                    "enum": ["adopt", "donate", "both", "unknown"],
                },
            },
        },
    },
    {
        "name": "recall_past_conversations",
        "description": (
            "Retrieve summaries of past conversations with this user. "
            "Use for returning users to recall their history and preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for in past conversations (e.g. 'liked dogs', 'lifestyle')",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_calendar_availability",
        "description": "Get available shelter visit time slots for the coming week.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "How many days ahead to check (default: 7)",
                },
            },
        },
    },
    {
        "name": "book_shelter_visit",
        "description": (
            "Create a shelter visit booking for the user and a specific dog. "
            "Automatically notifies the admin. Use after the user confirms a date and time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dog_id": {"type": "string", "description": "UUID of the dog to visit"},
                "scheduled_at": {
                    "type": "string",
                    "description": "ISO 8601 datetime string, e.g. '2026-04-10T10:00:00'",
                },
                "notes": {"type": "string", "description": "Optional notes from the user"},
            },
            "required": ["dog_id", "scheduled_at"],
        },
    },
    {
        "name": "notify_admin",
        "description": (
            "Send a Telegram message to the shelter admin. "
            "Use when there is strong adoption interest, even without a confirmed visit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to send to the admin"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "send_donation_info",
        "description": "Return donation links for the shelter. Use when the user wants to donate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "donation_type": {
                    "type": "string",
                    "enum": ["onetime", "recurring", "both"],
                    "description": "Type of donation the user wants to make",
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------

async def handle_search_dogs(
    size: str | None = None,
    temperament: list[str] | None = None,
) -> list[dict]:
    dogs = await search_dogs(size=size, temperament=temperament)
    return [
        {
            "id": str(d.id),
            "name": d.name,
            "breed": d.breed,
            "age_estimate": d.age_estimate,
            "size": d.size,
            "gender": d.gender,
            "temperament": d.temperament,
            "story_preview": (d.story or "")[:150] + "..." if d.story and len(d.story) > 150 else d.story,
            "thumbnails": d.thumbnails,
            "status": d.status,
        }
        for d in dogs
    ]


async def handle_get_dog_profile(dog_id: str) -> dict:
    dog = await get_dog_by_id(dog_id)
    if not dog:
        return {"error": f"Dog with ID {dog_id} not found"}
    return {
        "id": str(dog.id),
        "name": dog.name,
        "breed": dog.breed,
        "age_estimate": dog.age_estimate,
        "size": dog.size,
        "gender": dog.gender,
        "temperament": dog.temperament,
        "medical_notes": dog.medical_notes,
        "story": dog.story,
        "photos": dog.photos,
        "thumbnails": dog.thumbnails,
        "status": dog.status,
    }


async def handle_get_user_profile(telegram_id: int) -> dict:
    user = await get_or_create_user(UserCreate(telegram_id=telegram_id))
    return {
        "id": str(user.id),
        "telegram_id": user.telegram_id,
        "name": user.name,
        "language": user.language,
        "living_situation": user.living_situation,
        "location": user.location,
        "experience_with_dogs": user.experience_with_dogs,
        "lifestyle_notes": user.lifestyle_notes,
        "preferences": user.preferences,
        "funnel_stage": user.funnel_stage,
        "liked_dog_ids": [str(d) for d in user.liked_dog_ids],
        "intent": user.intent,
    }


async def handle_update_user_profile(user_id: str, **update_fields: Any) -> dict:
    # Filter to only valid UserUpdate fields
    valid_fields = {
        k: v
        for k, v in update_fields.items()
        if k in UserUpdate.model_fields and v is not None
    }
    await update_user(user_id, UserUpdate(**valid_fields))
    return {"updated": True, "user_id": user_id, "fields_updated": list(valid_fields.keys())}


async def handle_recall_past_conversations(user_id: str, query: str) -> list[dict]:
    memories = await get_recent_conversations(user_id, limit=5)
    return [
        {
            "summary": m.summary,
            "facts": m.extracted_facts,
            "messages_count": m.messages_count,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in memories
    ]


async def handle_check_calendar_availability(days_ahead: int = 7) -> list[dict]:
    return await get_available_slots(days_ahead)


async def handle_book_shelter_visit(
    telegram_id: int,
    dog_id: str,
    scheduled_at: str,
    notes: str | None = None,
) -> dict:
    from datetime import datetime

    user = await get_or_create_user(UserCreate(telegram_id=telegram_id))
    dog = await get_dog_by_id(dog_id)
    dog_name = dog.name if dog else dog_id

    scheduled_dt = datetime.fromisoformat(scheduled_at)
    calendar_event_id = await create_calendar_event(
        summary=f"Shelter visit: {user.name or 'Adopter'} + {dog_name}",
        start_time=scheduled_dt,
        description=notes or "",
    )

    booking = await create_booking(
        BookingCreate(
            user_id=user.id,
            dog_id=dog_id,  # type: ignore[arg-type]
            scheduled_at=scheduled_dt,
            google_calendar_event_id=calendar_event_id,
        )
    )

    admin_msg = (
        f"🐕 <b>New shelter visit booked!</b>\n\n"
        f"<b>Visitor:</b> {user.name or 'Unknown'} (@{user.telegram_username or 'N/A'})\n"
        f"<b>Dog:</b> {dog_name}\n"
        f"<b>When:</b> {scheduled_at}\n"
        f"<b>Notes:</b> {notes or 'None'}"
    )
    await send_admin_notification(admin_msg)

    return {
        "booking_id": str(booking.id),
        "confirmed": True,
        "dog_name": dog_name,
        "scheduled_at": scheduled_at,
        "calendar_event_id": calendar_event_id,
    }


async def handle_notify_admin(message: str) -> dict:
    await send_admin_notification(message)
    return {"sent": True}


async def handle_send_donation_info(donation_type: str = "both") -> dict:
    onetime_url = settings.DONATION_ONETIME_URL
    recurring_url = settings.DONATION_RECURRING_URL

    if donation_type == "onetime":
        return {
            "type": "onetime",
            "url": onetime_url,
            "description": "One-time donation via GoFundMe — every amount helps with food, vet bills, and shelter.",
        }
    if donation_type == "recurring":
        return {
            "type": "recurring",
            "url": recurring_url,
            "description": "Monthly support — become a regular sponsor and help us plan ahead.",
        }
    # "both"
    return {
        "onetime_url": onetime_url,
        "recurring_url": recurring_url,
        "description": "You can make a one-time donation or set up monthly support.",
    }


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

async def execute_tool(
    tool_name: str,
    tool_input: dict,
    executors: dict[str, Callable[..., Coroutine]],
) -> str:
    """Dispatch a tool call to the correct executor and return a JSON string result."""
    executor = executors.get(tool_name)
    if not executor:
        return f"Unknown tool: {tool_name}"

    try:
        result = await executor(**tool_input)
        if isinstance(result, (dict, list)):
            return json.dumps(result, default=str)
        return str(result)
    except Exception as exc:
        logger.error("Tool error [%s]: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": f"Tool '{tool_name}' failed: {str(exc)}"})
