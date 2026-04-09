"""The agent loop: user message → Claude → tool use → response.

This is a raw Anthropic SDK loop — no frameworks, no magic.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from src.agent.system_prompt import build_system_prompt
from src.agent.tools import (
    TOOL_DEFINITIONS,
    execute_tool,
    handle_book_shelter_visit,
    handle_check_calendar_availability,
    handle_get_dog_profile,
    handle_get_user_profile,
    handle_notify_admin,
    handle_recall_past_conversations,
    handle_search_dogs,
    handle_send_donation_info,
    handle_update_user_profile,
)
from src.config import settings
from src.db.memory import save_conversation
from src.db.users import get_or_create_user
from src.models.schemas import ConversationMemoryCreate, UserCreate

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 8192
MAX_TOOL_LOOPS = 10  # safety cap
MAX_SESSION_TURNS = 15  # keep last 15 user/assistant pairs per session

# In-memory conversation history keyed by telegram_id.
# Each entry is a list of {"role": ..., "content": ...} dicts (text only, no tool blocks).
_sessions: dict[int, list[dict]] = {}


@dataclass
class AgentResponse:
    """Result of a single agent run — text reply plus optional photos to send."""
    text: str
    photos: list[str] = field(default_factory=list)


def build_tool_executors(telegram_id: int, user_id: str) -> dict:
    """Create tool executors with user context bound via closures."""
    return {
        "search_dogs": handle_search_dogs,
        "get_dog_profile": handle_get_dog_profile,
        "get_user_profile": lambda **_: handle_get_user_profile(telegram_id=telegram_id),
        "update_user_profile": lambda **kwargs: handle_update_user_profile(user_id=user_id, **kwargs),
        "recall_past_conversations": lambda **kwargs: handle_recall_past_conversations(user_id=user_id, **kwargs),
        "check_calendar_availability": handle_check_calendar_availability,
        "book_shelter_visit": lambda **kwargs: handle_book_shelter_visit(telegram_id=telegram_id, **kwargs),
        "notify_admin": handle_notify_admin,
        "send_donation_info": handle_send_donation_info,
    }


async def run_agent(user_message: str, telegram_id: int) -> AgentResponse:
    """Run the agent loop for a single user message.

    1. Load / create the user profile.
    2. Build the system prompt with user context.
    3. Call Claude with tools.
    4. Loop: execute tools → feed results back → repeat until text response.
    5. Save conversation summary.
    6. Return AgentResponse(text, photos).
    """
    # --- 1. Load user ---
    user = await get_or_create_user(UserCreate(telegram_id=telegram_id))
    user_id = str(user.id)

    # --- 2. Build context ---
    system_prompt = build_system_prompt(user)
    executors = build_tool_executors(telegram_id=telegram_id, user_id=user_id)

    # Prepend session history so Claude remembers the ongoing conversation
    history = _sessions.get(telegram_id, [])
    messages: list[dict[str, Any]] = history + [{"role": "user", "content": user_message}]

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    final_text = ""
    photos_to_send: list[str] = []

    # --- 3-4. Agent loop ---
    for _ in range(MAX_TOOL_LOOPS):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
            tools=TOOL_DEFINITIONS,  # type: ignore[arg-type]
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    final_text = block.text
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.info("Executing tool: %s (id=%s)", block.name, block.id)
                result_str = await execute_tool(block.name, block.input, executors)

                # Collect full-size photos from dog profile lookups
                if block.name == "get_dog_profile":
                    photos_to_send = _extract_photos(result_str)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        else:
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text
            break

    if not final_text:
        final_text = "I'm sorry, I couldn't generate a response. Please try again."

    # --- 5. Persist session history (clean text turns only, no tool blocks) ---
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_text},
    ]
    _sessions[telegram_id] = updated_history[-(MAX_SESSION_TURNS * 2):]

    # --- 6. Save to DB (extracted facts via Haiku) ---
    try:
        await _save_conversation_memory(
            user_id=user_id,
            user_message=user_message,
            response=final_text,
            client=client,
        )
    except Exception as exc:
        logger.warning("Failed to save conversation memory: %s", exc)

    return AgentResponse(text=final_text, photos=photos_to_send)


def _extract_photos(tool_result_str: str) -> list[str]:
    """Pull photo URLs out of a get_dog_profile tool result string."""
    try:
        data = json.loads(tool_result_str)
        if isinstance(data, dict):
            return data.get("photos") or []
    except (json.JSONDecodeError, TypeError):
        pass
    return []


async def _save_conversation_memory(
    user_id: str,
    user_message: str,
    response: str,
    client: anthropic.AsyncAnthropic,
) -> None:
    summary = f"User: {user_message[:300]}\nAgent: {response[:300]}"
    extracted_facts = await _extract_facts(user_message, response, client)
    await save_conversation(
        ConversationMemoryCreate(
            user_id=user_id,  # type: ignore[arg-type]
            summary=summary,
            extracted_facts=extracted_facts,
            messages_count=2,
        )
    )


async def _extract_facts(
    user_message: str,
    response: str,
    client: anthropic.AsyncAnthropic,
) -> dict:
    """Use Claude Haiku to extract key facts from a conversation turn."""
    try:
        result = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract key facts from this conversation as a JSON object. "
                        "Include: dogs_discussed (names), user_preferences, decisions_made, intent_signals. "
                        "Return ONLY a valid JSON object, no explanation.\n\n"
                        f"User: {user_message[:400]}\n"
                        f"Agent: {response[:400]}"
                    ),
                }
            ],
        )
        text = result.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as exc:
        logger.warning("Fact extraction failed: %s", exc)
        return {}
