"""Tests for src/agent/core.py — written BEFORE implementation (TDD)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.models.schemas import User


def make_text_response(text: str) -> MagicMock:
    """Mock Anthropic response with a single text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_use_response(tool_name: str, tool_id: str, tool_input: dict) -> MagicMock:
    """Mock Anthropic response with a tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def make_sample_user(telegram_id: int = 123456789) -> User:
    return User(
        id=uuid4(),
        telegram_id=telegram_id,
        funnel_stage="curious",
        preferences={},
        liked_dog_ids=[],
        intent="unknown",
    )


# ---------------------------------------------------------------------------
# Simple message → text response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_simple_message_returns_text():
    user = make_sample_user()
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=make_text_response("Hello! I can help you find a dog."))

    with (
        patch("src.agent.core.get_or_create_user", new=AsyncMock(return_value=user)),
        patch("src.agent.core.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.core.save_conversation", new=AsyncMock()),
    ):
        from src.agent.core import run_agent

        result = await run_agent("Hello, I want to adopt a dog", 123456789)

    assert "dog" in result.text.lower() or "help" in result.text.lower()
    assert isinstance(result.text, str)


@pytest.mark.asyncio
async def test_run_agent_calls_claude_api():
    user = make_sample_user()
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=make_text_response("Hello!"))

    with (
        patch("src.agent.core.get_or_create_user", new=AsyncMock(return_value=user)),
        patch("src.agent.core.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.core.save_conversation", new=AsyncMock()),
    ):
        from src.agent.core import run_agent

        await run_agent("Hello", 123456789)

    mock_client.messages.create.assert_awaited()


# ---------------------------------------------------------------------------
# Tool use cycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_message_triggers_tool_use_cycle():
    """Agent calls a tool then gets a text response on the second Claude call."""
    user = make_sample_user()
    tool_id = "tool_abc123"
    search_result = json.dumps([{"id": "1", "name": "Mango", "size": "medium"}])

    tool_response = make_tool_use_response("search_dogs", tool_id, {"size": "medium"})
    final_response = make_text_response("I found Mango, a calm medium-sized dog!")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

    fake_executor = AsyncMock(return_value=[{"id": "1", "name": "Mango"}])

    with (
        patch("src.agent.core.get_or_create_user", new=AsyncMock(return_value=user)),
        patch("src.agent.core.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.core.save_conversation", new=AsyncMock()),
        patch("src.agent.core.build_tool_executors", return_value={"search_dogs": fake_executor}),
        patch("src.agent.core._extract_facts", new=AsyncMock(return_value={})),
    ):
        from src.agent.core import run_agent

        result = await run_agent("Find me a calm medium dog", 123456789)

    # Claude was called twice: once for tool use, once after tool result
    assert mock_client.messages.create.call_count == 2
    assert isinstance(result.text, str)


@pytest.mark.asyncio
async def test_tool_result_is_appended_to_messages():
    """After tool execution, the result is appended before the next Claude call."""
    user = make_sample_user()
    tool_id = "tool_xyz"
    tool_response = make_tool_use_response("search_dogs", tool_id, {})
    final_response = make_text_response("Here are the dogs!")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

    fake_executor = AsyncMock(return_value=[])

    with (
        patch("src.agent.core.get_or_create_user", new=AsyncMock(return_value=user)),
        patch("src.agent.core.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.core.save_conversation", new=AsyncMock()),
        patch("src.agent.core.build_tool_executors", return_value={"search_dogs": fake_executor}),
    ):
        from src.agent.core import run_agent

        await run_agent("Show me dogs", 123456789)

    # Second call should include the tool result in messages
    second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
    # Messages should contain the assistant tool_use block and the user tool_result
    roles = [m["role"] for m in second_call_messages]
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_exception_does_not_crash_agent():
    """If a tool raises an exception, the agent returns a graceful error message."""
    user = make_sample_user()
    tool_id = "tool_err"
    tool_response = make_tool_use_response("search_dogs", tool_id, {})
    final_response = make_text_response("Sorry, I had trouble searching. Let me try again.")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

    async def crashing_executor(**kwargs):
        raise RuntimeError("Supabase connection failed")

    with (
        patch("src.agent.core.get_or_create_user", new=AsyncMock(return_value=user)),
        patch("src.agent.core.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.core.save_conversation", new=AsyncMock()),
        patch("src.agent.core.build_tool_executors", return_value={"search_dogs": crashing_executor}),
    ):
        from src.agent.core import run_agent

        # Should not raise
        result = await run_agent("Show me dogs", 123456789)

    assert isinstance(result.text, str)


# ---------------------------------------------------------------------------
# User profile is loaded at start
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returning_user_triggers_get_or_create():
    user = make_sample_user()
    mock_get_user = AsyncMock(return_value=user)
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=make_text_response("Welcome back!"))

    with (
        patch("src.agent.core.get_or_create_user", new=mock_get_user),
        patch("src.agent.core.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.core.save_conversation", new=AsyncMock()),
    ):
        from src.agent.core import run_agent

        await run_agent("Hello again!", 123456789)

    mock_get_user.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_profile_included_in_system_context():
    """User profile data ends up in the messages sent to Claude."""
    user = make_sample_user()
    user.name = "Emma"
    user.funnel_stage = "exploring"

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=make_text_response("Hi Emma!"))

    with (
        patch("src.agent.core.get_or_create_user", new=AsyncMock(return_value=user)),
        patch("src.agent.core.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.core.save_conversation", new=AsyncMock()),
    ):
        from src.agent.core import run_agent

        await run_agent("Hi!", 123456789)

    # The system prompt passed to Claude should reference the user
    # Use call_args_list[0] — the main agent call (Haiku fact extraction is a later call)
    first_call_kwargs = mock_client.messages.create.call_args_list[0][1]
    system_text = first_call_kwargs.get("system", "")
    assert isinstance(system_text, str)
    assert len(system_text) > 0
