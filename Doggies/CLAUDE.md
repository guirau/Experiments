# CLAUDE.md — Instructions for Claude Code

## Project Overview

Doggies is a Telegram bot powered by an AI agent that helps stray dogs get adopted in Koh Phangan, Thailand. The agent converses with potential adopters, matches them with dogs, books shelter visits, and handles donations.

Read `docs/PRD.md` for full product requirements, architecture, schema, and agent design.

---

## Tech Stack

- **Language:** Python 3.11+
- **Backend:** FastAPI + Uvicorn
- **AI Agent:** Anthropic SDK (raw tool use loop — no LangChain, no frameworks)
- **Database:** Supabase (Postgres + pgvector)
- **File Storage:** Supabase Storage (dog photos)
- **Bot Interface:** python-telegram-bot (v20+, async)
- **Calendar:** Google Calendar via `gws` CLI
- **Models/Validation:** Pydantic v2

---

## Project Structure

```
doggies/
├── CLAUDE.md                  # This file — instructions for you
├── README.md                  # Human-readable setup & run instructions
├── pyproject.toml             # Dependencies (use poetry or pip)
├── .env.example               # Environment variables template
│
├── docs/
│   └── PRD.md                 # Product Requirements Document
│
├── src/
│   ├── __init__.py
│   ├── main.py                # FastAPI app entry point
│   │
│   ├── bot/
│   │   ├── __init__.py
│   │   └── telegram.py        # Telegram bot setup + message handlers
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py            # Agent loop: message → Claude → tool use → response
│   │   ├── system_prompt.py   # System prompt string
│   │   └── tools.py           # Tool definitions + execution functions
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── client.py          # Supabase client setup
│   │   ├── dogs.py            # Dog CRUD operations
│   │   ├── users.py           # User profile CRUD
│   │   └── memory.py          # Conversation memory (pgvector embeddings)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── calendar.py        # Google Calendar integration via gws
│   │   ├── photos.py          # Photo upload, optimization, and retrieval via Supabase Storage
│   │   └── notifications.py   # Admin notification via Telegram
│   │
│   └── models/
│       ├── __init__.py
│       └── schemas.py         # Pydantic models for all entities
│
├── scripts/
│   ├── seed_dogs.py           # Populate DB with sample dogs
│   └── setup_db.sql           # Database schema creation
│
└── tests/
    ├── __init__.py
    ├── conftest.py            # Shared fixtures (mock Supabase, mock Claude, etc.)
    ├── test_agent_loop.py     # Agent loop: tool use cycle, response handling
    ├── test_tools.py          # Each tool function in isolation
    ├── test_db_dogs.py        # Dog CRUD
    ├── test_db_users.py       # User profile CRUD
    ├── test_db_memory.py      # Conversation memory + vector search
    ├── test_calendar.py       # Calendar availability + booking
    ├── test_photos.py         # Photo upload, optimization, retrieval
    └── test_bot_handlers.py   # Telegram message handling (text + photos)
```

---

## Development Workflow

### CRITICAL: Test-Driven Development

**Always write tests BEFORE writing implementation code.**

For every module or feature:
1. **Write the test file first** — define what the function/class should do via test cases
2. **Run the tests** — confirm they fail (red)
3. **Write the implementation** — make the tests pass (green)
4. **Refactor** — clean up while tests stay green

This is non-negotiable. Do not skip to implementation.

### Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_agent_loop.py -v

# Run with coverage
pytest tests/ -v --cov=src

# Run tests matching a pattern
pytest tests/ -v -k "test_search_dogs"
```

### Running the App

```bash
# Start FastAPI server (development)
uvicorn src.main:app --reload --port 8000

# Start Telegram bot (runs alongside FastAPI)
python -m src.bot.telegram
```

---

## Coding Conventions

### General
- Use type hints everywhere
- Use `async/await` throughout — FastAPI and python-telegram-bot are both async
- Use Pydantic models for all data flowing in/out of functions
- Keep functions small and focused — one function, one job
- Use descriptive variable names, no abbreviations

### Agent Loop (`src/agent/core.py`)
The agent loop is the core of the application. It follows this exact cycle:

```python
async def run_agent(user_message: str, telegram_id: int) -> str:
    # 1. Load user profile + conversation history
    # 2. Build messages array: system prompt + context + user message
    # 3. Call Claude API with tools
    # 4. LOOP: if response has tool_use blocks:
    #    a. Execute each tool
    #    b. Append tool results to messages
    #    c. Call Claude API again
    #    d. Repeat until response is text only
    # 5. After final response:
    #    a. Update user profile with extracted info
    #    b. Save conversation summary + embedding to memory
    # 6. Return text response to user
```

**Do not use any agent framework.** Build this loop manually with the Anthropic SDK. This is intentional for learning.

### Tool Implementation (`src/agent/tools.py`)
- Each tool is a standalone async function
- Tool functions receive parsed input, return a string result
- Use a dispatcher dict to map tool names to functions
- Always handle errors gracefully — return error messages, don't crash the loop

```python
# Pattern for tool dispatcher
TOOL_HANDLERS = {
    "search_dogs": handle_search_dogs,
    "get_dog_profile": handle_get_dog_profile,
    "get_user_profile": handle_get_user_profile,
    # ...
}

async def execute_tool(tool_name: str, tool_input: dict) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}"
    try:
        result = await handler(**tool_input)
        return json.dumps(result) if isinstance(result, dict) else str(result)
    except Exception as e:
        return f"Tool error: {str(e)}"
```

### Database (`src/db/`)
- Use Supabase Python client (`supabase-py`)
- Each entity (dogs, users, memory) gets its own module
- All DB functions are async
- Return Pydantic models, not raw dicts

### Error Handling
- Never let exceptions crash the Telegram bot or agent loop
- Log errors with context (user_id, tool_name, etc.)
- Return user-friendly messages when things go wrong
- Use Python's `logging` module, not print statements

### Environment Variables
- All secrets and config in `.env`
- Access via `pydantic-settings` (BaseSettings class)
- Never hardcode URLs, keys, or IDs

### Photo Handling (`src/services/photos.py`)

Dog photos are stored in Supabase Storage and optimized for Telegram delivery.

**Upload flow (admin adds a dog):**
1. Receive original image (from Telegram photo message or seed script)
2. Optimize with Pillow:
   - Resize to max 1280px on longest side (Telegram's optimal photo size)
   - Convert to JPEG if not already
   - Compress to quality=85 (good balance of size vs quality)
   - Strip EXIF metadata
   - Target: each photo under 300KB
3. Generate a thumbnail (320px wide, quality=70) for quick previews
4. Upload both to Supabase Storage: `dogs/{dog_id}/photo_{n}.jpg` and `dogs/{dog_id}/thumb_{n}.jpg`
5. Store public URLs in the dog's `photos` array (full) and `thumbnails` array (thumbs)

**Retrieval flow (sending to adopter in chat):**
1. Agent decides to show a dog → tool returns photo URLs
2. Bot handler sends photos via Telegram's `send_photo` using the public Supabase URL
3. For multiple photos, use Telegram's `send_media_group` (album) — max 10 photos per group
4. Always send thumbnails in search results, full photos when showing a specific dog profile

**Key rules:**
- Use `Pillow` (PIL) for all image processing — no other image libraries
- Always process images before uploading — never store raw uploads
- Use Supabase Storage public URLs (enable public access on the bucket)
- Handle missing/broken photos gracefully — dog profile should still work without photos

---

## Key Design Decisions

1. **Raw Anthropic SDK, no frameworks** — We want to understand the agent loop, not hide it behind abstractions
2. **Single agent with tools, not multi-agent** — One agent with clear tool boundaries is simpler and sufficient
3. **Structured memory + vector memory** — User profiles store hard facts; pgvector stores conversational context
4. **Telegram first** — Minimum viable interface, maximum reach in Thailand
5. **Async everything** — FastAPI, python-telegram-bot, and Supabase client are all async-native
6. **Optimized photos via Supabase Storage** — Full-size + thumbnail variants, processed with Pillow, served as public URLs

---

## Testing Strategy

### What to Test

**Agent loop (`test_agent_loop.py`):**
- Simple message → text response (no tool use)
- Message that triggers one tool → correct tool called → response uses result
- Message that triggers multiple tools in sequence
- Tool returns error → agent handles gracefully
- Returning user → agent loads profile and memory before responding

**Tools (`test_tools.py`):**
- `search_dogs`: filters by size, temperament, etc.
- `get_dog_profile`: returns full profile, handles missing dog
- `get_user_profile`: returns profile, handles new user
- `update_user_profile`: updates fields correctly
- `recall_past_conversations`: returns relevant memories
- `check_calendar_availability`: returns available slots
- `book_shelter_visit`: creates booking + calendar event
- `notify_admin`: sends Telegram message
- `send_donation_info`: returns correct links by type

**Database (`test_db_*.py`):**
- CRUD operations for dogs and users
- Vector similarity search returns relevant conversations
- Edge cases: duplicate users, missing records

**Bot handlers (`test_bot_handlers.py`):**
- Text message → passed to agent → response sent back
- Photo message from admin → downloaded, optimized, uploaded to Supabase
- Agent response with dog photos → sent as Telegram photo/album
- Error in agent → user gets friendly error message

**Photos (`test_photos.py`):**
- Image optimization: output is JPEG, max 1280px, under 300KB
- Thumbnail generation: 320px wide, lower quality
- EXIF metadata is stripped
- Non-JPEG input (PNG, WEBP) is converted to JPEG
- Upload to Supabase Storage returns correct public URL
- Corrupt/invalid image input handled gracefully

### How to Mock

- **Mock Claude API** with fixed responses (text-only and tool-use)
- **Mock Supabase** with in-memory dicts or SQLite
- **Mock Telegram** with `python-telegram-bot`'s test utilities
- **Mock gws** with fixture responses for calendar operations

### Fixtures (`conftest.py`)

Provide reusable fixtures:
- `sample_dogs` — list of dog dicts for seeding
- `sample_user` — a returning user with history
- `mock_claude_client` — patched Anthropic client
- `mock_supabase` — patched Supabase client
- `sample_conversation_history` — past messages for memory tests
- `sample_image` — a small in-memory image (Pillow Image) for photo tests
- `mock_supabase_storage` — patched storage client for upload/download

---

## Build Order (Hackathon)

Follow this order. Each phase builds on the previous one.

### Phase 1: Foundation (Hours 1-2)
1. Set up project structure, `pyproject.toml`, `.env.example`
2. Write `setup_db.sql` and create Supabase tables
3. Write + test `src/db/client.py` (Supabase connection)
4. Write + test `src/db/dogs.py` (CRUD)
5. Write + test `src/db/users.py` (CRUD)
6. Write + run `scripts/seed_dogs.py`

### Phase 2: Agent Core (Hours 3-5)
1. Write tests for agent loop (`test_agent_loop.py`)
2. Implement `src/agent/system_prompt.py`
3. Implement `src/agent/tools.py` — tool definitions + handlers
4. Write tests for each tool (`test_tools.py`)
5. Implement `src/agent/core.py` — the agent loop
6. Test agent via CLI script before connecting to Telegram

### Phase 3: Telegram + Calendar (Hours 5-7)
1. Write tests for bot handlers (`test_bot_handlers.py`)
2. Implement `src/bot/telegram.py`
3. Implement `src/services/calendar.py` (gws integration)
4. Implement `src/services/notifications.py`
5. Write + test `src/db/memory.py` (pgvector)
6. Full integration test: message → agent → tools → response

### Phase 4: Polish + Demo (Hours 7-8)
1. Test scenarios: new user, returning user, donor, tourist
2. Test multilingual responses
3. Fix edge cases
4. Prepare demo

---

## Reminders

- **Tests first, always.** Don't write implementation without a failing test.
- **Keep the agent loop simple.** It's a while loop, not rocket science.
- **Don't over-engineer.** This is a hackathon. Working > perfect.
- **Commit often.** Small, working increments.
- **If stuck on a tool integration, mock it and move on.** Calendar not working? Return fake slots. Supabase down? Use SQLite. Ship the agent logic, fix integrations later.
