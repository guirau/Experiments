# Doggies 🐾

AI-Powered Dog Adoption Assistant for Koh Phangan, Thailand.

A Telegram bot that helps stray dogs find loving homes through natural conversation, intelligent matching, and frictionless shelter visit booking.

## Quick Start

### Prerequisites

- Python 3.11+
- [Supabase](https://supabase.com) account (free tier works)
- [Telegram Bot Token](https://t.me/BotFather)
- [Anthropic API Key](https://console.anthropic.com)
- [Google Workspace CLI (gws)](https://github.com/nicholasgasior/gws) configured with calendar access

### Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd doggies

# Install dependencies
pip install -e ".[dev]"

# Copy env template and fill in your keys
cp .env.example .env

# Set up database tables in Supabase
# Run the SQL from scripts/setup_db.sql in Supabase SQL editor

# Seed sample dogs
python scripts/seed_dogs.py

# Run tests
pytest tests/ -v

# Start the bot
python -m src.main
```

### Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `TELEGRAM_BOT_TOKEN` | From @BotFather |
| `TELEGRAM_ADMIN_CHAT_ID` | Shelter admin's Telegram chat ID |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Supabase service role key |
| `SUPABASE_STORAGE_BUCKET` | Storage bucket name (default: `dog-photos`) |
| `GOOGLE_CALENDAR_ID` | Shelter's Google Calendar ID |
| `GOFUNDME_LINK` | One-time donation link |
| `RECURRING_DONATION_LINK` | Monthly donation link |

## How It Works

1. Someone messages the Telegram bot
2. The AI agent loads their profile (or creates one)
3. Natural conversation about their lifestyle, preferences, experience
4. Agent searches the dog database and recommends matches with personal pitches
5. Dog photos are sent directly in chat — thumbnails for browsing, full albums for specific dogs
6. When ready: book a shelter visit → calendar event → admin notified
7. Or: share donation links for one-time or recurring support
8. Everything is remembered for next time

## Documentation

- **[PRD](docs/PRD.md)** — Full product requirements, architecture, schema, agent design
- **[CLAUDE.md](CLAUDE.md)** — Development instructions for Claude Code

## Tech Stack

- **Bot:** python-telegram-bot
- **Backend:** FastAPI
- **AI:** Anthropic SDK (Claude, raw tool use)
- **Database:** Supabase (Postgres + pgvector)
- **Images:** Pillow (optimization) + Supabase Storage
- **Calendar:** Google Calendar via gws
- **Language:** Python 3.11+

## License

MIT
