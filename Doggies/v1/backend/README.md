# Doggies v1 — Backend

Python FastAPI backend serving dog data from Supabase.

## Requirements

- Python 3.11+

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your SUPABASE_URL and SUPABASE_ANON_KEY
```

## Start

```bash
# Development (auto-reload on file changes)
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --port 8000
```

API will be available at http://localhost:8000

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/dogs | Returns all dogs `{ dogs: [{ id, name }] }` |
| GET | /api/dogs/{id} | Returns full dog object by UUID |

## API Docs

Interactive docs available at http://localhost:8000/docs (Swagger UI)
