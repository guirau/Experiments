# Doggies v1 — Product Requirements Document

## 1. Problem & Users

**Problem:** The shelter admin needs a simple internal tool to browse dogs in
the Supabase database and view individual dog profiles. The Telegram bot (v0)
handles public-facing adoption conversations. This web interface covers the
admin's separate need to quickly check who is in the system.

**Primary user:** Shelter admin (single person). This is not a public-facing app.

---

## 2. Core Value

A password-gated, read-only web view of the dogs currently in the Supabase
database.

---

## 3. Core Flows

Three screens, one meaningful operation.

### Flow 1 — Login

1. Admin opens the app.
2. A login form renders (username + password fields).
3. Admin enters `admin` / `1234`.
4. On match: app transitions to the dog list. On mismatch: form shows an inline error.
5. Auth state lives in React state only. Page reload re-prompts login. This is
   intentional mock-grade behavior for v1 — no session persistence needed.

### Flow 2 — Dog List

1. After login, the app fetches all dogs from `GET /api/dogs`.
2. Each row in the list shows the dog's **name only** plus a "Dog Details" button.
3. No pagination, search, filtering, or sorting in v1.

### Flow 3 — Dog Detail

1. Admin clicks "Dog Details" on a list row.
2. App navigates to `/dogs/[id]` and fetches from `GET /api/dogs/:id`.
3. Detail view renders all profile fields: name, breed, age_estimate, size,
   gender, temperament, medical_notes, story, status, photos (rendered as images),
   and intake_date.
4. A back navigation control returns to the dog list.

---

## 4. Data Model

v1 reads from the existing `dogs` table in the shared Supabase project (same
instance as v0). v1 is read-only — no writes, no new tables.

**Fields consumed:**

| Field | Type | Used in |
|---|---|---|
| id | UUID | URL param, list row key |
| name | text | List + detail |
| breed | text | Detail |
| age_estimate | text | Detail |
| size | text | Detail |
| gender | text | Detail |
| temperament | text[] | Detail |
| medical_notes | text | Detail |
| story | text | Detail |
| photos | text[] | Detail (rendered as `<img>` tags) |
| status | text | Detail |
| intake_date | timestamptz | Detail |

Fields not used in v1: thumbnails, instagram_post_text, created_at, updated_at.

For the full schema definition see `v0/scripts/setup_db.sql`.

---

## 5. Architecture

**Monorepo layout:**

```
v1/
├── frontend/                     # Next.js App Router, TypeScript
│   ├── app/
│   │   ├── page.tsx              # Login screen (Flow 1)
│   │   ├── dogs/
│   │   │   ├── page.tsx          # Dog list (Flow 2)
│   │   │   └── [id]/
│   │   │       └── page.tsx      # Dog detail (Flow 3)
│   │   └── layout.tsx
│   ├── lib/
│   │   └── api.ts                # Fetch helpers pointing at NEXT_PUBLIC_API_URL
│   └── ... (config files)
└── backend/                      # Python + FastAPI
    ├── main.py                   # App entrypoint, CORS config
    ├── routers/
    │   └── dogs.py               # GET /api/dogs, GET /api/dogs/{id}
    ├── db/
    │   └── supabase.py           # Supabase client (supabase-py, anon key)
    └── requirements.txt
```

**Request flow:**

```
Browser → Next.js frontend (port 3000)
              ↓ fetch(NEXT_PUBLIC_API_URL + "/api/dogs")
         FastAPI backend (port 8000)
              ↓ supabase-py anon key
         Supabase Postgres (shared with v0)
```

**Stack:** Next.js App Router (TypeScript) for the UI; Python FastAPI for the API layer. FastAPI connects to Supabase via `supabase-py` using the anon key. The frontend calls the backend over HTTP — no Next.js API routes used for data fetching.

**Database:** Shared Supabase project from v0. RLS is not enabled on the `dogs` table — no policy changes needed. Env vars (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) are shared from `v0/.env`.

**Authentication:** Client-side React state check — `username === "admin" && password === "1234"`. No Supabase Auth, no JWTs, no cookies. Login is forgotten on page reload — intentional for v1.

---

## 6. API Surface

| Method | Path | Response shape | Purpose |
|---|---|---|---|
| GET | /api/dogs | `{ dogs: { id: string, name: string }[] }` | Populate dog list |
| GET | /api/dogs/:id | Full dog object (fields listed in §4) | Populate detail view |

Both handlers are implemented in FastAPI (`backend/routers/dogs.py`) and query
Supabase via `supabase-py`. No auth check on the API routes in v1 — the
credential gate is frontend-only.

---

## 7. Non-Goals (v1)

- No create, update, or delete operations
- No Supabase Auth or real session management
- No photo upload or image processing
- No Telegram integration
- No AI agent, matching logic, or recommendations
- No pagination, search, or filtering on the dog list
- No role-based access control
- No production hardening of the auth check (internal tooling only)

---

## 8. Resolved Decisions

1. **Supabase RLS:** Not enabled on `dogs`. No policy changes needed — anon key
   reads freely.

2. **Empty photos array:** Render a "No photos" text label in the detail view
   when `photos` is empty or null.

3. **Env vars:** Reuse `SUPABASE_URL` and `SUPABASE_ANON_KEY` from `v0/.env`
   directly in `v1/backend/.env` (FastAPI reads them) and expose
   `NEXT_PUBLIC_API_URL=http://localhost:8000` in `v1/frontend/.env.local`.
