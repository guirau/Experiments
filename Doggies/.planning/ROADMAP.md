# Roadmap: Doggies v1

## Overview

Three phases deliver a working internal dog-browsing tool. Phase 1 builds and
validates the FastAPI backend with live Supabase data. Phase 2 adds the Next.js
frontend with login and dog list. Phase 3 wires the detail screen and completes
the full end-to-end flow.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Backend Foundation** - FastAPI scaffold + Supabase connection + both API endpoints working (completed 2026-04-21)
- [ ] **Phase 2: Frontend Auth + Dog List** - Next.js scaffold + login screen + dog list screen
- [ ] **Phase 3: Dog Detail + Integration** - Detail screen + photos handling + back navigation + end-to-end flow

## Phase Details

### Phase 1: Backend Foundation
**Goal**: The FastAPI backend is running locally and serving live dog data from Supabase
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-03, INFRA-04, API-01, API-02, API-03
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Backend scaffold: dirs, requirements.txt, config.py, database.py, .env, README
- [x] 01-02-PLAN.md — API endpoints: routers/dogs.py + main.py with CORS

**Success Criteria** (what must be TRUE):
  1. Running `uvicorn main:app` starts the FastAPI server on port 8000 with no errors
  2. `GET /api/dogs` returns a JSON object with a `dogs` array containing `{id, name}` entries read from the Supabase `dogs` table
  3. `GET /api/dogs/{id}` returns the full dog object for a valid UUID
  4. The backend reads `SUPABASE_URL` and `SUPABASE_ANON_KEY` from `v1/backend/.env` and connects successfully
  5. The `v1/` monorepo structure exists with `frontend/` and `backend/` directories and documented start commands

### Phase 2: Frontend Auth + Dog List
**Goal**: The admin can open the app in a browser, log in, and see the list of all dogs
**Depends on**: Phase 1
**Requirements**: INFRA-02, AUTH-01, AUTH-02, AUTH-03, LIST-01, LIST-02, LIST-03
**Success Criteria** (what must be TRUE):
  1. Opening `http://localhost:3000` shows a login form with username and password fields
  2. Entering `admin` / `1234` transitions the app to the dog list screen
  3. Entering wrong credentials shows an inline error message on the login form
  4. Reloading the page after login returns to the login screen (no persisted session)
  5. The dog list shows all dog names fetched from `GET /api/dogs`, each with a "Dog Details" button
**Plans**: TBD
**UI hint**: yes

### Phase 3: Dog Detail + Integration
**Goal**: Clicking any dog in the list shows its full profile, and the admin can navigate back
**Depends on**: Phase 2
**Requirements**: DETAIL-01, DETAIL-02, DETAIL-03, DETAIL-04, DETAIL-05
**Success Criteria** (what must be TRUE):
  1. Clicking "Dog Details" on any list row navigates to the dog's detail screen
  2. The detail screen displays all profile fields: name, breed, age_estimate, size, gender, temperament, medical_notes, story, status, and intake_date
  3. When a dog has photos, they render as `<img>` elements; when photos is empty or null, "No photos" is shown
  4. A back button on the detail screen returns the admin to the dog list
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Backend Foundation | 2/2 | Complete   | 2026-04-21 |
| 2. Frontend Auth + Dog List | 0/TBD | Not started | - |
| 3. Dog Detail + Integration | 0/TBD | Not started | - |
