# Doggies v1

## What This Is

A password-gated, read-only internal web tool for a dog shelter admin to browse
dogs stored in Supabase and view individual dog profiles. The frontend is
Next.js; the backend is Python FastAPI. It is not a public-facing app.

## Core Value

The admin can log in and immediately see which dogs are in the system, with
one click to view full profile details for any dog.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Admin can log in with username `admin` / password `1234`
- [ ] Login failure shows an inline error message
- [ ] Admin sees a list of all dogs (names only) after logging in
- [ ] Each dog in the list has a "Dog Details" button
- [ ] Clicking "Dog Details" shows the full dog profile (all fields)
- [ ] Dog detail view shows "No photos" when the photos array is empty
- [ ] A back button on the detail view returns to the dog list
- [ ] FastAPI backend serves `GET /api/dogs` returning `{ dogs: [{id, name}] }`
- [ ] FastAPI backend serves `GET /api/dogs/{id}` returning the full dog object

### Out of Scope

- Create / update / delete operations — read-only v1
- Supabase Auth or real session management — in-memory React state only
- Pagination, search, filtering — too early
- Photo upload or image processing — not needed
- Telegram integration — v0 handles that
- AI agent or matching logic — future milestone
- Role-based access control — single admin only
- Production hardening of auth — internal tooling

## Context

- **v0:** Telegram bot with Claude agent, pgvector memory, Google Calendar — this is unrelated to v1's web UI.
- **Supabase:** Shared project from v0. `dogs` table has no RLS. Env vars (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) reused from `v0/.env`.
- **Monorepo layout:** `v1/frontend/` (Next.js App Router, TypeScript) + `v1/backend/` (Python FastAPI).
- **Future:** Vercel deployment for frontend, so Next.js App Router structure is intentional.
- **Dog schema:** name, breed, age_estimate, size, gender, temperament, medical_notes, story, photos, status, intake_date (from `v0/scripts/setup_db.sql`).

## Constraints

- **Tech stack:** Next.js App Router + TypeScript (frontend); Python + FastAPI + supabase-py (backend) — decided, no deviations
- **Auth:** Hardcoded `admin` / `1234` check in React state — intentional mock-grade behavior
- **Data:** Read-only against existing `dogs` table — no schema changes
- **Deployment target:** Local dev now, Vercel-compatible structure for future

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FastAPI over Next.js API routes | User wants `frontend/` + `backend/` monorepo shape for future expansion | — Pending |
| Shared Supabase project with v0 | No new project needed; same dog data | — Pending |
| In-memory auth state only | Mock-grade v1; no session complexity | — Pending |
| Monorepo in `v1/` | Clean separation from v0, Vercel-compatible | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-21 after initialization*
