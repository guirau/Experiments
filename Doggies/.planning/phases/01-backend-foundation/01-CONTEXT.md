# Phase 1: Backend Foundation - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the Python FastAPI backend in `v1/backend/` with two working endpoints that
read live dog data from Supabase. No frontend work in this phase. Phase is complete
when `uvicorn main:app` starts cleanly and both endpoints return correct data.

</domain>

<decisions>
## Implementation Decisions

### Project Structure
- **D-01:** Use a modular layout — `main.py` (app entry), `config.py` (env vars),
  `database.py` (supabase-py client), `routers/dogs.py` (both dog routes).
- **D-02:** `routers/dogs.py` owns `GET /api/dogs` and `GET /api/dogs/{id}`.

### Python Tooling
- **D-03:** Manage dependencies with `requirements.txt`. Standard `pip install -r` workflow.
  No Poetry, no uv — keep setup commands obvious.

### Claude's Discretion
- CORS setup: Allow-all (`*`) for local dev is fine — Claude decides approach.
- Error handling for unknown dog IDs: Claude decides 404 response shape.
- Python version: Claude picks a current stable version (3.11+).
- Lifespan / startup pattern for the Supabase client: Claude decides.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — All v1 requirements with traceability; Phase 1
  covers INFRA-01, INFRA-03, INFRA-04, API-01, API-02, API-03.

### PRD
- `v1/docs/prd/prd.md` — Full product spec including dog schema field list
  (used to verify `GET /api/dogs/{id}` returns complete data).

### Project Context
- `.planning/PROJECT.md` — Constraints, key decisions, out-of-scope list.
- `CLAUDE.md` — Tech stack decisions and monorepo structure.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — `v1/backend/` does not exist yet; this phase creates it from scratch.

### Established Patterns
- Supabase env vars (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) already used in `v0/.env`.
  Phase 1 reuses the same var names in `v1/backend/.env` — no new Supabase project.
- No RLS on the `dogs` table — anon key reads all rows freely.

### Integration Points
- Frontend will call `http://localhost:8000/api/dogs` (Phase 2). Backend must have
  CORS enabled for `localhost:3000`.

</code_context>

<specifics>
## Specific Ideas

- Backend runs on port 8000 (specified in Phase 1 success criteria).
- `GET /api/dogs` response shape is locked: `{ dogs: [{ id: UUID, name: string }] }`.
- `GET /api/dogs/{id}` returns the full dog object (all fields from PRD §4).
- Start command is `uvicorn main:app` (or with `--reload` for dev) — must be documented.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-backend-foundation*
*Context gathered: 2026-04-21*
