# Design — Prospecting trigger service (run discover/enrich from the UI)

> Superpowers `brainstorming` output (approved 2026-07-03). Implementation plan:
> `docs/superpowers/plans/2026-07-03-prospecting-trigger-service.md`.

## Context

The Prospecting feature's data comes from two Python CLIs (`src/discover.py`, `src/enrich.py`)
run by hand from a terminal. The user wants to trigger them from the web UI with buttons,
**without rewriting the scripts** — via a new service that calls them. The web app was
client-only (no server code); the scripts are Python that assume CWD = repo root and carry a
hard cost guard (enrich only touches `enrich_status='queued'`).

## Confirmed decisions (this session)

1. **Service = a small FastAPI app** (`src/api.py`), not a Next.js route and not a subprocess.
   It *is* Python, so it runs from repo root and **imports and calls the same functions the CLIs
   use** — sharing the venv, `.env`, `db.py`, and (for free) the enrich cost guard. A Next route
   would be node-spawning-python with cwd juggling + stdout parsing.
2. **Import & call**, with one **non-behavioral** edit: extract `enrich.py`'s loop into
   `run_enrichment(client, api_key, queued)` that `main()` calls (CLI unchanged). `discover.py`
   already exposes `run_discovery`/`run_scoring`, imported untouched.
3. **Background jobs + polling**: buttons start a job and return immediately; the UI polls
   `GET /status` and re-fetches prospects when the job settles.
4. The enrich **"N queued / ~$cost" confirm is computed client-side** from the already-loaded
   `enrich_status` — no preview endpoint (avoid building the same knowledge twice).

## Architecture

**`src/api.py` (FastAPI), run `uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000`:**
- `load_dotenv()` at import (db.py reads env at call time, never loads `.env` itself).
- **Bind 127.0.0.1 only** — endpoints spend money and are unauthenticated; localhost is the
  mitigation, plus the client-side cost confirm.
- Endpoints: `GET /health`, `GET /status` (both jobs' state), `POST /discover` (`{limit?}`),
  `POST /enrich`. In-memory `JOBS` keyed by name with a **running-guard → 409**.
- Each job runner builds its **own** `db.get_client()` (fresh per run, env loaded first),
  replicates the script's env-var checks, and wraps everything so `SystemExit`/exceptions become
  the job's `error` state — never a 500 that breaks polling. `enrich` runs
  `enrich.queued_only(db.fetch_queued_prospects(client))` → `enrich.run_enrichment(...)`, so the
  guard is preserved exactly.
- Deps: `fastapi` + `uvicorn[standard]`.

**Frontend:**
- `web/next.config.ts` — rewrite `/api/py/:path*` → `PY_API_URL` (default `http://127.0.0.1:8000`)
  so the browser calls **same-origin** (no CORS).
- `web/lib/prospectingApi.ts` — `triggerDiscover`, `triggerEnrich`, `fetchJobStatus`.
- `web/hooks/useProspects.ts` — `refresh()` (re-fetch after a run) + derived `queuedCount`.
- `web/components/prospecting/ProspectingControls.tsx` — control bar in `ProspectingPanel`:
  Discover (+ limit input), Enrich queued (N) with cost confirm, a `setInterval` poll of
  `/status` that stops + refreshes on settle, and a status line. The map's existing client-side
  "Queue" write is unchanged; the new Enrich button is what spends money, explicitly.

## Honest scope note

The scripts still have **not run against live Google/SerpApi/Claude** (blocked on the user's DB
migration + paid keys). This service only *triggers* them, so the two layers stay independently
runnable — a data bug is isolable via the unchanged CLI + `--inspect`.

## Verification (done)

- `poetry run pytest` → 30 passed (incl. new `run_enrichment` + FastAPI `TestClient` tests).
- Empirically confirmed `uvicorn --app-dir src` from repo root resolves the repo-root `.env`
  (SUPABASE + ANTHROPIC load); `POST /discover` + `/enrich` run as background jobs and capture
  missing-key errors into job state (not a 500).
- Web `tsc` + `eslint` + `next build` clean.
- Full integration smoke (Playwright): `/api/py/health` proxied through Next returns ok; clicking
  **Discover** drove proxy → job → poll → status line
  ("Discover: error — GOOGLE_MAPS_API_KEY is not set in .env"). No React/Leaflet crashes.
- Deferred (needs keys/DB): the live discover→map-select→enrich→prices→tracker happy path.

## Out of scope

Auth · persisted job queue / websockets · production deploy + process orchestration · any change
to script *behavior* · changing the map's client-side "Queue" write.
