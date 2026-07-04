# Prospecting Trigger Service Implementation Plan

> Superpowers plan (implemented 2026-07-03). Spec:
> `docs/superpowers/specs/2026-07-03-prospecting-trigger-service-design.md`.

**Goal:** trigger `src/discover.py` / `src/enrich.py` from the Prospecting tab's buttons via a
small FastAPI service that imports and calls the same functions the CLIs use (background jobs +
status polling), keeping the scripts' behavior + cost guard intact.

**Status (2026-07-03): implemented & statically verified.** `poetry run pytest` → 30 passed;
web `tsc`/`eslint`/`next build` clean; full proxy→job→poll→status chain exercised via Playwright.
Live Google/SerpApi/Claude E2E deferred (needs the user's SQL migration + paid keys).

---

## Task 1 — Deps
- [x] `pyproject.toml` — add `fastapi` + `uvicorn[standard]`; `poetry install`.

## Task 2 — Backend (non-behavioral refactor + service)
- [x] `src/enrich.py` — extract `run_enrichment(client, api_key, queued) -> (enriched, errored)`
      from `main()`; `main()` calls it (CLI output identical).
- [x] `src/api.py` — FastAPI app: `load_dotenv()` at import; in-memory `JOBS` with running-guard
      (409); `GET /health`, `GET /status`, `POST /discover {limit?}`, `POST /enrich`; job runners
      build their own client, replicate env checks, and turn `SystemExit`/exceptions into job
      `error` state. `enrich` job uses `queued_only(fetch_queued_prospects(...))` (guard intact).
- [x] Run: `poetry run uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000`.

## Task 3 — Backend tests
- [x] `tests/test_enrich.py` — `run_enrichment` updates every queued row with the right status +
      returns correct counts (monkeypatched `enrich_prospect` + `db.update_prospect`).
- [x] `tests/test_api.py` — `TestClient`: `/health`, `/status` shape, and 409-when-running for
      both `POST /discover` and `POST /enrich`.

## Task 4 — Frontend
- [x] `web/next.config.ts` — rewrite `/api/py/:path*` → `PY_API_URL` (default `127.0.0.1:8000`).
- [x] `web/lib/prospectingApi.ts` — `triggerDiscover` / `triggerEnrich` / `fetchJobStatus`.
- [x] `web/hooks/useProspects.ts` — `refresh()` + derived `queuedCount`.
- [x] `web/components/prospecting/ProspectingControls.tsx` — Discover (+ limit), Enrich queued (N)
      with cost confirm, `/status` polling, status line; mounted in `ProspectingPanel`.

## Task 5 — Docs
- [x] README — "Prospecting" run section (env keys, SQL, CLI, uvicorn command).
- [x] This spec + plan.

## Verification (done, except live)
- [x] `poetry run pytest` → 30 passed.
- [x] uvicorn from repo root resolves repo-root `.env`; jobs run in background; missing-key errors
      captured into job state (not 500).
- [x] `tsc` + `eslint` + `next build` clean.
- [x] Playwright: `/api/py/health` via Next proxy → ok; Discover button → proxy → job → poll →
      status line shows the captured error; no crashes.
- [ ] **Live E2E** (needs SQL applied + `GOOGLE_MAPS_API_KEY` + `SERPAPI_KEY`): Discover populates
      prospects; map-select → Enrich queued prices them; results refresh in the UI.

## Out of scope
Auth · persisted job queue / websockets · production deploy + orchestration · script behavior
changes · the map's client-side "Queue" write.
