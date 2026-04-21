---
phase: 1
slug: 01-backend-foundation
status: complete
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-21
---

# Phase 1 — Validation Strategy

> Per-phase validation contract reconstructed from Phase 1 PLAN/SUMMARY artifacts (State B).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (installed in `.venv`) |
| **Config file** | none — not yet configured |
| **Quick run command** | `cd v1/backend && source .venv/bin/activate && pytest tests/ -q` |
| **Full suite command** | `cd v1/backend && source .venv/bin/activate && pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds (unit tests, mocked Supabase) |

---

## Sampling Rate

- **After every task commit:** Run quick run command
- **After every plan wave:** Run full suite command
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-T1 | 01 | 1 | INFRA-01 | — | N/A | static | `test -d v1/backend/routers && test -d v1/frontend && echo PASS` | ✅ | ✅ green |
| 01-01-T2 | 01 | 1 | INFRA-03, API-03 | T-01-01 | .env gitignored; no credentials in example | static | `grep SUPABASE_ANON_KEY v1/backend/config.py v1/backend/.env.example` | ✅ | ✅ green |
| 01-01-T3 | 01 | 1 | INFRA-04 | — | N/A | static | `grep "uvicorn main:app" v1/backend/README.md` | ✅ | ✅ green |
| 01-02-T1 | 02 | 2 | API-01, API-02 | T-02-01 | parameterized .eq() filter; malformed UUID → 404 | static | `grep "HTTPException(status_code=404" v1/backend/routers/dogs.py` | ✅ | ✅ green |
| 01-02-T2 | 02 | 2 | API-01, API-02 | T-02-02 | CORS explicit + wildcard, no credentials | static | `grep "localhost:3000" v1/backend/main.py` | ✅ | ✅ green |
| 01-02-CK | 02 | 2 | API-01, API-02 | — | live endpoints return correct shapes | manual | `curl http://localhost:8000/api/dogs` | N/A | ✅ approved |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

None — user elected to keep all behavioral verification as manual-only (see Manual-Only section).
Static grep-based checks in the plan `<automated>` blocks cover structural correctness.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `GET /api/dogs` returns `{ dogs: [{id, name}] }` from live Supabase | API-01 | Requires live Supabase credentials + running server; user-approved in Plan 02 human-verify checkpoint | `curl http://localhost:8000/api/dogs` — expect JSON with `dogs` array |
| `GET /api/dogs/{id}` returns full 12-field object | API-02 | Requires live data; user-approved in Plan 02 checkpoint | `curl http://localhost:8000/api/dogs/{real_uuid}` — expect all 12 PRD fields |
| `GET /api/dogs/{bad_uuid}` returns HTTP 404 | API-02 | Requires running server; user-approved in Plan 02 checkpoint | `curl -o /dev/null -w "%{http_code}" http://localhost:8000/api/dogs/00000000-0000-0000-0000-000000000000` — expect `404` |
| FastAPI reads SUPABASE_URL and SUPABASE_ANON_KEY at startup | API-03 | Requires populated .env and uvicorn process; implicit in server starting cleanly | Start uvicorn — if .env missing/empty it raises RuntimeError on startup |
| `uvicorn main:app --reload --port 8000` starts cleanly | INFRA-04 | Runtime behavior; user-approved in Plan 02 checkpoint | Run from `v1/backend/` with venv active — expect no startup errors |
| Monorepo shape: `v1/frontend/` and `v1/backend/` both exist | INFRA-01 | Filesystem structure; static grep already confirmed | `ls v1/` — expect `frontend/` and `backend/` |

---

## Validation Audit 2026-04-21

| Metric | Count |
|--------|-------|
| Gaps found | 6 |
| Resolved (automated) | 0 |
| Escalated to manual-only | 6 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify blocks in the plan
- [x] Wave 0 not needed (static checks sufficient for infrastructure phase)
- [x] Human-verify checkpoint in Plan 02 approved by user
- [ ] No automated pytest suite yet (manual-only elected)
- [ ] `nyquist_compliant: true` — **not set** (manual-only coverage)

**Approval:** manual-only approved 2026-04-21
