---
phase: 01-backend-foundation
reviewed: 2026-04-21T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - v1/backend/requirements.txt
  - v1/backend/config.py
  - v1/backend/database.py
  - v1/backend/.env.example
  - v1/backend/.gitignore
  - v1/backend/README.md
  - v1/frontend/README.md
  - v1/backend/routers/__init__.py
  - v1/backend/routers/dogs.py
  - v1/backend/main.py
findings:
  critical: 0
  warning: 2
  info: 3
  total: 5
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-04-21
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Reviewed the Phase 1 FastAPI + Supabase backend scaffold. The overall structure is clean and well-organized: settings validation at startup, a lazy singleton client, a focused dogs router, and a minimal main entrypoint. No hardcoded secrets, no authentication issues (intentionally absent per PRD), and no critical bugs.

Two warnings are present: a CORS configuration that is wider than intended, and unhandled Supabase client errors at the external service boundary. Three info-level items round out the findings. Nothing here blocks the phase from being marked complete.

## Warnings

### WR-01: CORS wildcard negates explicit origin allowlist

**File:** `v1/backend/main.py:21`
**Issue:** `allow_origins=["http://localhost:3000", "*"]` — the `"*"` entry makes the explicit `http://localhost:3000` entry redundant and permits any origin to send cross-origin requests. For an internal tool this is wider than intended, even though `allow_credentials=False` limits cookie exposure.
**Fix:** Remove `"*"` and keep an explicit list. Add the production frontend origin when it is known:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # extend with prod URL in Phase 3
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

---

### WR-02: Supabase errors propagate as unhandled 500s

**File:** `v1/backend/routers/dogs.py:22` and `v1/backend/routers/dogs.py:30-35`
**Issue:** `client.table(...).select(...).execute()` can raise `postgrest.exceptions.APIError` (or a network-level exception) on Supabase failure. Currently this propagates as a raw 500 with internal stack detail visible in the response, which leaks implementation info and is harder to debug from the client side.
**Fix:** Wrap Supabase calls and return a clean 503 with a logged detail:

```python
import logging
from postgrest.exceptions import APIError

logger = logging.getLogger(__name__)

@router.get("/dogs")
def list_dogs() -> dict[str, list[dict[str, Any]]]:
    client = get_supabase_client()
    try:
        response = client.table("dogs").select("id, name").execute()
    except APIError as exc:
        logger.error("Supabase error in list_dogs: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"dogs": response.data}
```

Apply the same pattern to `get_dog`.

---

## Info

### IN-01: `dog_id` parameter accepts any string — UUID validation is free

**File:** `v1/backend/routers/dogs.py:27`
**Issue:** The PRD specifies UUID identifiers. Declaring the parameter as `UUID` gives automatic 422 validation on malformed input and makes the contract explicit, at zero cost.
**Fix:**

```python
from uuid import UUID

@router.get("/dogs/{dog_id}")
def get_dog(dog_id: UUID) -> dict[str, Any]:
    client = get_supabase_client()
    response = (
        client.table("dogs")
        .select(DOG_DETAIL_FIELDS)
        .eq("id", str(dog_id))   # supabase-py expects str
        .execute()
    )
    ...
```

---

### IN-02: Logger instances defined but never used

**File:** `v1/backend/main.py:13`, `v1/backend/routers/dogs.py:7`
**Issue:** `logger = logging.getLogger(__name__)` is declared in both files but never called. This is dead code that signals incomplete error instrumentation (see WR-02).
**Fix:** Either remove the unused loggers, or use them — the most natural place is in the error paths added for WR-02 and for the 404 case in `get_dog`.

---

### IN-03: Global singleton init is not thread-safe under concurrent startup

**File:** `v1/backend/database.py:10-15`
**Issue:** The `check-then-set` pattern on `_client` is a data race if two requests arrive simultaneously before the client is initialized. Under uvicorn's default single-worker mode this is benign, but worth noting for future multi-worker deployment.
**Fix:** Initialize eagerly at module load (simpler) or guard with a lock:

```python
# Option A — eager init at module level (simplest)
from supabase import Client, create_client
from config import settings

_client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)

def get_supabase_client() -> Client:
    return _client
```

---

_Reviewed: 2026-04-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
