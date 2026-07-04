#!/usr/bin/env python3
"""FastAPI service that triggers the Prospecting scripts from the web UI.

It imports and calls the SAME functions the CLIs use (discover.run_discovery / run_scoring,
enrich.run_enrichment), so behavior and the SerpApi cost guard are identical — this is only a
trigger layer, it adds no data logic. Run from the repo root so `.env` + `import db` resolve:

    poetry run uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000

Bind to 127.0.0.1 only: these endpoints spend money (Google Places / SerpApi / Claude) and are
unauthenticated; localhost is the mitigation. Jobs run in the background; poll GET /status.
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

import db
import discover
import enrich

load_dotenv()  # db.get_client() reads env at call time and never loads .env itself

app = FastAPI(title="KP House Finder — Prospecting trigger")


# --- in-memory job state (single-user local tool; resets on restart) --------

def _idle():
    return {"status": "idle", "result": None, "error": None,
            "started_at": None, "finished_at": None}


JOBS = {"discover": _idle(), "enrich": _idle()}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _start(name):
    JOBS[name] = {"status": "running", "result": None, "error": None,
                  "started_at": _now(), "finished_at": None}


def _finish(name, result=None, error=None):
    job = JOBS[name]
    job["status"] = "error" if error else "done"
    job["result"] = result
    job["error"] = error
    job["finished_at"] = _now()


# --- job runners (each builds its own client; errors -> job state, never 500) ---

def _run_discover(limit):
    try:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in .env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
        client = db.get_client()  # may raise SystemExit if SUPABASE_* missing
        upserted = discover.run_discovery(client, api_key, limit)
        scored = discover.run_scoring(client, limit)
        _finish("discover", result={"upserted": upserted, "scored": scored})
    except (Exception, SystemExit) as e:  # noqa: BLE001 - surface as job error, keep serving
        _finish("discover", error=str(e) or e.__class__.__name__)


def _run_enrich():
    try:
        api_key = os.environ.get("SERPAPI_KEY")
        if not api_key:
            raise RuntimeError("SERPAPI_KEY is not set in .env")
        client = db.get_client()
        queued = enrich.queued_only(db.fetch_queued_prospects(client))  # cost guard preserved
        enriched, errored = enrich.run_enrichment(client, api_key, queued)
        _finish("enrich", result={"enriched": enriched, "errored": errored, "total": len(queued)})
    except (Exception, SystemExit) as e:  # noqa: BLE001
        _finish("enrich", error=str(e) or e.__class__.__name__)


# --- endpoints --------------------------------------------------------------

class DiscoverBody(BaseModel):
    limit: int | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    return JOBS


@app.post("/discover")
def start_discover(body: DiscoverBody, bg: BackgroundTasks):
    if JOBS["discover"]["status"] == "running":
        raise HTTPException(status_code=409, detail="A discover run is already in progress.")
    _start("discover")
    bg.add_task(_run_discover, body.limit)
    return {"status": "running", "job": "discover"}


@app.post("/enrich")
def start_enrich(bg: BackgroundTasks):
    if JOBS["enrich"]["status"] == "running":
        raise HTTPException(status_code=409, detail="An enrich run is already in progress.")
    _start("enrich")
    bg.add_task(_run_enrich)
    return {"status": "running", "job": "enrich"}
