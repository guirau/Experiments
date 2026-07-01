# Phase 1: Backend Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-21
**Phase:** 01-backend-foundation
**Areas discussed:** Project structure, Python tooling

---

## Project Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Single main.py | All routes + app in one file — simple, readable | |
| Modular layout | main.py + config.py + database.py + routers/dogs.py | ✓ |

**User's choice:** Modular layout
**Notes:** User selected the modular layout preview showing `main.py`, `config.py`,
`database.py`, and `routers/dogs.py`.

---

## Python Tooling

| Option | Description | Selected |
|--------|-------------|----------|
| requirements.txt | Classic pip install -r, zero overhead | ✓ |
| uv + pyproject.toml | Modern fast installer, more setup | |
| Poetry | Full dependency manager, overkill for 2 routes | |

**User's choice:** requirements.txt
**Notes:** Went with the recommended default — keeps setup commands obvious.

---

## Claude's Discretion

- CORS setup — user skipped; Claude decides (allow-all for local dev)
- Error handling — user skipped; Claude decides 404 shape
- Python version — not asked; Claude picks 3.11+
- Supabase client lifespan pattern — not asked; Claude decides

## Deferred Ideas

None.
