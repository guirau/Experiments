# Doggies — Project Guide

## Project

**Doggies v1** — A password-gated, read-only internal web tool for a dog shelter admin.
Frontend: Next.js App Router (TypeScript) in `v1/frontend/`.
Backend: Python FastAPI in `v1/backend/`, connected to a shared Supabase Postgres instance.

## GSD Workflow

This project uses the GSD (Get Shit Done) workflow. Planning artifacts live in `.planning/`.

**Key files:**
- `.planning/PROJECT.md` — living project context
- `.planning/REQUIREMENTS.md` — scoped v1 requirements with traceability
- `.planning/ROADMAP.md` — 3-phase execution plan
- `.planning/STATE.md` — current progress state

**Config:** YOLO mode · Coarse granularity · Parallel execution · Plan Check ✓ · Verifier ✓

## Phase Execution

Always read `.planning/STATE.md` and `.planning/ROADMAP.md` before starting any phase.

```
/gsd-plan-phase 1    → Backend Foundation (FastAPI + Supabase API endpoints)
/gsd-plan-phase 2    → Frontend Auth + Dog List (Next.js login + list screen)
/gsd-plan-phase 3    → Dog Detail + Integration (detail screen + back nav)
```

## Structure

```
Doggies/
├── .planning/          # GSD planning artifacts
├── v0/                 # Telegram bot with Claude agent (unrelated to v1)
├── v1/
│   ├── frontend/       # Next.js App Router, TypeScript
│   ├── backend/        # Python FastAPI + supabase-py
│   └── docs/prd/       # PRD (source of truth for v1 scope)
└── CLAUDE.md           # This file
```

## Key Decisions

- FastAPI is used even though Next.js API routes would suffice — preserves `frontend/`+`backend/` monorepo shape for future expansion
- Supabase project shared with v0 — no RLS, anon key reads freely
- Auth is in-memory React state only — `admin` / `1234`, no persistence, intentional mock
- All dog data is read-only in v1
