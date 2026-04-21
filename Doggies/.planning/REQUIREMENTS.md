# Requirements: Doggies v1

**Defined:** 2026-04-21
**Core Value:** Admin can log in and immediately browse dogs, with one click to see full profile details.

## v1 Requirements

### Authentication

- [ ] **AUTH-01**: Admin can submit username `admin` and password `1234` to log in
- [ ] **AUTH-02**: Login failure (wrong credentials) shows an inline error message
- [ ] **AUTH-03**: Auth state is held in React memory only — page reload re-prompts login

### Dog List

- [ ] **LIST-01**: Authenticated admin sees a list of all dogs fetched from `GET /api/dogs`
- [ ] **LIST-02**: Each dog row displays the dog's name only
- [ ] **LIST-03**: Each dog row has a "Dog Details" button

### Dog Detail

- [ ] **DETAIL-01**: Clicking "Dog Details" navigates to the dog detail screen for that dog
- [ ] **DETAIL-02**: Detail screen fetches full dog data from `GET /api/dogs/{id}`
- [ ] **DETAIL-03**: Detail screen renders: name, breed, age_estimate, size, gender, temperament, medical_notes, story, status, intake_date
- [ ] **DETAIL-04**: Photos field — if non-empty array renders `<img>` tags; if empty/null shows "No photos"
- [ ] **DETAIL-05**: Back button returns admin to the dog list screen

### API

- [x] **API-01**: `GET /api/dogs` returns `{ dogs: [{ id: UUID, name: string }] }` from Supabase
- [x] **API-02**: `GET /api/dogs/{id}` returns the full dog object (all fields in §4 of PRD)
- [x] **API-03**: FastAPI connects to Supabase via `supabase-py` using env vars

### Infrastructure

- [x] **INFRA-01**: Monorepo with `v1/frontend/` (Next.js) and `v1/backend/` (FastAPI)
- [ ] **INFRA-02**: `v1/frontend/.env.local` exposes `NEXT_PUBLIC_API_URL=http://localhost:8000`
- [x] **INFRA-03**: `v1/backend/.env` contains `SUPABASE_URL` and `SUPABASE_ANON_KEY` (shared from v0)
- [x] **INFRA-04**: Both services run locally with documented start commands

## v2 Requirements

### Dog List Enhancements

- **LIST-V2-01**: Pagination on the dog list
- **LIST-V2-02**: Filter dogs by status (available, adopted, fostered)
- **LIST-V2-03**: Search dogs by name

### Admin Operations

- **ADMIN-V2-01**: Admin can add a new dog via form
- **ADMIN-V2-02**: Admin can update a dog's status

### Auth

- **AUTH-V2-01**: Real session management (cookie or JWT)
- **AUTH-V2-02**: Supabase Auth integration

## Out of Scope

| Feature | Reason |
|---------|--------|
| Create / update / delete dogs | Read-only v1; writes deferred |
| Supabase Auth | In-memory check sufficient for internal mock tool |
| Photo upload | Not needed for read-only view |
| Telegram integration | v0 handles that separately |
| AI agent / matching logic | Future milestone |
| Role-based access control | Single admin account only |
| Production hardening of auth | Internal tooling only |
| Mobile app | Web-first |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| INFRA-04 | Phase 1 | Complete |
| API-01 | Phase 1 | Complete |
| API-02 | Phase 1 | Complete |
| API-03 | Phase 1 | Complete |
| INFRA-02 | Phase 2 | Pending |
| AUTH-01 | Phase 2 | Pending |
| AUTH-02 | Phase 2 | Pending |
| AUTH-03 | Phase 2 | Pending |
| LIST-01 | Phase 2 | Pending |
| LIST-02 | Phase 2 | Pending |
| LIST-03 | Phase 2 | Pending |
| DETAIL-01 | Phase 3 | Pending |
| DETAIL-02 | Phase 3 | Pending |
| DETAIL-03 | Phase 3 | Pending |
| DETAIL-04 | Phase 3 | Pending |
| DETAIL-05 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-21*
*Last updated: 2026-04-21 — traceability updated after roadmap creation (INFRA-02 moved to Phase 2)*
