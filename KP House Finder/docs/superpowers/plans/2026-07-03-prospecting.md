# KP House Finder Prospecting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** A proactive discovery channel — find accommodation businesses in Koh Phangan via
Google Places, score their solo-long-stay fit with Claude, let the user select places on a real
map to enrich with SerpApi prices, and track cold-outreach in the existing tracker.

**Architecture:** Web↔Supabase↔Python rails, no new backend server. Discovery + enrichment are
Python CLIs (mirroring `analyze.py`); the browser only reads/writes Supabase. A new
`prospects.enrich_status` column is both the map-selection queue and the safety guard —
`enrich.py` filters strictly on `'queued'` so it can never enrich unselected places. Pure
functions (payload builders, queue selection, response mapping) split from I/O for testing.

**Tech Stack:** Python 3.11 + Poetry, `requests`, `anthropic`, `supabase-py`; web is Next.js 16
+ React 19 + Tailwind v4 + `react-leaflet`/`leaflet` + OpenStreetMap.

**Spec:** `docs/superpowers/specs/2026-07-03-prospecting-design.md` (approved 2026-07-03).

**Status (2026-07-03):** All code is written and statically verified — `poetry run pytest`
green (25 tests incl. the enrichment guard); `discover.py`/`enrich.py` import + `--help` OK;
web `tsc`, `eslint`, and `next build` all clean; the Prospecting tab, its 3 subtabs, empty
state, and error handling render at runtime (Playwright smoke test) with no React/Leaflet
crashes. **Not yet done (needs your Supabase project + API keys):** run the two SQL files,
add `GOOGLE_MAPS_API_KEY` + `SERPAPI_KEY` to `.env`, and the live discover→map-select→enrich
→tracker end-to-end (incl. Leaflet tile/pin rendering, which the smoke test could not exercise
without seeded rows). See "Task 8" + the spec's Verification section.

---

## Task 1 — Database schema

- [x] `sql/prospects.sql` — new table keyed by `place_id` (STEP-1 create / STEP-2 anon RLS,
      mirrors `listings_parsed.sql`); columns for discovery, Claude scoring, `enrich_status`
      lifecycle, SerpApi enrichment, timestamps; indexes on `enrich_status` + `suitability_score`.
- [x] `sql/tracker.sql` — add nullable `prospect_id text` (+ `add column if not exists` line).
- [ ] **Verify:** run both in the Supabase SQL editor; confirm table + column exist.

## Task 2 — Secrets & dependencies

- [ ] `.env.example` + `.env` — add `GOOGLE_MAPS_API_KEY`, `SERPAPI_KEY`.
- [ ] `pyproject.toml` — add `requests` to `[project].dependencies`; `poetry install`.

## Task 3 — `src/db.py` prospect helpers

- [ ] Add `PROSPECTS_TABLE = "prospects"` + helpers mirroring the existing ones:
      `upsert_prospects(client, rows)` (upsert on `place_id`), `fetch_queued_prospects(client)`
      (**select ONLY `enrich_status='queued'`**), `existing_prospect_ids`, and a prospects branch
      in `inspect()` (counts per `enrich_status`). Keep them pure-I/O; no business logic.

## Task 4 — `src/discover.py` (Google Places discovery + Claude scoring)

- [ ] Pure: `build_prospect_row(place)` (Places result → `prospects` payload) and
      `place_search_payload(...)`. I/O: `search_places(query)` (Text Search New, `nextPageToken`
      paging, `X-Goog-FieldMask`), dedupe by `place_id`.
- [ ] Claude suitability pass reusing `extract.py` idioms (lazy client, `cache_control`, batched
      JSON array, coerce/clamp to 0–10) → `score_prospects(rows)` returning
      `{suitability_score, suitability_reason}`.
- [ ] CLI (`argparse`): `--inspect` (counts, no writes), `--limit N`, no-arg full run; fail fast
      if `GOOGLE_MAPS_API_KEY` / `ANTHROPIC_API_KEY` missing.
- [ ] **Verify:** `--inspect`, then `--limit 5` populates rows with lat/lng + contact + score.

## Task 5 — `src/enrich.py` (guarded SerpApi enrichment) + test

- [ ] Pure: `map_hotel_result(serp_json)` → `{rate_per_night_thb, bedrooms, bathrooms, ota_offers}`;
      `pick_enrichment(rows)` returning only rows to process.
- [ ] I/O: `serp_hotels(name)` (engine=google_hotels, THB, near-future check-in/out, vacation
      rentals); guarded main that fetches **only `enrich_status='queued'`**, sets `enriched`
      (+`enriched_at`) on success and `error` on no-match/HTTP error, never touching other rows.
- [ ] `--inspect` prints the exact queued list + call count + rough cost, writes nothing; fail
      fast if `SERPAPI_KEY` missing.
- [ ] `tests/test_enrich.py` — assert only `enrich_status='queued'` rows are selected, and
      `map_hotel_result` mapping (mirror `tests/test_analyze.py` style).
- [ ] **Verify:** `poetry run pytest` green; `--inspect` lists only queued rows.

## Task 6 — Frontend data layer

- [ ] `web/lib/types.ts` — `Prospect` type mirroring `prospects` (+ `ota_offers` shape).
- [ ] `web/lib/supabase.ts` — `fetchProspects()` (paginated), `queueForEnrichment(placeIds)`
      (`update {enrich_status:'queued'} .in('place_id', ids)`), `createTrackerFromProspect(p)`
      (insert tracker row prefilled with name/contact/maps-link + `prospect_id`).
- [ ] `web/hooks/useProspects.ts` — mirror `useListings` (fetch once; `{prospects, loading, error}`).

## Task 7 — Prospecting UI

- [ ] `web/components/prospecting/ProspectingMap.tsx` — react-leaflet + OSM, markers at real
      lat/lng, click-to-toggle selection, "Queue N for enrichment" button → `queueForEnrichment`.
- [ ] `web/components/prospecting/ProspectCard.tsx` — reuse card styling/CSS tokens; name, area,
      rating, suitability score + reason, price/rooms once enriched, `ota_offers` links,
      phone/website, Google Maps link, "Add to tracker" ("Tracked" when linked).
- [ ] `web/components/prospecting/ProspectingPanel.tsx` — owns `subView: map | cards | tracker`;
      cards reuse `ListingGrid` layout sorted by suitability; tracker reuses `TrackerTable`.
- [ ] `web/components/listings/Dashboard.tsx` — extend `View` with `"prospecting"`, add the
      standalone tab button (mirror "Tracker"), add render branch with `max-w-none`/no-sidebar.
- [ ] `web/package.json` — add `leaflet`, `react-leaflet`, `@types/leaflet`; import Leaflet CSS.
- [ ] **Verify:** `cd web && npm run dev` → Prospecting tab renders map pins, queue writes
      `enrich_status='queued'`, cards sort by suitability, Add-to-tracker creates a linked row.

## Task 8 — End-to-end verification

- [ ] Full run: discover → select on map → `enrich.py` → prices appear on cards → add to tracker.
- [ ] **Cost sanity:** confirm no SerpApi call fires for any non-queued prospect.

---

## Out of scope (later slices)

AI-drafted outreach · real-time in-browser enrichment · ToS caching-window handling · changes
to the FB/WhatsApp pipeline / `listings_parsed` / static `AreaMap` · auth · sending outreach.
