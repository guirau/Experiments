# Design — KP House Finder: Prospecting (Google Maps discovery + enrichment)

> Superpowers `brainstorming` output (approved 2026-07-03). The implementation plan
> derived from this lives in `docs/superpowers/plans/2026-07-03-prospecting.md`.

## Context

Today KP House Finder is reactive: the Chrome extension scrapes Facebook/WhatsApp posts →
Supabase `fb_posts` → `src/analyze.py` + Claude → `listings_parsed` → the Next.js dashboard
in `web/`. It only surfaces places whose owners already posted an ad.

This feature adds a **proactive second discovery channel**: find accommodation *businesses*
in Koh Phangan (villas, bungalows, resorts, guesthouses) via the Google Maps/Places API,
place them on a real map, enrich the promising ones with pricing, score their fit for a solo
long-stay tenant, and cold-pitch owners a long-term deal — tracking outreach in the existing
tracker.

**Constraints:** keep it simple and personal, reuse the existing web↔Supabase↔Python rails,
no new backend server, control paid-API spend deliberately.

## Key premise correction (resolved during brainstorming)

The original idea — "use Google Maps to scrape houses and their set prices" — does not hold
as stated, and the design is built on the corrected version:

- The **plain Google Places API returns businesses/POIs, not rental prices.** Booking/Agoda/
  Airbnb rates seen in the Maps UI come from Google's partner-gated Travel Partner API, which
  is closed to us. Prices instead come from **SerpApi's Google Hotels API** (`rate_per_night`,
  `all_offers`, `bedrooms`, `gps_coordinates`).
- Even those are **nightly** prices. The **long-term monthly "set price" is not published
  anywhere** — it is what you negotiate, recorded manually after outreach. Nightly price is a
  quality tier / negotiation anchor only.
- Google Maps/Hotels only indexes **bookable** places, so this channel = "accommodation
  businesses to cold-pitch", categorically distinct from owner-posted FB/WhatsApp listings.

## Confirmed decisions

1. **Hybrid source:** Google Places for broad discovery + contact + location; **SerpApi only
   to enrich the places the user visually selects on the map** (cost control — SerpApi bills
   per call).
2. **Enrichment is queue-based, no new backend server:** the browser marks selected pins
   `enrich_status='queued'` in Supabase; `src/enrich.py` does the SerpApi calls in batch.
3. **Hard safety guard:** `enrich.py` enriches **ONLY** `enrich_status='queued'` rows, never
   anything else, with an `--inspect` dry-run that lists exactly what will be called + the cost
   before spending.
4. **Tracker:** reuse the existing `tracker` table via an "Add to tracker" button on each
   prospect card, linked by a new nullable `tracker.prospect_id` column.
5. **Map:** React-Leaflet + OpenStreetMap tiles (free, no API key, no extra billing).
6. **Claude:** **suitability scoring only** (0–10 + short reason) to rank "best" findings. No
   AI-drafted outreach — outreach stays manual via the card's contact details.

## Approach (chosen): queue-via-status-column, batch Python enrichment

The web app never calls SerpApi directly. Discovery and enrichment are Python CLIs (like
`analyze.py`); the browser only ever reads/writes Supabase. The `prospects.enrich_status`
column does double duty as **both** the map-selection queue and the safety guard: `enrich.py`
filtering strictly on `'queued'` is structurally incapable of touching places the user didn't
pick.

**Alternatives considered and rejected:** *instant in-browser enrichment* via a new Next.js
API route (adds the app's first server surface + duplicates enrichment logic in JS, and spends
SerpApi money on every click); *plain Places only* (no prices — fails the core goal); *SerpApi
only* (misses places not on Google Hotels and spends on everything).

## Data model

### New: `sql/prospects.sql` (STEP-1 create / STEP-2 RLS, mirrors `listings_parsed.sql`)

Keyed by Google `place_id` so re-running discovery upserts instead of duplicating (like
`listings_parsed` keys on `fb_posts.id`).

- **Discovery (Google Places):** `place_id` (PK), `name`, `formatted_address`, `lat`, `lng`,
  `phone`, `website`, `google_rating`, `user_ratings_total`, `property_type`,
  `source default 'google_maps'`.
- **Claude scoring:** `suitability_score` (0–10), `suitability_reason`.
- **Enrichment lifecycle:** `enrich_status default 'discovered'` →
  `discovered` / `queued` / `enriched` / `error`. **The safety guard hinges on this column.**
- **SerpApi enrichment:** `rate_per_night_thb`, `bedrooms`, `bathrooms`,
  `ota_offers` (jsonb `[{source, price, link}]`).
- **Timestamps:** `created_at`, `enriched_at`.
- Indexes on `enrich_status` (queue query) and `suitability_score` (ranking); anon
  select/insert/update RLS.

### Modified: `sql/tracker.sql`

One nullable column, no behavior change: `prospect_id text` (via
`alter table tracker add column if not exists prospect_id text;`). Lets "Add to tracker" link
a prospect card to its row and lets the card show "already tracked".

## Architecture (units & boundaries)

Same split as the rest of the pipeline: **pure functions** (payload builders, queue selection,
SerpApi-response → row mapping) separated from **I/O shells** (Supabase, REST calls) so the
only logic with correctness risk is unit-testable without network.

### Backend — two new CLIs, reusing `db.py` + `extract.py` idioms

- **`src/discover.py`** — Google Places **Text Search (New)** across KP (queries like
  "villas/bungalows/resorts/long-term rental in Koh Phangan"; page via `nextPageToken`), dedupe
  by `place_id`, upsert into `prospects` (reuse `db.py` upsert-on-conflict) with
  `enrich_status='discovered'`. Then a Claude suitability pass over unscored rows (reuse
  `extract.py`'s lazy `Anthropic()` client, `cache_control` caching, batched JSON-array prompt,
  `_coerce`-style clamping) writing `suitability_score` + `suitability_reason`. CLI:
  `--inspect` / `--limit N` / no-arg full run.
- **`src/enrich.py`** (the guarded script) — select **ONLY** `prospects` where
  `enrich_status='queued'`; `--inspect` prints the exact place list + SerpApi call count + rough
  cost, writes nothing; for each queued row call SerpApi **Google Hotels** (`engine=google_hotels`,
  `currency=THB`, near-future check-in/out, vacation-rental offers enabled), best-effort name
  match → `rate_per_night_thb`, `bedrooms`/`bathrooms`, `ota_offers`; on success set
  `enrich_status='enriched'` + `enriched_at`, on no-match/error set `'error'`. Never touches a
  non-queued row (test-enforced).
- **`src/db.py`** gains prospect helpers (`PROSPECTS_TABLE`, fetch/upsert/queue-select) mirroring
  the existing `fb_posts`/`listings_parsed` helpers.

### Frontend — one top-level tab, three subtabs (first subtab pattern in the app)

- **`web/components/listings/Dashboard.tsx`** — extend the `View` union with `"prospecting"`;
  add a standalone tab button (mirror the existing "Tracker" button); add a render branch reusing
  the `max-w-none` / sidebar-hidden treatment already used for `tracker`.
- **`web/components/prospecting/ProspectingPanel.tsx`** — owns
  `subView: "map" | "cards" | "tracker"`.
- **`web/components/prospecting/ProspectingMap.tsx`** — React-Leaflet + OSM, one marker per
  prospect at real lat/lng, click-to-toggle selection, and a "Queue N for enrichment" button →
  `queueForEnrichment(placeIds)`.
- **Cards subtab** — reuse `ListingGrid`/card styling + CSS tokens; each card shows name, area,
  `google_rating` (+count), suitability score + reason, price/room-type once enriched,
  `ota_offers` links, phone/website contact, a Google Maps link, and an **"Add to tracker"**
  button (shows "Tracked" when a `tracker` row with that `prospect_id` exists). Sort by
  suitability score for "best" findings.
- **Tracker subtab** — the existing `TrackerTable` + `useTracker`, now also fed by
  prospect-linked rows.
- **Data layer** — `fetchProspects()`, `queueForEnrichment(placeIds)`
  (`update {enrich_status:'queued'} .in('place_id', ids)`), `createTrackerFromProspect(p)` in
  `web/lib/supabase.ts`; new `web/hooks/useProspects.ts` mirroring `useListings`. New `Prospect`
  type in `web/lib/types.ts`.
- **Deps** — `leaflet`, `react-leaflet`, `@types/leaflet` in `web/package.json` + Leaflet CSS.

## Secrets / dependencies

- `pyproject.toml`: add `requests` for the two REST APIs.
- `.env` / `.env.example`: add `GOOGLE_MAPS_API_KEY`, `SERPAPI_KEY` (same `load_dotenv()`
  pattern; both scripts fail fast if their key is missing).

## Error handling

- Missing `GOOGLE_MAPS_API_KEY` / `SERPAPI_KEY` → fail fast with a clear message (mirror the
  existing `ANTHROPIC_API_KEY` check).
- SerpApi no-match / HTTP error for a queued row → mark that row `enrich_status='error'` and
  continue; never crash the batch, never silently succeed.
- Web: missing `NEXT_PUBLIC_SUPABASE_*` → clear setup message; empty prospects → `EmptyState`.

## Testing (proportionate)

- **Unit (pytest, `pythonpath=["src"]`):** the enrichment **queue-selection guard** (only
  `enrich_status='queued'` rows are ever selected) — the highest-risk, money-touching logic;
  the SerpApi-response → row mapper; the discovery place → `prospects` payload builder. Mirror
  the pure-function style of `tests/test_analyze.py` / `test_db.py`.
- **Manual end-to-end:** the verification steps below (live APIs are not unit-tested, matching
  the project's existing convention).

## Verification (end to end)

1. **Schema:** run `sql/prospects.sql` and the `tracker` `alter` in the Supabase SQL editor;
   confirm the table + `prospect_id` column exist.
2. **Discovery:** `poetry run python src/discover.py --inspect`, then `--limit 5`; confirm
   `prospects` rows with lat/lng + contact + a `suitability_score`.
3. **Guard (critical):** with some rows `discovered` and some `queued`, run
   `poetry run python src/enrich.py --inspect`; confirm it lists **only** the `queued` rows.
   `poetry run pytest` green for the guard test.
4. **Enrichment:** run `enrich.py`; queued rows flip to `enriched` with `rate_per_night_thb` /
   `ota_offers`; non-queued rows untouched.
5. **Frontend:** `cd web && npm run dev` → **Prospecting** tab: map pins at real coordinates,
   selecting pins + "Queue N" sets `enrich_status='queued'`, cards show score/price/contact and
   sort by suitability, "Add to tracker" creates a `tracker` row that appears in the Tracker
   subtab.
6. **Cost sanity:** no SerpApi call happens except for explicitly-queued prospects.

## Out of scope (later slices)

AI-drafted outreach messages · real-time in-browser enrichment / new backend route · Google
Places ToS caching-window handling (single-user tool) · changes to the FB/WhatsApp pipeline,
`listings_parsed`, or the static `AreaMap` · auth · sending outreach from the app.
