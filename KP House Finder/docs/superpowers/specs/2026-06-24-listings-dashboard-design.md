# Design — KP House Finder: Next.js listings dashboard (V1)

> Superpowers `brainstorming` output (approved 2026-06-24). The implementation plan
> derived from this will live in `docs/superpowers/plans/`.

## Context

The data pipeline is complete: `listings_parsed` in Supabase holds 2624 parsed rows
(1805 offers, `discard_reason IS NULL`) with the full tiered schema documented in
`docs/FIELD_SCHEMA.md`. The user wants a **personal "real-estate website"** to browse and
**filter** those listings by the stored columns — to find a long-term Koh Phangan rental
(**Goal 1:** a full 12 months including high season) and to spot **sublet potential**
(**Goal 1b:** Airbnb/Booking income). It replaces hand-querying the table in SQL.

**Constraints:** keep it simple, personal use, runs locally, Next.js.

## Confirmed decisions
- **Layout:** filter sidebar + responsive **card grid** (real-estate feel).
- **Filters:** EVERY `listings_parsed` column gets a UI control, ordered by relevance —
  Core first, then goal-critical (long-term, sublet), then the rest.
- **Hosting:** local only (`npm run dev`); keep deploy-ready, no Vercel yet.
- **Read-only dashboard:** it queries `listings_parsed`; it never writes.

## Approach (chosen): client-side, fetch-all-once + in-memory filtering

On load, fetch all offers once and filter/sort **in the browser** — instant, no
per-filter network round-trips, no API layer. The dataset is small and bounded
(~1805 rows × ~40 columns ≈ a few hundred KB). Filter state is mirrored in the **URL
query string** so views are shareable and survive reload (web "URL-as-state" pattern).

Alternatives considered and rejected: **server-side filtering** (a round-trip per filter,
more code, loading states — overkill at this scale) and a **build-time static snapshot**
(data goes stale as the pipeline adds rows).

**Critical correctness note:** the Supabase read MUST paginate via `.range()`. PostgREST
caps a single response at 1000 rows and there are 1805 offers; a naive `.select()` would
silently drop ~800 listings. Port the loop from `db.py:paginate`.

## Stack & conventions
- **Next.js (App Router, 15.x) + TypeScript + Tailwind CSS v4.**
- **supabase-js**, read-only, anon key exposed as `NEXT_PUBLIC_SUPABASE_URL` /
  `NEXT_PUBLIC_SUPABASE_ANON_KEY` in `web/.env.local`. The anon key is already a
  client-side secret (the Chrome extension uses it); RLS grants anon `SELECT` on
  `listings_parsed`.
- **Intentional design, not a stock template** (web design-quality rules): light
  "island editorial" aesthetic, real type-scale hierarchy, a warm palette via CSS custom
  properties in `globals.css`, card depth + designed hover/focus/active states, semantic
  HTML, compositor-friendly transitions. Light mode (not dark-by-default).
- Feature-folder organization; small, focused files.

## Architecture (units & boundaries)

The system splits into a **pure filtering core** and a **thin UI/IO shell** so the only
logic with real correctness risk is unit-testable without a browser.

- `lib/types.ts` — `Listing` type mirroring `listings_parsed`, **plus joined
  `link` / `url`** (these live in `fb_posts`, not `listings_parsed`).
- `lib/supabase.ts` — supabase-js client + `fetchOffers()`: paginated `.range()` loop
  over `listings_parsed` (`discard_reason IS NULL`), **and** a paginated fetch of
  `fb_posts(id, link, url)` merged onto each listing by `id` (so cards can link to the
  original post — `listings_parsed` doesn't carry the post URL; `id` is sometimes a URL
  but sometimes an `fbid_<postid>` token, so we join rather than rely on it).
- `lib/filters.ts` — **pure core**: `FilterState`, `defaultFilters`, `applyFilters(rows,
  state)` predicate, `sortListings`, and the `filtersToSearchParams` / `searchParamsToFilters`
  URL codec.
- `lib/areas.ts` — `area_canonical` slug → display name (from `docs/FIELD_SCHEMA.md`).
- `hooks/useListings.ts` — fetch once; expose `{ listings, loading, error }`.
- `hooks/useFilters.ts` — filter state synced to the URL (read on load, push on change).
- `components/filters/*` — `FilterSidebar` composing `PriceRange`, `MultiSelect`,
  `TriStateToggle`, `AmenityToggles`.
- `components/listings/*` — `ListingGrid`, `ListingCard`, `ListingDetails` (expand to all
  populated fields), `SortBar`.
- `components/ui/*` — `Chip`, `Badge`, `EmptyState`, `Skeleton`.
- `app/page.tsx` — server shell rendering a client `<Dashboard/>` island that wires
  `useListings` + `useFilters` → `applyFilters` → `ListingGrid`.

```
web/
  app/        layout.tsx · page.tsx · globals.css
  components/ filters/ · listings/ · ui/
  lib/        types.ts · supabase.ts · filters.ts · areas.ts
  hooks/      useListings.ts · useFilters.ts
```

## Filters (V1, ordered by relevance)
1. **Core** — `price_thb` (min/max range), `area_canonical` (multiselect chips),
   `property_type` (multiselect), `bedrooms` (min), `bathrooms` (min).
2. **Long-term (Goal 1)** — `year_round` (tri-state), `season` (multiselect),
   `min_stay_months` (max acceptable).
3. **Sublet (Goal 1b)** — `subletting_allowed` (tri-state).
4. **Cost** — `deposit_thb` (max), `water_included` / `internet_included` (tri-state).
5. **Amenities** — `has_aircon` · `has_wifi` · `has_pool` · `has_kitchen` · `has_parking`
   · `furnished` · `pet_friendly` · `sea_view` · `has_workspace` · `has_terrace`
   (toggles) · `near_construction`.
6. **Quality / meta** — `parse_confidence` (multiselect), `post_language` (multiselect).

**Nullable-boolean filters are tri-state** (Any / Yes / No), default **Any**, so
`NULL`=unknown rows are never hidden unless the user opts in. "Clear all" resets to
defaults. A result count + active-filter chips sit above the grid.

## Card & sort
- **ListingCard:** price (₿/month, or "ask" when null) · area · `property_type`·beds·baths
  · goal badges (`year_round`, `subletting_allowed`, `season`) · confidence dot ·
  "view ↗" to the original FB post (joined `link` / `url`; fall back to `id` when it is
  itself a URL) · click to expand `ListingDetails` (all populated fields).
- **Sort:** newest (`listed_at` desc, default), price asc/desc, confidence.
- **Scope:** offers only by default (`discard_reason IS NULL`); a toggle can reveal flagged.

## Error handling
- Missing `NEXT_PUBLIC_SUPABASE_*` → a clear setup message, not a blank page.
- Fetch failure → visible error state with a retry.
- Zero matches → `EmptyState` with a "clear filters" action.

## Testing (proportionate)
- **Unit (Vitest):** `lib/filters.ts` — `applyFilters` (price range, area multiselect,
  tri-state nullable bools **including NULL handling**, min/max bounds) and the
  filters↔URL round-trip. These pure functions hold the real logic.
- **Smoke (optional Playwright):** dashboard loads, cards render, a price filter reduces
  the count; visual spot-check at 375 / 1024 / 1440.

## Verification (end to end)
1. `cd web && npm install && npm run dev` → dashboard loads with a card grid.
2. Result count equals the offers in Supabase (~1805) — proves pagination didn't truncate
   at 1000.
3. Set price ≤ 12000 + area=srithanu + year_round=Yes → grid + count update instantly;
   the URL reflects the filters; reload preserves them.
4. A card's "view ↗" opens the original FB post; expand shows all populated fields.
5. `npm run test` green for `lib/filters.ts`.

## Out of scope (later slices)
Vercel deploy · WhatsApp listings · cross-source dedup · saved searches / favorites ·
map view · auth (personal/local) · editing data (read-only dashboard).
