# KP House Finder Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local, personal Next.js dashboard to browse and filter the ~1805 offer rows in Supabase `listings_parsed`, to find a long-term Koh Phangan rental (and spot sublet potential).

**Architecture:** Client-side fetch-all-once + in-memory filtering. On load, fetch every offer (paginated past PostgREST's 1000-row cap) plus `fb_posts(id,link,url)` to merge post links; filter/sort in the browser; mirror filter state in the URL. A pure `lib/filters.ts` core holds all logic (unit-tested with Vitest); React components are thin. Left filter rail + responsive 3-up card grid.

**Tech Stack:** Next.js 15 (App Router), TypeScript, Tailwind CSS v4, `@supabase/supabase-js`, Vitest.

**Spec:** `docs/superpowers/specs/2026-06-24-listings-dashboard-design.md` (approved).

---

## File structure (all under `web/`)

```
web/
  app/        layout.tsx · page.tsx · globals.css
  components/
    filters/  FilterSidebar.tsx · PriceRange.tsx · MultiSelect.tsx · TriStateToggle.tsx · NumberMax.tsx · AmenityToggles.tsx
    listings/ Dashboard.tsx · ListingGrid.tsx · ListingCard.tsx · ListingDetails.tsx · SortBar.tsx · ActiveChips.tsx
    ui/       Chip.tsx · Badge.tsx · EmptyState.tsx · Skeleton.tsx
  lib/        types.ts · constants.ts · areas.ts · filters.ts · filters.test.ts · supabase.ts · merge.ts · merge.test.ts
  hooks/      useListings.ts · useFilters.ts
  .env.example · .env.local (user-created, gitignored)
```

Responsibilities: `lib/filters.ts` = pure filter/sort/URL logic (tested). `lib/merge.ts` = pure link-merge (tested). `lib/supabase.ts` = network I/O only. `hooks/*` = React glue. `components/*` = presentation.

---

## Shared definitions (used across tasks)

**`web/lib/types.ts`** — the row shape (mirrors `listings_parsed` + joined `link`/`url`):

```typescript
export type Tri = "any" | "yes" | "no";

export interface Listing {
  // control / meta
  id: string;
  source_table: string | null;
  source: string | null;
  listed_at: string | null;
  raw_text: string | null;
  parser_version: string | null;
  discard_reason: string | null;
  // classification
  is_offer: string | null;
  post_language: string | null;
  parse_confidence: string | null;
  multi_listing: boolean | null;
  // tier 1
  price_thb: number | null;
  price_low_thb: number | null;
  price_high_thb: number | null;
  price_period: string | null;
  season: string | null;
  bedrooms: number | null;
  bathrooms: number | null;
  property_type: string | null;
  area_raw: string | null;
  area_canonical: string | null;
  // tier 2
  min_stay_months: number | null;
  available_from: string | null;
  available_until: string | null;
  year_round: boolean | null;
  subletting_allowed: boolean | null;
  // tier 3
  deposit_thb: number | null;
  electricity_rate_thb_per_unit: number | null;
  water_included: boolean | null;
  internet_included: boolean | null;
  // tier 4
  has_aircon: boolean | null;
  has_wifi: boolean | null;
  furnished: boolean | null;
  has_kitchen: boolean | null;
  has_pool: boolean | null;
  has_parking: boolean | null;
  pet_friendly: boolean | null;
  sea_view: boolean | null;
  has_workspace: boolean | null;
  has_terrace: boolean | null;
  near_road: boolean | null;
  near_construction: string | null;
  furnishings_list: string | null;
  // tier 5
  contact_raw: string | null;
  contact_phone: string | null;
  size_sqm: number | null;
  // joined from fb_posts
  link: string | null;
  url: string | null;
}

export type SortKey = "newest" | "price_asc" | "price_desc" | "confidence";

export interface FilterState {
  priceMin: number | null;
  priceMax: number | null;
  areas: string[];
  propertyTypes: string[];
  bedroomsMin: number | null;
  bathroomsMin: number | null;
  yearRound: Tri;
  seasons: string[];
  minStayMax: number | null;
  subletting: Tri;
  depositMax: number | null;
  waterIncluded: Tri;
  internetIncluded: Tri;
  amenities: string[]; // listing[field] must be === true for each
  confidences: string[];
  languages: string[];
  sort: SortKey;
}
```

**Filtering conventions (the rules the tests enforce):**
- **Numeric range filters** (`priceMin/Max`, `bedroomsMin`, `bathroomsMin`, `minStayMax`, `depositMax`): a listing whose value is `null` (unknown) **passes** — unknowns are never hidden; use sort to prioritise known values.
- **Tri-state** (`yearRound`, `subletting`, `waterIncluded`, `internetIncluded`): `"any"` passes all; `"yes"` requires `=== true`; `"no"` requires `=== false`. (`null` is hidden only when the user opts into yes/no.)
- **Multiselect** (`areas`, `propertyTypes`, `seasons`, `confidences`, `languages`): empty = no constraint; otherwise the listing's value must be in the set.
- **Amenities**: for each field in the list, `listing[field] === true`.

**`web/lib/constants.ts`** — enum option lists for the controls (copy from `docs/FIELD_SCHEMA.md`):

```typescript
export const PROPERTY_TYPES = ["house", "villa", "bungalow", "apartment", "studio", "room", "unknown"];
export const SEASONS = ["low", "high", "full_year", "unknown"];
export const CONFIDENCES = ["high", "medium", "low"];
export const LANGUAGES = ["en", "th", "mixed", "other"];
export const AMENITY_FIELDS: { field: keyof import("./types").Listing; label: string }[] = [
  { field: "has_aircon", label: "Aircon" }, { field: "has_wifi", label: "Wifi" },
  { field: "has_pool", label: "Pool" }, { field: "has_kitchen", label: "Kitchen" },
  { field: "has_parking", label: "Parking" }, { field: "furnished", label: "Furnished" },
  { field: "pet_friendly", label: "Pet friendly" }, { field: "sea_view", label: "Sea view" },
  { field: "has_workspace", label: "Workspace" }, { field: "has_terrace", label: "Terrace" },
];
export const AREA_ENUM = ["thong_sala","ban_tai","ban_kai","haad_rin","srithanu","chaloklum","mae_haad","hin_kong","woktum","haad_yao","haad_salad","haad_son","thong_nai_pan","bottle_beach","than_sadet","haad_yuan_tien","madeua_wan","plai_laem","other","unknown"];
```

---

## Task 1: Scaffold the `web/` app

**Files:** create `web/` (via create-next-app), `web/vitest.config.ts`, `web/.env.example`; modify `web/package.json`, root `.gitignore`.

- [ ] **Step 1: Scaffold Next.js**

Run (from repo root):
```bash
npx create-next-app@latest web --typescript --tailwind --app --eslint --no-src-dir --import-alias "@/*" --use-npm
```
Accept defaults if prompted. Expected: `web/` with `app/`, `package.json`, Tailwind v4 wired (`app/globals.css` has `@import "tailwindcss";`).

- [ ] **Step 2: Install runtime + test deps**

```bash
cd web && npm install @supabase/supabase-js && npm install -D vitest
```

- [ ] **Step 3: Add Vitest config + test script**

Create `web/vitest.config.ts`:
```typescript
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { environment: "node", include: ["lib/**/*.test.ts"] },
});
```
In `web/package.json`, add to `"scripts"`: `"test": "vitest run"`.

- [ ] **Step 4: Env example + gitignore**

Create `web/.env.example`:
```
NEXT_PUBLIC_SUPABASE_URL=https://your-project-ref.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```
Confirm `web/.gitignore` (created by create-next-app) ignores `.env*`. Run: `cd web && git check-ignore .env.local` → expect `.env.local`. If not ignored, append `.env*.local` to `web/.gitignore`.

- [ ] **Step 5: Verify the app boots**

Run: `cd web && npm run dev` then open http://localhost:3000 — expect the Next.js starter page. Stop the server (Ctrl-C).

- [ ] **Step 6: Commit**

```bash
git add web .gitignore && git commit -m "chore: scaffold web/ Next.js dashboard app + vitest"
```

---

## Task 2: Types, constants, areas

**Files:** Create `web/lib/types.ts`, `web/lib/constants.ts`, `web/lib/areas.ts`.

- [ ] **Step 1: Create `web/lib/types.ts`**

Paste the full `Listing`, `Tri`, `SortKey`, `FilterState` definitions from "Shared definitions" above.

- [ ] **Step 2: Create `web/lib/constants.ts`**

Paste the full constants block from "Shared definitions" above.

- [ ] **Step 3: Create `web/lib/areas.ts`**

```typescript
const NAMES: Record<string, string> = {
  thong_sala: "Thong Sala", ban_tai: "Ban Tai", ban_kai: "Ban Kai", haad_rin: "Haad Rin",
  srithanu: "Sri Thanu", chaloklum: "Chaloklum", mae_haad: "Mae Haad", hin_kong: "Hin Kong",
  woktum: "Woktum", haad_yao: "Haad Yao", haad_salad: "Haad Salad", haad_son: "Haad Son",
  thong_nai_pan: "Thong Nai Pan", bottle_beach: "Bottle Beach", than_sadet: "Than Sadet",
  haad_yuan_tien: "Haad Yuan/Tien", madeua_wan: "Madeua Wan", plai_laem: "Plai Laem",
  other: "Other", unknown: "Unknown",
};
export function areaName(slug: string | null): string {
  if (!slug) return "Unknown";
  return NAMES[slug] ?? slug;
}
```

- [ ] **Step 4: Commit**

```bash
git add web/lib && git commit -m "feat: dashboard types, enum constants, area names"
```

---

## Task 3: Filter core (`lib/filters.ts`) — TDD

**Files:** Create `web/lib/filters.test.ts`, then `web/lib/filters.ts`.

- [ ] **Step 1: Write the failing tests**

Create `web/lib/filters.test.ts`:
```typescript
import { describe, it, expect } from "vitest";
import { defaultFilters, applyFilters, sortListings, filtersToParams, paramsToFilters } from "./filters";
import type { Listing, FilterState } from "./types";

function L(over: Partial<Listing>): Listing {
  return { id: "x", source_table: null, source: null, listed_at: null, raw_text: null,
    parser_version: null, discard_reason: null, is_offer: null, post_language: null,
    parse_confidence: null, multi_listing: null, price_thb: null, price_low_thb: null,
    price_high_thb: null, price_period: null, season: null, bedrooms: null, bathrooms: null,
    property_type: null, area_raw: null, area_canonical: null, min_stay_months: null,
    available_from: null, available_until: null, year_round: null, subletting_allowed: null,
    deposit_thb: null, electricity_rate_thb_per_unit: null, water_included: null,
    internet_included: null, has_aircon: null, has_wifi: null, furnished: null,
    has_kitchen: null, has_pool: null, has_parking: null, pet_friendly: null, sea_view: null,
    has_workspace: null, has_terrace: null, near_road: null, near_construction: null,
    furnishings_list: null, contact_raw: null, contact_phone: null, size_sqm: null,
    link: null, url: null, ...over };
}
const f = (over: Partial<FilterState> = {}): FilterState => ({ ...defaultFilters(), ...over });

describe("applyFilters", () => {
  it("returns all rows with default filters", () => {
    const rows = [L({ price_thb: 5000 }), L({ price_thb: 9000 })];
    expect(applyFilters(rows, f())).toHaveLength(2);
  });

  it("price range keeps in-range and keeps null price (unknown not hidden)", () => {
    const rows = [L({ price_thb: 5000 }), L({ price_thb: 20000 }), L({ price_thb: null })];
    const out = applyFilters(rows, f({ priceMax: 10000 }));
    expect(out.map(r => r.price_thb).sort()).toEqual([5000, null].sort());
  });

  it("area multiselect constrains to selected slugs", () => {
    const rows = [L({ area_canonical: "srithanu" }), L({ area_canonical: "ban_tai" })];
    expect(applyFilters(rows, f({ areas: ["srithanu"] }))).toHaveLength(1);
  });

  it("tri-state yes requires true and hides null+false", () => {
    const rows = [L({ year_round: true }), L({ year_round: false }), L({ year_round: null })];
    expect(applyFilters(rows, f({ yearRound: "yes" })).map(r => r.year_round)).toEqual([true]);
  });

  it("tri-state no requires false", () => {
    const rows = [L({ subletting_allowed: true }), L({ subletting_allowed: false })];
    expect(applyFilters(rows, f({ subletting: "no" })).map(r => r.subletting_allowed)).toEqual([false]);
  });

  it("bedroomsMin keeps >= and keeps null", () => {
    const rows = [L({ bedrooms: 1 }), L({ bedrooms: 3 }), L({ bedrooms: null })];
    const out = applyFilters(rows, f({ bedroomsMin: 2 })).map(r => r.bedrooms);
    expect(out.sort()).toEqual([3, null].sort());
  });

  it("minStayMax keeps <= and keeps null", () => {
    const rows = [L({ min_stay_months: 2 }), L({ min_stay_months: 12 }), L({ min_stay_months: null })];
    const out = applyFilters(rows, f({ minStayMax: 6 })).map(r => r.min_stay_months);
    expect(out.sort()).toEqual([2, null].sort());
  });

  it("amenities require each field true", () => {
    const rows = [L({ has_pool: true, has_wifi: true }), L({ has_pool: true, has_wifi: null })];
    expect(applyFilters(rows, f({ amenities: ["has_pool", "has_wifi"] }))).toHaveLength(1);
  });

  it("combines filters with AND", () => {
    const rows = [
      L({ price_thb: 8000, area_canonical: "srithanu", year_round: true }),
      L({ price_thb: 8000, area_canonical: "ban_tai", year_round: true }),
    ];
    expect(applyFilters(rows, f({ priceMax: 10000, areas: ["srithanu"], yearRound: "yes" }))).toHaveLength(1);
  });
});

describe("sortListings", () => {
  it("price_asc puts nulls last", () => {
    const rows = [L({ price_thb: null }), L({ price_thb: 9000 }), L({ price_thb: 5000 })];
    expect(sortListings(rows, "price_asc").map(r => r.price_thb)).toEqual([5000, 9000, null]);
  });
  it("newest sorts listed_at desc", () => {
    const rows = [L({ listed_at: "2026-01-01" }), L({ listed_at: "2026-06-01" })];
    expect(sortListings(rows, "newest").map(r => r.listed_at)).toEqual(["2026-06-01", "2026-01-01"]);
  });
  it("confidence orders high > medium > low", () => {
    const rows = [L({ parse_confidence: "low" }), L({ parse_confidence: "high" }), L({ parse_confidence: "medium" })];
    expect(sortListings(rows, "confidence").map(r => r.parse_confidence)).toEqual(["high", "medium", "low"]);
  });
});

describe("URL codec round-trip", () => {
  it("survives filters -> params -> filters", () => {
    const state = f({ priceMin: 5000, priceMax: 15000, areas: ["srithanu", "ban_tai"],
      yearRound: "yes", subletting: "no", bedroomsMin: 2, amenities: ["has_pool"],
      seasons: ["full_year"], sort: "price_asc" });
    expect(paramsToFilters(filtersToParams(state))).toEqual(state);
  });
  it("empty params yields defaults", () => {
    expect(paramsToFilters(new URLSearchParams())).toEqual(defaultFilters());
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd web && npx vitest run lib/filters.test.ts`
Expected: FAIL — cannot find module `./filters` / exports undefined.

- [ ] **Step 3: Implement `web/lib/filters.ts`**

```typescript
import type { Listing, FilterState, Tri, SortKey } from "./types";

export function defaultFilters(): FilterState {
  return { priceMin: null, priceMax: null, areas: [], propertyTypes: [], bedroomsMin: null,
    bathroomsMin: null, yearRound: "any", seasons: [], minStayMax: null, subletting: "any",
    depositMax: null, waterIncluded: "any", internetIncluded: "any", amenities: [],
    confidences: [], languages: [], sort: "newest" };
}

const geOrNull = (v: number | null, min: number | null) => min == null || v == null || v >= min;
const leOrNull = (v: number | null, max: number | null) => max == null || v == null || v <= max;
const inSet = (v: string | null, set: string[]) => set.length === 0 || (v != null && set.includes(v));
function tri(v: boolean | null, t: Tri) {
  if (t === "any") return true;
  if (t === "yes") return v === true;
  return v === false;
}

export function applyFilters(rows: Listing[], s: FilterState): Listing[] {
  return rows.filter((r) =>
    leOrNull(r.price_thb, s.priceMax) && geOrNull(r.price_thb, s.priceMin) &&
    inSet(r.area_canonical, s.areas) && inSet(r.property_type, s.propertyTypes) &&
    geOrNull(r.bedrooms, s.bedroomsMin) && geOrNull(r.bathrooms, s.bathroomsMin) &&
    tri(r.year_round, s.yearRound) && inSet(r.season, s.seasons) &&
    leOrNull(r.min_stay_months, s.minStayMax) && tri(r.subletting_allowed, s.subletting) &&
    leOrNull(r.deposit_thb, s.depositMax) && tri(r.water_included, s.waterIncluded) &&
    tri(r.internet_included, s.internetIncluded) &&
    s.amenities.every((a) => (r as unknown as Record<string, unknown>)[a] === true) &&
    inSet(r.parse_confidence, s.confidences) && inSet(r.post_language, s.languages)
  );
}

const CONF_RANK: Record<string, number> = { high: 3, medium: 2, low: 1 };
const nlast = (v: number | null) => (v == null ? Infinity : v);

export function sortListings(rows: Listing[], sort: SortKey): Listing[] {
  const out = [...rows];
  if (sort === "price_asc") out.sort((a, b) => nlast(a.price_thb) - nlast(b.price_thb));
  else if (sort === "price_desc") out.sort((a, b) => (b.price_thb ?? -Infinity) - (a.price_thb ?? -Infinity));
  else if (sort === "confidence") out.sort((a, b) => (CONF_RANK[b.parse_confidence ?? ""] ?? 0) - (CONF_RANK[a.parse_confidence ?? ""] ?? 0));
  else out.sort((a, b) => (b.listed_at ?? "").localeCompare(a.listed_at ?? "")); // newest
  return out;
}

// ---- URL codec ----
const CSV = (a: string[]) => a.join(",");
const unCSV = (s: string | null) => (s ? s.split(",").filter(Boolean) : []);
const numOrNull = (s: string | null) => (s != null && s !== "" ? Number(s) : null);

export function filtersToParams(s: FilterState): URLSearchParams {
  const p = new URLSearchParams();
  const d = defaultFilters();
  const setNum = (k: string, v: number | null) => v != null && p.set(k, String(v));
  const setArr = (k: string, v: string[]) => v.length && p.set(k, CSV(v));
  const setTri = (k: string, v: Tri) => v !== "any" && p.set(k, v);
  setNum("priceMin", s.priceMin); setNum("priceMax", s.priceMax);
  setArr("areas", s.areas); setArr("types", s.propertyTypes);
  setNum("bedsMin", s.bedroomsMin); setNum("bathsMin", s.bathroomsMin);
  setTri("yearRound", s.yearRound); setArr("seasons", s.seasons);
  setNum("minStayMax", s.minStayMax); setTri("sublet", s.subletting);
  setNum("depositMax", s.depositMax); setTri("water", s.waterIncluded);
  setTri("internet", s.internetIncluded); setArr("amenities", s.amenities);
  setArr("conf", s.confidences); setArr("lang", s.languages);
  if (s.sort !== d.sort) p.set("sort", s.sort);
  return p;
}

export function paramsToFilters(p: URLSearchParams): FilterState {
  const d = defaultFilters();
  const tg = (k: string, fb: Tri): Tri => { const v = p.get(k); return v === "yes" || v === "no" ? v : fb; };
  return { ...d,
    priceMin: numOrNull(p.get("priceMin")), priceMax: numOrNull(p.get("priceMax")),
    areas: unCSV(p.get("areas")), propertyTypes: unCSV(p.get("types")),
    bedroomsMin: numOrNull(p.get("bedsMin")), bathroomsMin: numOrNull(p.get("bathsMin")),
    yearRound: tg("yearRound", d.yearRound), seasons: unCSV(p.get("seasons")),
    minStayMax: numOrNull(p.get("minStayMax")), subletting: tg("sublet", d.subletting),
    depositMax: numOrNull(p.get("depositMax")), waterIncluded: tg("water", d.waterIncluded),
    internetIncluded: tg("internet", d.internetIncluded), amenities: unCSV(p.get("amenities")),
    confidences: unCSV(p.get("conf")), languages: unCSV(p.get("lang")),
    sort: (p.get("sort") as SortKey) || d.sort };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd web && npx vitest run lib/filters.test.ts`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit**

```bash
git add web/lib/filters.ts web/lib/filters.test.ts && git commit -m "feat: pure filter/sort/URL core with unit tests"
```

---

## Task 4: Supabase fetch + link merge

**Files:** Create `web/lib/merge.test.ts`, `web/lib/merge.ts`, `web/lib/supabase.ts`.

- [ ] **Step 1: Write the failing merge test**

Create `web/lib/merge.test.ts`:
```typescript
import { describe, it, expect } from "vitest";
import { mergeLinks } from "./merge";
import type { Listing } from "./types";

const L = (id: string): Listing => ({ ...({} as Listing), id, link: null, url: null });

describe("mergeLinks", () => {
  it("attaches link/url from fb_posts by id", () => {
    const offers = [L("a"), L("b")];
    const posts = [{ id: "a", link: "L-a", url: "U-a" }, { id: "c", link: "L-c", url: "U-c" }];
    const out = mergeLinks(offers, posts);
    expect(out.find((r) => r.id === "a")).toMatchObject({ link: "L-a", url: "U-a" });
    expect(out.find((r) => r.id === "b")).toMatchObject({ link: null, url: null });
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd web && npx vitest run lib/merge.test.ts`
Expected: FAIL — cannot find module `./merge`.

- [ ] **Step 3: Implement `web/lib/merge.ts`**

```typescript
import type { Listing } from "./types";

export interface PostLink { id: string; link: string | null; url: string | null; }

export function mergeLinks(offers: Listing[], posts: PostLink[]): Listing[] {
  const byId = new Map(posts.map((p) => [p.id, p]));
  return offers.map((o) => {
    const p = byId.get(o.id);
    return p ? { ...o, link: p.link, url: p.url } : o;
  });
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd web && npx vitest run lib/merge.test.ts` → Expected: PASS.

- [ ] **Step 5: Implement `web/lib/supabase.ts`** (network I/O; paginates past the 1000-row cap)

```typescript
import { createClient } from "@supabase/supabase-js";
import type { Listing } from "./types";
import { mergeLinks, type PostLink } from "./merge";

const PAGE = 1000;

function client() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) throw new Error("Missing NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY in web/.env.local");
  return createClient(url, key);
}

async function pageAll<T>(fetchPage: (from: number, to: number) => Promise<T[]>): Promise<T[]> {
  const rows: T[] = [];
  let from = 0;
  for (;;) {
    const page = await fetchPage(from, from + PAGE - 1);
    rows.push(...page);
    if (page.length < PAGE) return rows;
    from += PAGE;
  }
}

export async function fetchOffers(): Promise<Listing[]> {
  const sb = client();
  const offers = await pageAll<Listing>(async (from, to) => {
    const { data, error } = await sb.from("listings_parsed").select("*")
      .is("discard_reason", null).order("listed_at", { ascending: false }).range(from, to);
    if (error) throw error;
    return (data ?? []) as Listing[];
  });
  const posts = await pageAll<PostLink>(async (from, to) => {
    const { data, error } = await sb.from("fb_posts").select("id,link,url").range(from, to);
    if (error) throw error;
    return (data ?? []) as PostLink[];
  });
  return mergeLinks(offers, posts);
}
```

- [ ] **Step 6: Commit**

```bash
git add web/lib/merge.ts web/lib/merge.test.ts web/lib/supabase.ts && git commit -m "feat: paginated Supabase fetch + fb_posts link merge"
```

---

## Task 5: Hooks (`useListings`, `useFilters`)

**Files:** Create `web/hooks/useListings.ts`, `web/hooks/useFilters.ts`.

- [ ] **Step 1: Create `web/hooks/useListings.ts`**

```typescript
"use client";
import { useEffect, useState } from "react";
import { fetchOffers } from "@/lib/supabase";
import type { Listing } from "@/lib/types";

export function useListings() {
  const [listings, setListings] = useState<Listing[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let on = true;
    fetchOffers()
      .then((rows) => on && setListings(rows))
      .catch((e) => on && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => on && setLoading(false));
    return () => { on = false; };
  }, []);
  return { listings, loading, error };
}
```

- [ ] **Step 2: Create `web/hooks/useFilters.ts`** (URL-synced filter state)

```typescript
"use client";
import { useCallback, useEffect, useState } from "react";
import { defaultFilters, filtersToParams, paramsToFilters } from "@/lib/filters";
import type { FilterState } from "@/lib/types";

export function useFilters() {
  const [filters, setFilters] = useState<FilterState>(defaultFilters);
  // hydrate from URL on mount
  useEffect(() => { setFilters(paramsToFilters(new URLSearchParams(window.location.search))); }, []);
  // push to URL on change
  useEffect(() => {
    const qs = filtersToParams(filters).toString();
    const next = qs ? `?${qs}` : window.location.pathname;
    window.history.replaceState(null, "", next);
  }, [filters]);
  const reset = useCallback(() => setFilters(defaultFilters()), []);
  return { filters, setFilters, reset };
}
```

- [ ] **Step 3: Commit**

```bash
git add web/hooks && git commit -m "feat: useListings + URL-synced useFilters hooks"
```

---

## Task 6: Design tokens + UI primitives

**Files:** Modify `web/app/globals.css`; create `web/components/ui/Chip.tsx`, `Badge.tsx`, `EmptyState.tsx`, `Skeleton.tsx`.

- [ ] **Step 1: Replace `web/app/globals.css`** with tokens + base (island-editorial, light)

```css
@import "tailwindcss";

:root {
  --bg: oklch(98% 0.012 95);
  --surface: oklch(100% 0 0);
  --ink: oklch(25% 0.02 250);
  --muted: oklch(55% 0.02 250);
  --line: oklch(90% 0.01 250);
  --accent: oklch(62% 0.13 200);      /* island teal */
  --accent-ink: oklch(98% 0.01 200);
  --good: oklch(62% 0.15 150);        /* year-round / positive */
  --warn: oklch(70% 0.15 70);
  --radius: 14px;
  --shadow: 0 1px 2px oklch(25% 0.02 250 / 0.06), 0 8px 24px oklch(25% 0.02 250 / 0.06);
}

html, body { background: var(--bg); color: var(--ink); }
body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: inherit; }
*:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
```

- [ ] **Step 2: Create `web/components/ui/Badge.tsx`**

```tsx
export function Badge({ children, tone = "default" }: { children: React.ReactNode; tone?: "default" | "good" | "muted" }) {
  const bg = tone === "good" ? "var(--good)" : tone === "muted" ? "var(--line)" : "var(--accent)";
  const fg = tone === "muted" ? "var(--muted)" : "var(--accent-ink)";
  return <span style={{ background: bg, color: fg }} className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium">{children}</span>;
}
```

- [ ] **Step 3: Create `web/components/ui/Chip.tsx`**

```tsx
export function Chip({ label, onRemove }: { label: string; onRemove?: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      {label}
      {onRemove && <button onClick={onRemove} aria-label={`Remove ${label}`} className="opacity-60 hover:opacity-100">×</button>}
    </span>
  );
}
```

- [ ] **Step 4: Create `web/components/ui/Skeleton.tsx`**

```tsx
export function Skeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="h-40 animate-pulse rounded-2xl" style={{ background: "var(--line)" }} />
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Create `web/components/ui/EmptyState.tsx`**

```tsx
export function EmptyState({ onClear }: { onClear: () => void }) {
  return (
    <div className="rounded-2xl border p-10 text-center" style={{ borderColor: "var(--line)" }}>
      <p className="text-lg font-medium">No listings match these filters</p>
      <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>Try widening your price or area.</p>
      <button onClick={onClear} className="mt-4 rounded-full px-4 py-2 text-sm font-medium" style={{ background: "var(--accent)", color: "var(--accent-ink)" }}>Clear all filters</button>
    </div>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add web/app/globals.css web/components/ui && git commit -m "feat: design tokens + UI primitives (Badge/Chip/Skeleton/EmptyState)"
```

---

## Task 7: Filter controls + sidebar

**Files:** Create `web/components/filters/TriStateToggle.tsx`, `MultiSelect.tsx`, `PriceRange.tsx`, `NumberMax.tsx`, `AmenityToggles.tsx`, `FilterSidebar.tsx`.

- [ ] **Step 1: `TriStateToggle.tsx`**

```tsx
"use client";
import type { Tri } from "@/lib/types";
const OPTS: Tri[] = ["any", "yes", "no"];
export function TriStateToggle({ label, value, onChange }: { label: string; value: Tri; onChange: (v: Tri) => void }) {
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">{label}</div>
      <div className="inline-flex overflow-hidden rounded-lg border" style={{ borderColor: "var(--line)" }}>
        {OPTS.map((o) => (
          <button key={o} onClick={() => onChange(o)} className="px-3 py-1 text-xs capitalize"
            style={o === value ? { background: "var(--accent)", color: "var(--accent-ink)" } : { background: "var(--surface)" }}>{o}</button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: `MultiSelect.tsx`**

```tsx
"use client";
export function MultiSelect({ label, options, selected, onChange, render }:
  { label: string; options: string[]; selected: string[]; onChange: (v: string[]) => void; render?: (o: string) => string }) {
  const toggle = (o: string) => onChange(selected.includes(o) ? selected.filter((x) => x !== o) : [...selected, o]);
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">{label}</div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((o) => (
          <button key={o} onClick={() => toggle(o)} className="rounded-full border px-2.5 py-1 text-xs"
            style={selected.includes(o) ? { background: "var(--accent)", color: "var(--accent-ink)", borderColor: "var(--accent)" } : { borderColor: "var(--line)" }}>
            {render ? render(o) : o}
          </button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: `PriceRange.tsx`**

```tsx
"use client";
export function PriceRange({ min, max, onChange }: { min: number | null; max: number | null; onChange: (min: number | null, max: number | null) => void }) {
  const parse = (s: string) => (s === "" ? null : Number(s));
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">Price ฿/month</div>
      <div className="flex items-center gap-2">
        <input type="number" inputMode="numeric" placeholder="min" value={min ?? ""} onChange={(e) => onChange(parse(e.target.value), max)} className="w-24 rounded-lg border px-2 py-1 text-sm" style={{ borderColor: "var(--line)" }} />
        <span style={{ color: "var(--muted)" }}>–</span>
        <input type="number" inputMode="numeric" placeholder="max" value={max ?? ""} onChange={(e) => onChange(min, parse(e.target.value))} className="w-24 rounded-lg border px-2 py-1 text-sm" style={{ borderColor: "var(--line)" }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: `NumberMax.tsx`** (reused for beds-min, min-stay-max, deposit-max)

```tsx
"use client";
export function NumberMax({ label, value, onChange, placeholder = "any" }: { label: string; value: number | null; onChange: (v: number | null) => void; placeholder?: string }) {
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">{label}</div>
      <input type="number" inputMode="numeric" placeholder={placeholder} value={value ?? ""} onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} className="w-24 rounded-lg border px-2 py-1 text-sm" style={{ borderColor: "var(--line)" }} />
    </div>
  );
}
```

- [ ] **Step 5: `AmenityToggles.tsx`**

```tsx
"use client";
import { AMENITY_FIELDS } from "@/lib/constants";
export function AmenityToggles({ selected, onChange }: { selected: string[]; onChange: (v: string[]) => void }) {
  const toggle = (f: string) => onChange(selected.includes(f) ? selected.filter((x) => x !== f) : [...selected, f]);
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">Amenities</div>
      <div className="flex flex-wrap gap-1.5">
        {AMENITY_FIELDS.map(({ field, label }) => (
          <button key={field as string} onClick={() => toggle(field as string)} className="rounded-full border px-2.5 py-1 text-xs"
            style={selected.includes(field as string) ? { background: "var(--accent)", color: "var(--accent-ink)", borderColor: "var(--accent)" } : { borderColor: "var(--line)" }}>{label}</button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 6: `FilterSidebar.tsx`** (composes controls in the relevance order from the spec)

```tsx
"use client";
import type { FilterState } from "@/lib/types";
import { AREA_ENUM, PROPERTY_TYPES, SEASONS, CONFIDENCES, LANGUAGES } from "@/lib/constants";
import { areaName } from "@/lib/areas";
import { PriceRange } from "./PriceRange";
import { MultiSelect } from "./MultiSelect";
import { TriStateToggle } from "./TriStateToggle";
import { NumberMax } from "./NumberMax";
import { AmenityToggles } from "./AmenityToggles";

export function FilterSidebar({ filters, setFilters, onClear }:
  { filters: FilterState; setFilters: (f: FilterState) => void; onClear: () => void }) {
  const set = (patch: Partial<FilterState>) => setFilters({ ...filters, ...patch });
  const H = ({ children }: { children: React.ReactNode }) => <h3 className="mb-2 mt-4 text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>{children}</h3>;
  return (
    <aside className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Filters</h2>
        <button onClick={onClear} className="text-xs underline" style={{ color: "var(--muted)" }}>Clear all</button>
      </div>
      <H>Core</H>
      <PriceRange min={filters.priceMin} max={filters.priceMax} onChange={(priceMin, priceMax) => set({ priceMin, priceMax })} />
      <MultiSelect label="Area" options={AREA_ENUM} selected={filters.areas} onChange={(areas) => set({ areas })} render={areaName} />
      <MultiSelect label="Type" options={PROPERTY_TYPES} selected={filters.propertyTypes} onChange={(propertyTypes) => set({ propertyTypes })} />
      <NumberMax label="Min bedrooms" value={filters.bedroomsMin} onChange={(bedroomsMin) => set({ bedroomsMin })} />
      <NumberMax label="Min bathrooms" value={filters.bathroomsMin} onChange={(bathroomsMin) => set({ bathroomsMin })} />
      <H>Long-term (Goal 1)</H>
      <TriStateToggle label="Year-round" value={filters.yearRound} onChange={(yearRound) => set({ yearRound })} />
      <MultiSelect label="Season" options={SEASONS} selected={filters.seasons} onChange={(seasons) => set({ seasons })} />
      <NumberMax label="Max min-stay (months)" value={filters.minStayMax} onChange={(minStayMax) => set({ minStayMax })} />
      <H>Sublet (Goal 1b)</H>
      <TriStateToggle label="Subletting allowed" value={filters.subletting} onChange={(subletting) => set({ subletting })} />
      <H>Cost</H>
      <NumberMax label="Max deposit ฿" value={filters.depositMax} onChange={(depositMax) => set({ depositMax })} />
      <TriStateToggle label="Water included" value={filters.waterIncluded} onChange={(waterIncluded) => set({ waterIncluded })} />
      <TriStateToggle label="Internet included" value={filters.internetIncluded} onChange={(internetIncluded) => set({ internetIncluded })} />
      <H>Amenities</H>
      <AmenityToggles selected={filters.amenities} onChange={(amenities) => set({ amenities })} />
      <H>Quality / language</H>
      <MultiSelect label="Confidence" options={CONFIDENCES} selected={filters.confidences} onChange={(confidences) => set({ confidences })} />
      <MultiSelect label="Language" options={LANGUAGES} selected={filters.languages} onChange={(languages) => set({ languages })} />
    </aside>
  );
}
```

- [ ] **Step 7: Commit**

```bash
git add web/components/filters && git commit -m "feat: filter controls + sidebar (relevance-ordered)"
```

---

## Task 8: Listing card, details, grid, sort bar

**Files:** Create `web/components/listings/ListingCard.tsx`, `ListingDetails.tsx`, `ListingGrid.tsx`, `SortBar.tsx`, `ActiveChips.tsx`.

- [ ] **Step 1: `ListingDetails.tsx`** (renders all populated fields)

```tsx
import type { Listing } from "@/lib/types";
const HIDE = new Set(["id", "source_table", "parser_version", "raw_text"]);
export function ListingDetails({ listing }: { listing: Listing }) {
  const entries = Object.entries(listing).filter(([k, v]) => !HIDE.has(k) && v !== null && v !== "");
  return (
    <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 border-t pt-3 text-xs" style={{ borderColor: "var(--line)" }}>
      {entries.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-2">
          <dt style={{ color: "var(--muted)" }}>{k}</dt>
          <dd className="text-right font-medium">{String(v)}</dd>
        </div>
      ))}
      {listing.raw_text && <div className="col-span-2 mt-2"><dt style={{ color: "var(--muted)" }}>raw_text</dt><dd className="whitespace-pre-wrap">{listing.raw_text}</dd></div>}
    </dl>
  );
}
```

- [ ] **Step 2: `ListingCard.tsx`**

```tsx
"use client";
import { useState } from "react";
import type { Listing } from "@/lib/types";
import { areaName } from "@/lib/areas";
import { Badge } from "@/components/ui/Badge";
import { ListingDetails } from "./ListingDetails";

function postHref(l: Listing): string | null {
  if (l.url) return l.url;
  if (l.link) return l.link;
  if (l.id?.startsWith("http")) return l.id;
  return null;
}
const price = (l: Listing) => (l.price_thb != null ? `฿${l.price_thb.toLocaleString()}/mo` : "Ask");

export function ListingCard({ listing }: { listing: Listing }) {
  const [open, setOpen] = useState(false);
  const href = postHref(listing);
  const specs = [listing.property_type, listing.bedrooms != null ? `${listing.bedrooms}bd` : null, listing.bathrooms != null ? `${listing.bathrooms}ba` : null].filter(Boolean).join(" · ");
  return (
    <article className="rounded-2xl border p-4 transition-shadow hover:shadow-[var(--shadow)]" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      <div className="flex items-baseline justify-between">
        <span className="text-lg font-semibold">{price(listing)}</span>
        <span className="text-sm" style={{ color: "var(--muted)" }}>{areaName(listing.area_canonical)}</span>
      </div>
      <p className="mt-0.5 text-sm" style={{ color: "var(--muted)" }}>{specs || "—"}</p>
      <div className="mt-2 flex flex-wrap gap-1">
        {listing.year_round === true && <Badge tone="good">Year-round</Badge>}
        {listing.subletting_allowed === true && <Badge>Sublet OK</Badge>}
        {listing.season && listing.season !== "unknown" && <Badge tone="muted">{listing.season}</Badge>}
        {listing.parse_confidence && <Badge tone="muted">{listing.parse_confidence}</Badge>}
      </div>
      <div className="mt-3 flex items-center gap-3 text-sm">
        <button onClick={() => setOpen((o) => !o)} className="underline" style={{ color: "var(--accent)" }}>{open ? "Hide details" : "Details"}</button>
        {href && <a href={href} target="_blank" rel="noreferrer" className="underline" style={{ color: "var(--accent)" }}>View on FB ↗</a>}
      </div>
      {open && <ListingDetails listing={listing} />}
    </article>
  );
}
```

- [ ] **Step 3: `ListingGrid.tsx`**

```tsx
import type { Listing } from "@/lib/types";
import { ListingCard } from "./ListingCard";
export function ListingGrid({ listings }: { listings: Listing[] }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {listings.map((l) => <ListingCard key={l.id} listing={l} />)}
    </div>
  );
}
```

- [ ] **Step 4: `SortBar.tsx`**

```tsx
"use client";
import type { SortKey } from "@/lib/types";
const OPTS: { k: SortKey; label: string }[] = [
  { k: "newest", label: "Newest" }, { k: "price_asc", label: "Price ↑" },
  { k: "price_desc", label: "Price ↓" }, { k: "confidence", label: "Confidence" },
];
export function SortBar({ count, sort, onSort }: { count: number; sort: SortKey; onSort: (s: SortKey) => void }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm" style={{ color: "var(--muted)" }}>{count.toLocaleString()} listings</span>
      <label className="text-sm">Sort{" "}
        <select value={sort} onChange={(e) => onSort(e.target.value as SortKey)} className="rounded-lg border px-2 py-1" style={{ borderColor: "var(--line)" }}>
          {OPTS.map((o) => <option key={o.k} value={o.k}>{o.label}</option>)}
        </select>
      </label>
    </div>
  );
}
```

- [ ] **Step 5: `ActiveChips.tsx`** (shows applied filters as removable chips)

```tsx
"use client";
import type { FilterState } from "@/lib/types";
import { defaultFilters } from "@/lib/filters";
import { Chip } from "@/components/ui/Chip";
export function ActiveChips({ filters, setFilters }: { filters: FilterState; setFilters: (f: FilterState) => void }) {
  const d = defaultFilters();
  const chips: { label: string; clear: Partial<FilterState> }[] = [];
  if (filters.priceMin != null) chips.push({ label: `≥ ฿${filters.priceMin}`, clear: { priceMin: null } });
  if (filters.priceMax != null) chips.push({ label: `≤ ฿${filters.priceMax}`, clear: { priceMax: null } });
  filters.areas.forEach((a) => chips.push({ label: a, clear: { areas: filters.areas.filter((x) => x !== a) } }));
  if (filters.yearRound !== "any") chips.push({ label: `year-round: ${filters.yearRound}`, clear: { yearRound: d.yearRound } });
  if (filters.subletting !== "any") chips.push({ label: `sublet: ${filters.subletting}`, clear: { subletting: d.subletting } });
  filters.amenities.forEach((a) => chips.push({ label: a, clear: { amenities: filters.amenities.filter((x) => x !== a) } }));
  if (!chips.length) return null;
  return <div className="mb-3 flex flex-wrap gap-1.5">{chips.map((c, i) => <Chip key={i} label={c.label} onRemove={() => setFilters({ ...filters, ...c.clear })} />)}</div>;
}
```

- [ ] **Step 6: Commit**

```bash
git add web/components/listings && git commit -m "feat: listing card/details/grid + sort bar + active chips"
```

---

## Task 9: Dashboard wiring + page + layout

**Files:** Create `web/components/listings/Dashboard.tsx`; replace `web/app/page.tsx`, `web/app/layout.tsx`.

- [ ] **Step 1: `Dashboard.tsx`** (client island: data + filters + layout)

```tsx
"use client";
import { useMemo, useState } from "react";
import { useListings } from "@/hooks/useListings";
import { useFilters } from "@/hooks/useFilters";
import { applyFilters, sortListings } from "@/lib/filters";
import { FilterSidebar } from "@/components/filters/FilterSidebar";
import { ListingGrid } from "./ListingGrid";
import { SortBar } from "./SortBar";
import { ActiveChips } from "./ActiveChips";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

export function Dashboard() {
  const { listings, loading, error } = useListings();
  const { filters, setFilters, reset } = useFilters();
  const [drawer, setDrawer] = useState(false);
  const results = useMemo(() => sortListings(applyFilters(listings, filters), filters.sort), [listings, filters]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <header className="sticky top-0 z-10 -mx-4 mb-4 flex items-center justify-between border-b px-4 py-3 backdrop-blur" style={{ borderColor: "var(--line)", background: "color-mix(in oklch, var(--bg) 80%, transparent)" }}>
        <h1 className="text-lg font-bold tracking-tight">KP House Finder</h1>
        <button className="rounded-full border px-3 py-1 text-sm lg:hidden" style={{ borderColor: "var(--line)" }} onClick={() => setDrawer(true)}>Filters</button>
      </header>

      {error && <div className="rounded-xl border p-4 text-sm" style={{ borderColor: "var(--warn)" }}>Couldn’t load listings: {error}</div>}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr]">
        <div className="hidden lg:block"><div className="sticky top-20"><FilterSidebar filters={filters} setFilters={setFilters} onClear={reset} /></div></div>

        <main>
          <SortBar count={results.length} sort={filters.sort} onSort={(s) => setFilters({ ...filters, sort: s })} />
          <div className="mt-3"><ActiveChips filters={filters} setFilters={setFilters} /></div>
          {loading ? <Skeleton /> : results.length === 0 ? <EmptyState onClear={reset} /> : <ListingGrid listings={results} />}
        </main>
      </div>

      {drawer && (
        <div className="fixed inset-0 z-20 lg:hidden">
          <div className="absolute inset-0 bg-black/30" onClick={() => setDrawer(false)} />
          <div className="absolute left-0 top-0 h-full w-80 overflow-y-auto p-4" style={{ background: "var(--surface)" }}>
            <button className="mb-2 text-sm underline" onClick={() => setDrawer(false)}>Close</button>
            <FilterSidebar filters={filters} setFilters={setFilters} onClear={reset} />
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Replace `web/app/page.tsx`**

```tsx
import { Dashboard } from "@/components/listings/Dashboard";
export default function Page() {
  return <Dashboard />;
}
```

- [ ] **Step 3: Replace `web/app/layout.tsx`**

```tsx
import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "KP House Finder", description: "Personal Koh Phangan rental dashboard" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
```

- [ ] **Step 4: Typecheck + lint + build**

Run: `cd web && npx tsc --noEmit && npm run lint && npm run build`
Expected: no type errors, lint clean, production build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/components/listings/Dashboard.tsx web/app/page.tsx web/app/layout.tsx && git commit -m "feat: dashboard wiring, page, layout (left rail + responsive grid + mobile drawer)"
```

---

## Task 10: End-to-end verification

**Files:** none (operational). Requires `web/.env.local` with the user's Supabase values.

- [ ] **Step 1: Create env**

`cp web/.env.example web/.env.local` and fill `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY` (same values as the root `.env`, **user-provided**).

- [ ] **Step 2: Run the full unit suite**

Run: `cd web && npm run test`
Expected: PASS (filters + merge tests green).

- [ ] **Step 3: Dev run + manual checks**

Run: `cd web && npm run dev` → http://localhost:3000
Verify:
- Card grid loads; the count reads ~**1805** (proves pagination didn’t truncate at 1000).
- Set price ≤ 12000 + Area = Sri Thanu + Year-round = Yes → grid + count update instantly; the **URL** gains `?priceMax=12000&areas=srithanu&yearRound=yes`; reload preserves the filters.
- A card’s **View on FB ↗** opens the original post; **Details** expands all populated fields.
- Resize to mobile width → the Filters drawer works; grid becomes 1-column.

- [ ] **Step 4: Final commit (if any tweaks)**

```bash
git add -A && git commit -m "test: verify dashboard end to end" || echo "nothing to commit"
```

---

## Notes for the executor
- **TDD applies to `lib/filters.ts` and `lib/merge.ts`** (pure logic) — write test, watch fail, implement, watch pass, commit. UI components are verified by typecheck/build + the manual checks in Task 10 (per web testing rules, visual/runtime checks carry more signal than brittle markup assertions).
- **Don’t** modify the Python pipeline, `listings_parsed.sql`, or the Chrome extension — the dashboard is read-only over existing data.
- **Secrets:** only `web/.env.example` is committed; `web/.env.local` is gitignored.
- **The 1000-row cap** is the key correctness risk — `fetchOffers` MUST keep its `range()` pagination loop.
- **Deferred from spec (deliberate V1 scope cut):** the optional "reveal flagged listings" toggle is not built — `fetchOffers` fetches offers only (`discard_reason IS NULL`). Adding it later means a second fetch + a header toggle; the schema already supports it.
```
