# Design — KP Rentals: `fb_posts` → Structured Listings (Analysis Phase, slice 1)

> Superpowers `brainstorming` design output (approved). The task-by-task build plan
> derived from this is `docs/superpowers/plans/2026-06-24-fb-posts-structured-extraction.md`.

## Context

The KP Rentals pipeline scrapes Facebook group posts into Supabase (`fb_posts`); the
value lives in the unstructured `text` column. The **analysis layer is not built yet**
(`PROJECT_HANDOFF.md` §6). A prior session built `extract.py` — a working,
source-agnostic LLM extractor (Anthropic SDK, `claude-haiku-4-5`, JSON-only prompt,
incremental) that today reads/writes CSV. This slice **re-points that engine at
`fb_posts`** and writes a richer structured schema into a new `listings_parsed` table.

**Scope this slice: `fb_posts` only.** WhatsApp (`wa_messages`) and cross-source dedup
come later.

### Purpose (drives the schema)
The user needs, within ~2–3 weeks, a place to rent for a **full 12 months starting
ASAP, including high season** (Goal 1, primary), and secondarily to spot places with
strong **Airbnb/Booking sublet potential in high season** (Goal 1b). This is why the
schema captures rental-vs-sale, year-round availability, seasonal pricing, and
sublet permission explicitly. "Best for me" vs "best for subletting" are two ranking
**lenses over the same structured data** (a later analysis/UI slice), not extra
extracted fields; urgency is a sort on `listed_at` + `available_from`.

### Decisions (confirmed with user)
- **Output:** new Supabase table `listings_parsed`, keyed by original `fb_posts.id`.
- **DB access:** `supabase-py` over REST + anon key (matches the extension's RLS pattern).
- **One combined LLM call** returns relevance + intent + language + all fields as one
  JSON object (not four sequential passes).
- **Soft-flag, never discard.** Every processed row is written; exclusion is recorded
  via `discard_reason` (`not_a_listing` / `wanted` / `for_sale` / `not_koh_phangan` /
  `not_long_term`), not row deletion (preserves incrementality / avoids re-charging the
  LLM to re-drop the same rows). `for_sale` keeps purchase listings out of your rentals.
- **Long-term:** flag, don't drop — capture `price_period`/`season`/`min_stay_months`;
  clearly short-term (nightly/weekly) gets `discard_reason='not_long_term'`.
- **Property scope:** all dwellings (house/villa/bungalow/apartment/studio/room).
- **Language:** keep & extract any language; `post_language` enum gains `other`. Never
  discard on language grounds.
- **Areas:** canonical KP enum confirmed (18 areas + `other` + `unknown`, listed below).

## Architecture

Three independently testable pieces:

1. **`extract.py` refactor** — expose a pure `extract_fields(text) -> dict` returning the
   full field dict **including `is_offer`, `discard_reason`, `post_language`,
   `parse_confidence`** (no silent drop; callers decide). CSV `main()` keeps current
   behavior. Reuse `call_claude`, JSON-parse logic; **replace the prompt + field list**
   with the new schema below; bump `max_tokens` to ~2048.
2. **`db.py`** — supabase-py client from `SUPABASE_URL`/`SUPABASE_ANON_KEY`; helpers
   `fetch_source_rows()`, `existing_parsed_ids()`, `upsert_parsed()`, `inspect()`.
   **All full-table reads paginate via `.range()`** (PostgREST caps at 1000 rows by
   default — otherwise already-parsed rows get re-sent to the LLM and `--source all`
   silently truncates).
3. **`analyze.py`** — CLI: `--inspect` (counts per source + samples + live-schema check
   vs handoff), default run extracts un-parsed `fb_posts` rows and upserts. Flags:
   `--limit N`, `--inspect`.

## `listings_parsed` schema

**Control / metadata**
| column | type | notes |
| --- | --- | --- |
| `id` | text PK | = `fb_posts.id` (idempotent upsert, `on_conflict=id`) |
| `source_table` | text | `'fb_posts'` |
| `source` | text | FB group, from `fb_posts.source` |
| `listed_at` | timestamptz | = `fb_posts.created_at` (reliable time axis per handoff §3) |
| `raw_text` | text | original `fb_posts.text` |
| `discard_reason` | text enum | `not_a_listing` / `wanted` / `for_sale` / `not_koh_phangan` / `not_long_term` / NULL. **Offers view = `WHERE discard_reason IS NULL`** |
| `post_language` | text enum | `en` / `th` / `mixed` / `other` |
| `parse_confidence` | text enum | `low` / `medium` / `high` |
| `parser_version` | text | schema/prompt version; lets re-runs re-extract only stale rows |
| `multi_listing` | bool | true when one post bundles several properties (then confidence=low) |
| `parsed_at` | timestamptz | default now() |

**Tier 1 — core filters**
`is_offer` enum(offer/wanted/ambiguous) · `price_thb` int · `price_low_thb` int ·
`price_high_thb` int · `price_period` enum(month/week/night/unknown) ·
`season` enum(low/high/full_year/unknown) · `bedrooms` int · `bathrooms` int ·
`property_type` enum(house/villa/bungalow/apartment/studio/room/unknown) ·
`area_raw` text · `area_canonical` enum (see list).
*Price rule:* `price_thb` = monthly figure (low end if a range); `price_low/high_thb`
hold seasonal ranges; `price_period` disambiguates the unit.

**Tier 2 — long-term suitability**
`min_stay_months` int · `available_from` text(date or `now`) · `available_until` date (usually NULL) ·
`year_round` bool (available the full year incl. high season; NULL=unknown) ·
`subletting_allowed` bool (Airbnb/Booking permitted; NULL=unknown).

**Tier 3 — total cost**
`deposit_thb` int · `electricity_rate_thb_per_unit` numeric ·
`water_included` bool · `internet_included` bool.
*Convention:* nullable bool where **NULL = unknown** (unknown ≠ no), applied to all bools.

**Tier 4 — amenities** (nullable bool, NULL=unknown)
`has_aircon` · `has_wifi` · `furnished` · `has_kitchen` · `has_pool` · `has_parking` ·
`pet_friendly` · `sea_view` · `has_workspace` · `has_terrace` · `near_road` ·
`near_construction` enum(construction/quiet/unknown) · `furnishings_list` text.

**Tier 5 — contact / misc**
`contact_raw` text · `contact_phone` text(normalized) · `size_sqm` int.

## Canonical KP area enum (CONFIRMED)
`thong_sala`, `ban_tai`, `ban_kai`, `haad_rin`, `srithanu`, `chaloklum`, `mae_haad`,
`hin_kong`, `woktum`, `haad_yao`, `haad_salad`, `haad_son`, `thong_nai_pan`,
`bottle_beach`, `than_sadet`, `haad_yuan_tien`, `madeua_wan`, `plai_laem`,
`other`, `unknown`.

## Incrementality
Keyed by `fb_posts.id`. `analyze.py` reads `existing_parsed_ids()` (paginated), pulls
only source `id`s not yet present, and upserts. Non-housing / wanted / for-sale /
short-term rows are written *with their `discard_reason`*, so they count as processed
and never re-hit the LLM. `parser_version` enables a future targeted re-extract when the
schema evolves.

## Out of scope (next slices)
WhatsApp (`wa_messages`) extraction · cross-source unified listings (FB↔WA dedup) ·
the filtered offers / sublet-ranking dashboard UI.
