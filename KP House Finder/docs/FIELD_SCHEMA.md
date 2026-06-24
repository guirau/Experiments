# `listings_parsed` field schema reference

Canonical reference for the structured fields extracted from `fb_posts.text` (and later
`wa_messages`) by `extract.py`. Keep this in sync with `extract.py` (`MODEL_FIELDS`,
`ENUMS`, `AREA_ENUM`) and `listings_parsed.sql`.

**Purpose.** Turn unstructured Koh Phangan rental posts into a queryable table so the user
can find a place to rent for a **full 12 months including high season** (Goal 1) and spot
places with strong **Airbnb/Booking sublet potential** (Goal 1b) — without scrolling
Facebook daily. "Best for me" vs "best for subletting" are query-time lenses over this one
table, not extra fields.

**`PARSER_VERSION = "fb-1.0"`** — stamped on every parsed row. Bump when the prompt or
schema changes so re-runs can re-extract only stale rows.

## Conventions

- **NULL = unknown, not "no".** Every nullable boolean is `true` only if clearly present,
  `false` only if clearly stated absent, `NULL` if unmentioned. Unknown ≠ false.
- **Soft-flag, never discard.** Non-offers are written *with* a `discard_reason`, never
  deleted — this keeps the pipeline idempotent and avoids re-charging the LLM. The offers
  view is `WHERE discard_reason IS NULL`.
- **Price rule.** `price_thb` = the MONTHLY figure (the lower value if a range). Seasonal
  ranges go in `price_low_thb` / `price_high_thb`. `price_period` disambiguates the unit.
- **Enums coerce.** Invalid enum values from the model fall back to the listed default
  (e.g. an unknown area → `unknown`); see `ENUMS` in `extract.py`.

## Control / metadata columns

| column | type | meaning |
| --- | --- | --- |
| `id` | text PK | = `fb_posts.id`; idempotent upsert key (`on_conflict=id`) |
| `source_table` | text | `'fb_posts'` (later `'wa_messages'`) |
| `source` | text | originating FB group, from `fb_posts.source` |
| `listed_at` | timestamptz | = `fb_posts.created_at` (reliable time axis) |
| `raw_text` | text | original `fb_posts.text` |
| `parser_version` | text | prompt/schema version (`fb-1.0`) |
| `parsed_at` | timestamptz | DB default `now()` |

## Classification

| field | type | values / meaning |
| --- | --- | --- |
| `discard_reason` | enum/NULL | `not_a_listing` / `wanted` / `for_sale` / `not_koh_phangan` / `not_long_term` / NULL (kept) |
| `is_offer` | enum | `offer` / `wanted` / `ambiguous` (fallback `ambiguous`) |
| `post_language` | enum | `en` / `th` / `mixed` / `other` (fallback `other`) |
| `parse_confidence` | enum | `low` / `medium` / `high` (fallback `low`) |
| `multi_listing` | bool | true if one post bundles several properties (then confidence=low, first property extracted) |

**`discard_reason` rules** (set ONE when the post should be excluded, else NULL):
- `not_a_listing` — not about a place to live (vehicle, job, service, item for sale, pet,
  event, chat). **Not** triggered merely by truncation/short text: a truncated "House for
  Rent…" is kept with low confidence.
- `wanted` — poster is LOOKING FOR a place, not offering one.
- `for_sale` — property being SOLD, not rented (purchase price, land for sale,
  leasehold/freehold sale). Keeps purchase listings out of the rentals.
- `not_koh_phangan` — clearly another location (Koh Samui, Koh Tao, mainland, …).
- `not_long_term` — clearly ONLY short-term/holiday (nightly/weekly) with no long-term option.

## Tier 1 — core filters

| field | type | meaning |
| --- | --- | --- |
| `price_thb` | int | monthly THB (lower end of a range) |
| `price_low_thb` / `price_high_thb` | int | seasonal low/high range when quoted |
| `price_period` | enum | `month` / `week` / `night` / `unknown` |
| `season` | enum | `low` / `high` / `full_year` / `unknown` |
| `bedrooms` / `bathrooms` | int | counts |
| `property_type` | enum | `house` / `villa` / `bungalow` / `apartment` / `studio` / `room` / `unknown` |
| `area_raw` | text | area/village as written |
| `area_canonical` | enum | mapped KP area (see list) |

## Tier 2 — long-term suitability (Goals 1 & 1b)

| field | type | meaning |
| --- | --- | --- |
| `min_stay_months` | int | minimum rental term in months |
| `available_from` | text | date or `"now"` |
| `available_until` | date | usually NULL |
| `year_round` | bool | true if available the FULL year incl. high season; false if low-season-only / owner reclaims / price hike in high season; NULL unknown |
| `subletting_allowed` | bool | true if Airbnb/Booking/subletting permitted; false if forbidden; NULL unknown |

## Tier 3 — total cost

| field | type | meaning |
| --- | --- | --- |
| `deposit_thb` | int | deposit |
| `electricity_rate_thb_per_unit` | numeric | electricity unit rate |
| `water_included` / `internet_included` | bool | NULL = unknown |

## Tier 4 — amenities (bool, NULL = unknown)

`has_aircon`, `has_wifi`, `furnished`, `has_kitchen`, `has_pool`, `has_parking`,
`pet_friendly`, `sea_view`, `has_workspace`, `has_terrace`, `near_road`;
`near_construction` enum (`construction` / `quiet` / `unknown`);
`furnishings_list` text (comma-joined extras).

## Tier 5 — contact / misc

`contact_raw` text (phone/Line/WhatsApp as written), `contact_phone` text (normalized),
`size_sqm` int.

## Canonical Koh Phangan areas (`area_canonical`)

| slug | area |
| --- | --- |
| `thong_sala` | Thong Sala |
| `ban_tai` | Ban Tai |
| `ban_kai` | Ban Kai / Baan Kai |
| `haad_rin` | Haad Rin |
| `srithanu` | Sri Thanu |
| `chaloklum` | Chaloklum |
| `mae_haad` | Mae Haad |
| `hin_kong` | Hin Kong |
| `woktum` | Woktum |
| `haad_yao` | Haad Yao |
| `haad_salad` | Haad Salad |
| `haad_son` | Haad Son |
| `thong_nai_pan` | Thong Nai Pan |
| `bottle_beach` | Bottle Beach / Haad Khuat |
| `than_sadet` | Than Sadet |
| `haad_yuan_tien` | Haad Yuan / Haad Tien |
| `madeua_wan` | Madeua Wan |
| `plai_laem` | Plai Laem |
| `other` | a real KP area not in this list |
| `unknown` | no area stated |
