# fb_posts → Structured Listings Implementation Plan

> **AMENDED 2026-06-24 during execution.** Two changes from the task list below:
> (1) a **v0 read-only Jupyter notebook** (`notebooks/v0_fb_posts_transform.ipynb`,
> 10 rows, one batched call) was added as a visual checkpoint before any Supabase writes;
> (2) the engine now **batches 10 posts per LLM call** with a per-row fallback, so
> `extract.py` exposes `extract_batch()` / `call_claude_batch()` and `extract_fields()`
> is a thin single-row wrapper. Tasks below remain accurate in intent; the prompt was
> also tuned so truncated-but-housing posts are kept. Canonical field reference now lives
> in `docs/FIELD_SCHEMA.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract structured rental fields from Supabase `fb_posts.text` via one LLM call and upsert them into a new `listings_parsed` table, incrementally and idempotently.

**Architecture:** Reuse the existing `extract.py` LLM engine — swap its prompt/field schema for the richer tiered schema and expose a pure `extract_fields(text) -> dict`. A new `db.py` wraps `supabase-py` (paginated reads, upsert, inspect). A new `analyze.py` CLI orchestrates: read un-parsed `fb_posts` ids, extract, map, upsert. Rows are never deleted — non-listings/wanted/short-term get a `discard_reason`; the offers view filters on it.

**Tech Stack:** Python 3.11, `anthropic` (`claude-haiku-4-5-20251001`), `supabase-py`, `python-dotenv`, `pytest`.

**Spec:** `/Users/guirau/.claude/plans/there-is-a-file-mighty-narwhal.md` (approved design).

---

## Shared definitions (used across tasks)

`PARSER_VERSION = "fb-1.0"` — stamped on every parsed row; bump when the prompt/schema changes.

**Model-returned keys** (the LLM returns exactly these as one JSON object):

```python
CLASSIFICATION = ["discard_reason", "is_offer", "post_language", "parse_confidence", "multi_listing"]
TIER1 = ["price_thb", "price_low_thb", "price_high_thb", "price_period", "season",
         "bedrooms", "bathrooms", "property_type", "area_raw", "area_canonical"]
TIER2 = ["min_stay_months", "available_from", "available_until", "year_round", "subletting_allowed"]
TIER3 = ["deposit_thb", "electricity_rate_thb_per_unit", "water_included", "internet_included"]
TIER4 = ["has_aircon", "has_wifi", "furnished", "has_kitchen", "has_pool", "has_parking",
         "pet_friendly", "sea_view", "has_workspace", "has_terrace", "near_road",
         "near_construction", "furnishings_list"]
TIER5 = ["contact_raw", "contact_phone", "size_sqm"]
MODEL_FIELDS = CLASSIFICATION + TIER1 + TIER2 + TIER3 + TIER4 + TIER5

AREA_ENUM = ["thong_sala", "ban_tai", "ban_kai", "haad_rin", "srithanu", "chaloklum",
             "mae_haad", "hin_kong", "woktum", "haad_yao", "haad_salad", "haad_son",
             "thong_nai_pan", "bottle_beach", "than_sadet", "haad_yuan_tien",
             "madeua_wan", "plai_laem", "other", "unknown"]
```

**DB row** = control metadata + `MODEL_FIELDS`:
control = `id, source_table, source, listed_at, raw_text, parser_version` (+ `parsed_at` defaulted in SQL).

---

## Task 1: Project setup (deps, env, .env.example)

**Files:**
- Modify: `pyproject.toml:9-12`
- Modify: `.env`
- Create: `.env.example`

- [ ] **Step 1: Add runtime + dev dependencies**

Run:
```bash
cd "/Users/guirau/GitHub/guirau/Experiments/KP House Finder"
poetry add "supabase@^2.0"
poetry add --group dev pytest
```
Expected: `pyproject.toml` gains `supabase` under `[project].dependencies` (or `[tool.poetry...]`) and `pytest` as a dev dependency; `poetry.lock` updates.

- [ ] **Step 2: Add Supabase keys to `.env`** (do NOT commit real values)

Append to `.env`:
```
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_ANON_KEY=<anon-key-from-supabase-dashboard>
```
(Use the same URL + anon key the Chrome extension popup uses — Supabase Dashboard → Project Settings → API.)

- [ ] **Step 3: Create `.env.example`**

```
# Anthropic API key for the extraction LLM
ANTHROPIC_API_KEY=sk-ant-...

# Supabase project (Dashboard -> Project Settings -> API)
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key
```

- [ ] **Step 4: Verify `.env` is git-ignored**

Run: `git check-ignore .env`
Expected: prints `.env`. If it prints nothing, add `.env` to `.gitignore` before continuing.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock .env.example .gitignore
git commit -m "chore: add supabase + pytest deps and .env.example"
```

---

## Task 2: Durable field-schema reference doc

**Files:**
- Create: `docs/FIELD_SCHEMA.md`

- [ ] **Step 1: Write `docs/FIELD_SCHEMA.md`**

Write a document containing: the project purpose (1 paragraph), `PARSER_VERSION`, every column in `listings_parsed` grouped as Control / Tier 1–5 with type + meaning, the `discard_reason` enum and the rule for each value, the `NULL = unknown` boolean convention, the price rule (`price_thb` = monthly, low end of a range; `price_low/high_thb` for seasonal ranges), and the full confirmed `AREA_ENUM` with a human-readable name beside each slug. This is the canonical reference — copy the values verbatim from the Shared definitions above and from the approved spec's schema section.

- [ ] **Step 2: Commit**

```bash
git add docs/FIELD_SCHEMA.md
git commit -m "docs: add listings_parsed field schema reference"
```

---

## Task 3: `listings_parsed.sql`

**Files:**
- Create: `listings_parsed.sql`

- [ ] **Step 1: Write the table + RLS + policies** (mirrors `chrome_extension/KP-Rentals-Exporter/supabase_schema.sql:36-45`)

```sql
-- listings_parsed: structured fields extracted from fb_posts (and later wa_messages).
-- Run in Supabase SQL Editor. STEP 1 creates the table; STEP 2 enables RLS + anon policies.

-- STEP 1
create table if not exists listings_parsed (
  -- control / metadata
  id                            text primary key,   -- = fb_posts.id
  source_table                  text not null,      -- 'fb_posts'
  source                        text,               -- FB group
  listed_at                     timestamptz,        -- = fb_posts.created_at
  raw_text                      text,
  parser_version                text,
  parsed_at                     timestamptz not null default now(),
  discard_reason                text,               -- not_a_listing|wanted|for_sale|not_koh_phangan|not_long_term|null
  post_language                 text,               -- en|th|mixed|other
  parse_confidence              text,               -- low|medium|high
  multi_listing                 boolean,
  -- tier 1
  is_offer                      text,               -- offer|wanted|ambiguous
  price_thb                     integer,
  price_low_thb                 integer,
  price_high_thb                integer,
  price_period                  text,               -- month|week|night|unknown
  season                        text,               -- low|high|full_year|unknown
  bedrooms                      integer,
  bathrooms                     integer,
  property_type                 text,               -- house|villa|bungalow|apartment|studio|room|unknown
  area_raw                      text,
  area_canonical                text,               -- AREA_ENUM
  -- tier 2 (long-term suitability)
  min_stay_months               integer,
  available_from                text,               -- date or 'now'
  available_until               date,
  year_round                    boolean,            -- available full year incl. high season; null = unknown
  subletting_allowed            boolean,            -- Airbnb/Booking permitted; null = unknown
  -- tier 3
  deposit_thb                   integer,
  electricity_rate_thb_per_unit numeric,
  water_included                boolean,            -- null = unknown
  internet_included             boolean,
  -- tier 4 (null = unknown)
  has_aircon                    boolean,
  has_wifi                      boolean,
  furnished                     boolean,
  has_kitchen                   boolean,
  has_pool                      boolean,
  has_parking                   boolean,
  pet_friendly                  boolean,
  sea_view                      boolean,
  has_workspace                 boolean,
  has_terrace                   boolean,
  near_road                     boolean,
  near_construction             text,               -- construction|quiet|unknown
  furnishings_list              text,
  -- tier 5
  contact_raw                   text,
  contact_phone                 text,
  size_sqm                      integer
);
create index if not exists listings_parsed_source_idx        on listings_parsed (source);
create index if not exists listings_parsed_discard_idx       on listings_parsed (discard_reason);
create index if not exists listings_parsed_parser_ver_idx    on listings_parsed (parser_version);

-- STEP 2
alter table listings_parsed enable row level security;
create policy "anon insert parsed" on listings_parsed for insert to anon with check (true);
create policy "anon select parsed" on listings_parsed for select to anon using (true);
create policy "anon update parsed" on listings_parsed for update to anon using (true) with check (true);
```

- [ ] **Step 2: Apply it in Supabase** (manual)

Paste STEP 1 then STEP 2 into the Supabase SQL Editor and run. Confirm the table appears under Table Editor with 0 rows.

- [ ] **Step 3: Commit**

```bash
git add listings_parsed.sql
git commit -m "feat: add listings_parsed schema + anon RLS policies"
```

---

## Task 4: Refactor `extract.py` — new schema + pure `extract_fields()`

**Files:**
- Modify: `extract.py` (replace lines 26-126 region: `MODEL`, field lists, `SYSTEM_PROMPT`, add `extract_fields`)
- Test: `tests/test_extract_fields.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_extract_fields.py
import extract

def test_short_text_is_marked_not_a_listing():
    fields = extract.extract_fields("hi")
    assert fields["discard_reason"] == "not_a_listing"
    assert fields["parse_confidence"] == "low"
    # every model key is present, defaulted to None
    for key in extract.MODEL_FIELDS:
        assert key in fields

def test_normalizes_model_output(monkeypatch):
    canned = {
        "discard_reason": None, "is_offer": "OFFER", "post_language": "EN",
        "parse_confidence": "high", "multi_listing": False,
        "price_thb": 15000, "price_period": "month", "bedrooms": 2,
        "property_type": "villa", "area_raw": "Sri Thanu", "area_canonical": "srithanu",
        "has_pool": True, "has_wifi": None,
        "furnishings_list": ["fridge", "TV", "terrace"],
    }
    monkeypatch.setattr(extract, "call_claude", lambda text: canned)
    fields = extract.extract_fields("a long enough listing text about a villa")
    assert fields["is_offer"] == "offer"           # lowercased enum
    assert fields["post_language"] == "en"
    assert fields["price_thb"] == 15000            # int preserved
    assert fields["has_pool"] is True              # bool preserved
    assert fields["has_wifi"] is None              # unknown preserved
    assert fields["furnishings_list"] == "fridge, TV, terrace"  # list joined
    assert fields["area_canonical"] == "srithanu"

def test_invalid_enum_falls_back(monkeypatch):
    monkeypatch.setattr(extract, "call_claude",
                        lambda text: {"area_canonical": "atlantis", "property_type": "spaceship"})
    fields = extract.extract_fields("a long enough listing text here for parsing")
    assert fields["area_canonical"] == "unknown"
    assert fields["property_type"] == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/guirau/GitHub/guirau/Experiments/KP House Finder" && poetry run pytest tests/test_extract_fields.py -v`
Expected: FAIL — `AttributeError: module 'extract' has no attribute 'extract_fields'` / `MODEL_FIELDS`.

- [ ] **Step 3: Replace the schema block in `extract.py`**

Replace the current `EXTRACT_FIELDS` / `ORIGINAL_COLS` / `CLASSIFY_FIELD` / `MODEL_KEYS` block (lines 28-63) with:

```python
PARSER_VERSION = "fb-1.0"

CLASSIFICATION = ["discard_reason", "is_offer", "post_language", "parse_confidence", "multi_listing"]
TIER1 = ["price_thb", "price_low_thb", "price_high_thb", "price_period", "season",
         "bedrooms", "bathrooms", "property_type", "area_raw", "area_canonical"]
TIER2 = ["min_stay_months", "available_from", "available_until", "year_round", "subletting_allowed"]
TIER3 = ["deposit_thb", "electricity_rate_thb_per_unit", "water_included", "internet_included"]
TIER4 = ["has_aircon", "has_wifi", "furnished", "has_kitchen", "has_pool", "has_parking",
         "pet_friendly", "sea_view", "has_workspace", "has_terrace", "near_road",
         "near_construction", "furnishings_list"]
TIER5 = ["contact_raw", "contact_phone", "size_sqm"]
MODEL_FIELDS = CLASSIFICATION + TIER1 + TIER2 + TIER3 + TIER4 + TIER5

# enum -> allowed values; anything else is coerced to the listed fallback
AREA_ENUM = ["thong_sala", "ban_tai", "ban_kai", "haad_rin", "srithanu", "chaloklum",
             "mae_haad", "hin_kong", "woktum", "haad_yao", "haad_salad", "haad_son",
             "thong_nai_pan", "bottle_beach", "than_sadet", "haad_yuan_tien",
             "madeua_wan", "plai_laem", "other", "unknown"]
ENUMS = {
    "discard_reason": (["not_a_listing", "wanted", "for_sale", "not_koh_phangan", "not_long_term"], None),
    "is_offer": (["offer", "wanted", "ambiguous"], "ambiguous"),
    "post_language": (["en", "th", "mixed", "other"], "other"),
    "parse_confidence": (["low", "medium", "high"], "low"),
    "price_period": (["month", "week", "night", "unknown"], "unknown"),
    "season": (["low", "high", "full_year", "unknown"], "unknown"),
    "property_type": (["house", "villa", "bungalow", "apartment", "studio", "room", "unknown"], "unknown"),
    "area_canonical": (AREA_ENUM, "unknown"),
    "near_construction": (["construction", "quiet", "unknown"], "unknown"),
}
LIST_FIELDS = {"furnishings_list"}
MIN_TEXT_LEN = 15
```

- [ ] **Step 4: Replace `SYSTEM_PROMPT`** (lines 75-108) with the new schema prompt

```python
SYSTEM_PROMPT = f"""You extract structured data from ONE rental-related message posted \
in a Koh Phangan (Thailand) Facebook housing group.

Return ONE JSON object with EXACTLY these keys (no others):
{json.dumps(MODEL_FIELDS, indent=2)}

GENERAL RULES
- Output ONLY the JSON object. No prose, markdown, or code fences.
- Use null for anything not stated. NEVER invent values.
- Booleans are true ONLY if clearly present, false ONLY if clearly stated absent, \
null if not mentioned. null means UNKNOWN, which is NOT the same as false.

CLASSIFICATION
- discard_reason: set to one of these when the post should be excluded, else null:
  - "not_a_listing": not about renting a place to live (bike/scooter/car, job, \
service, item for sale, pet, event, general chat).
  - "wanted": the poster is LOOKING FOR a place, not offering one.
  - "for_sale": the property is being SOLD, not rented (e.g. "for sale", "land for \
sale", a purchase/asking price to buy, leasehold/freehold sale). We want RENTALS only.
  - "not_koh_phangan": clearly for another location (Koh Samui, Koh Tao, mainland, etc.).
  - "not_long_term": clearly ONLY a short-term/holiday rental (nightly/weekly, "per \
night", holiday let) with no long-term option.
  A valid long-term-capable offer on Koh Phangan -> null.
- is_offer: "offer" (offering a place), "wanted" (seeking a place), or "ambiguous".
- post_language: "en", "th", "mixed", or "other" (any other language).
- parse_confidence: "high" if the post is clear and complete, "medium" if partial, \
"low" if vague/truncated/hard to read.
- multi_listing: true if the post bundles several distinct properties; then set \
parse_confidence to "low" and extract the FIRST property's details.

TIER 1 — CORE
- price_thb: integer THB, MONTHLY. "12,000 THB/month" -> 12000. If a range, the LOWER \
monthly value. If only nightly/weekly, leave price_thb null and set price_period.
- price_low_thb / price_high_thb: when the post quotes a seasonal/low–high range.
- price_period: "month" | "week" | "night" | "unknown".
- season: "low" | "high" | "full_year" | "unknown".
- property_type: house | villa | bungalow | apartment | studio | room | unknown.
- area_raw: the area/village as written (e.g. "Sri Thanu", "Hin Kong").
- area_canonical: map area_raw to EXACTLY ONE of: {", ".join(AREA_ENUM)}. Use "other" \
for a real KP area not in the list, "unknown" if no area is stated.

TIER 2 — LONG-TERM SUITABILITY: min_stay_months (integer), available_from (date or \
"now"), available_until (usually null); \
year_round: true if the place is explicitly available for the FULL year INCLUDING high \
season (Dec–Mar); false if it is low-season-only or the owner reclaims it / raises the \
price in high season; null if not stated. \
subletting_allowed: true if subletting / Airbnb / Booking is explicitly permitted, \
false if explicitly forbidden ("no Airbnb", "no subletting"), null if not mentioned.
TIER 3 — COST: deposit_thb, electricity_rate_thb_per_unit (number), water_included, \
internet_included (bool/null).
TIER 4 — AMENITIES (bool/null unless noted): has_aircon, has_wifi, furnished, \
has_kitchen, has_pool, has_parking, pet_friendly, sea_view, has_workspace, has_terrace, \
near_road; near_construction ("construction"|"quiet"|"unknown"); furnishings_list \
(short list of extras mentioned).
TIER 5 — CONTACT: contact_raw (phone/Line/WhatsApp as written), contact_phone \
(digits-normalized phone or null), size_sqm (integer or null)."""
```

- [ ] **Step 5: Add `extract_fields()` and a coercion helper** (after `call_claude`, ~line 127)

```python
def _empty_fields():
    return {k: None for k in MODEL_FIELDS}

def _coerce(fields):
    out = _empty_fields()
    for k, v in fields.items():
        if k not in out:
            continue
        if v is None:
            out[k] = None
        elif k in LIST_FIELDS and isinstance(v, list):
            out[k] = ", ".join(str(x) for x in v) or None
        elif k in ENUMS:
            allowed, fallback = ENUMS[k]
            s = str(v).strip().lower()
            out[k] = s if s in allowed else fallback
        else:
            out[k] = v
    return out

def extract_fields(text):
    """Pure: raw text -> normalized dict over MODEL_FIELDS (no side effects, no drop)."""
    text = (text or "").strip()
    if len(text) < MIN_TEXT_LEN:
        out = _empty_fields()
        out["discard_reason"] = "not_a_listing"
        out["parse_confidence"] = "low"
        return out
    return _coerce(call_claude(text))
```

- [ ] **Step 6: Bump `max_tokens` in `call_claude`** (40 fields need more room than the old 1024)

In `extract.py:113`, change `max_tokens=1024` to `max_tokens=2048`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `poetry run pytest tests/test_extract_fields.py -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
git add extract.py tests/test_extract_fields.py
git commit -m "feat: tiered listing schema + pure extract_fields()"
```

---

## Task 5: Repoint `extract.py` CSV `main()` onto `extract_fields()`

**Files:**
- Modify: `extract.py:148-213` (the `main()` body)

- [ ] **Step 1: Update `main()` to use the new engine** (CSV path keeps "drop junk" behavior)

Replace the per-row loop body (lines 173-201) so it calls `extract_fields` and drops only rows whose `discard_reason == "not_a_listing"`; write `MODEL_FIELDS` as columns alongside the original `text, contact, date, source, hash`:

```python
    out_cols = ["text", "contact", "date", "source", "hash"] + MODEL_FIELDS
    new_parsed = []
    dropped = 0
    for i, r in enumerate(todo, 1):
        fields = extract_fields(r.get("text") or "")
        if fields.get("discard_reason") == "not_a_listing":
            dropped += 1
            continue
        row = {c: r.get(c, "") for c in ["text", "contact", "date", "source", "hash"]}
        for k in MODEL_FIELDS:
            v = fields.get(k)
            row[k] = "" if v is None else ("true" if v is True else "false" if v is False else v)
        new_parsed.append(row)
        if i % 10 == 0 or i == len(todo):
            print(f"  parsed {i}/{len(todo)}")
```

(Keep the surrounding incremental-by-`hash` logic and the final CSV write unchanged.)

- [ ] **Step 2: Smoke-test the CSV path runs**

Run: `poetry run python -c "import extract; print(len(extract.MODEL_FIELDS), 'fields')"`
Expected: prints `40 fields` and no import error.

- [ ] **Step 3: Commit**

```bash
git add extract.py
git commit -m "refactor: CSV main() delegates to extract_fields"
```

---

## Task 6: `db.py` — supabase-py wrapper (paginated)

**Files:**
- Create: `db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test for the pagination helper**

```python
# tests/test_db.py
import db

def test_paginate_stops_on_short_page():
    pages = [list(range(1000)), list(range(1000, 1500))]  # 1000 then 500 (<page_size)
    calls = []
    def fetch_page(offset, size):
        calls.append((offset, size))
        return pages.pop(0) if pages else []
    out = db.paginate(fetch_page, page_size=1000)
    assert len(out) == 1500
    assert calls == [(0, 1000), (1000, 1000)]  # stops after the short second page

def test_paginate_single_short_page():
    out = db.paginate(lambda offset, size: [1, 2, 3] if offset == 0 else [], page_size=1000)
    assert out == [1, 2, 3]
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db'`.

- [ ] **Step 3: Implement `db.py`**

```python
#!/usr/bin/env python3
"""Supabase access for the KP Rentals analysis layer (supabase-py over REST)."""

import os
from collections import Counter
from supabase import create_client, Client

FB_TABLE = "fb_posts"
PARSED_TABLE = "listings_parsed"
PAGE_SIZE = 1000
FB_SELECT = "id,source,text,created_at"
FB_EXPECTED_COLS = {"id", "source", "text", "link", "date_raw", "url", "created_at"}


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_ANON_KEY in .env or the environment.")
    return create_client(url, key)


def paginate(fetch_page, page_size=PAGE_SIZE):
    """Call fetch_page(offset, size) until it returns fewer than page_size rows."""
    rows, offset = [], 0
    while True:
        page = fetch_page(offset, page_size)
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def existing_parsed_ids(client) -> set:
    def fetch(offset, size):
        res = client.table(PARSED_TABLE).select("id").range(offset, offset + size - 1).execute()
        return res.data or []
    return {r["id"] for r in paginate(fetch)}


def fetch_unparsed_fb_posts(client, existing_ids, limit=None) -> list:
    def fetch(offset, size):
        res = (client.table(FB_TABLE).select(FB_SELECT)
               .order("created_at").range(offset, offset + size - 1).execute())
        return res.data or []
    rows = [r for r in paginate(fetch) if r["id"] not in existing_ids]
    return rows[:limit] if limit else rows


def upsert_parsed(client, rows, batch=500):
    for i in range(0, len(rows), batch):
        client.table(PARSED_TABLE).upsert(rows[i:i + batch], on_conflict="id").execute()


def inspect(client):
    def fetch(offset, size):
        res = client.table(FB_TABLE).select("id,source").range(offset, offset + size - 1).execute()
        return res.data or []
    rows = paginate(fetch)
    print(f"{FB_TABLE}: {len(rows)} rows")
    for src, n in Counter(r["source"] for r in rows).most_common():
        print(f"  {src}: {n}")
    sample = client.table(FB_TABLE).select("*").limit(3).execute().data or []
    if sample:
        live = set(sample[0].keys())
        extra, missing = live - FB_EXPECTED_COLS, FB_EXPECTED_COLS - live
        if extra:
            print(f"  ! live columns not in handoff schema: {sorted(extra)}")
        if missing:
            print(f"  ! handoff columns missing live: {sorted(missing)}")
        print(f"  sample row keys: {sorted(live)}")
    parsed = client.table(PARSED_TABLE).select("id", count="exact").limit(1).execute()
    print(f"{PARSED_TABLE}: {parsed.count or 0} rows already parsed")
```

- [ ] **Step 4: Run to verify the pagination tests pass**

Run: `poetry run pytest tests/test_db.py -v`
Expected: PASS (2 passed). (Live-DB functions aren't unit-tested here — covered in Task 8.)

- [ ] **Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: db.py supabase wrapper with paginated reads"
```

---

## Task 7: `analyze.py` — CLI orchestrator

**Files:**
- Create: `analyze.py`
- Test: `tests/test_analyze.py`

- [ ] **Step 1: Write the failing test for the row mapper**

```python
# tests/test_analyze.py
import analyze
import extract

def test_build_parsed_row_maps_metadata_and_fields():
    post = {"id": "fbid_123", "source": "fb_999", "text": "2-bed villa Sri Thanu",
            "created_at": "2026-06-01T10:00:00Z"}
    fields = extract.extract_fields  # not called; supply a canned dict instead
    canned = {k: None for k in extract.MODEL_FIELDS}
    canned.update({"is_offer": "offer", "price_thb": 15000, "area_canonical": "srithanu"})
    row = analyze.build_parsed_row(post, canned)
    assert row["id"] == "fbid_123"
    assert row["source_table"] == "fb_posts"
    assert row["source"] == "fb_999"
    assert row["listed_at"] == "2026-06-01T10:00:00Z"
    assert row["raw_text"] == "2-bed villa Sri Thanu"
    assert row["parser_version"] == extract.PARSER_VERSION
    assert row["price_thb"] == 15000
    assert row["area_canonical"] == "srithanu"
    # no parsed_at in the payload (DB defaults it)
    assert "parsed_at" not in row
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analyze'`.

- [ ] **Step 3: Implement `analyze.py`**

```python
#!/usr/bin/env python3
"""Extract structured listings from Supabase fb_posts into listings_parsed.

Usage:
  python analyze.py --inspect        # row counts + samples + schema check, no writes
  python analyze.py                  # parse all un-parsed fb_posts
  python analyze.py --limit 5        # parse at most 5 new rows
"""

import os
import sys
import argparse
from dotenv import load_dotenv

import db
import extract


def build_parsed_row(post, fields):
    row = {
        "id": post["id"],
        "source_table": "fb_posts",
        "source": post.get("source"),
        "listed_at": post.get("created_at"),
        "raw_text": post.get("text"),
        "parser_version": extract.PARSER_VERSION,
    }
    row.update({k: fields.get(k) for k in extract.MODEL_FIELDS})
    return row


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="counts + samples, no writes")
    ap.add_argument("--limit", type=int, default=None, help="max new rows to parse")
    args = ap.parse_args()

    client = db.get_client()

    if args.inspect:
        db.inspect(client)
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in .env or the environment first.")

    existing = db.existing_parsed_ids(client)
    todo = db.fetch_unparsed_fb_posts(client, existing, args.limit)
    print(f"fb_posts un-parsed: {len(todo)} (already parsed: {len(existing)})")
    if not todo:
        print("Nothing new to extract.")
        return

    rows, dropped = [], 0
    for i, post in enumerate(todo, 1):
        fields = extract.extract_fields(post.get("text") or "")
        if fields.get("discard_reason"):
            dropped += 1
        rows.append(build_parsed_row(post, fields))
        if i % 10 == 0 or i == len(todo):
            print(f"  parsed {i}/{len(todo)}")

    db.upsert_parsed(client, rows)
    kept = len(rows) - dropped
    print(f"\nUpserted {len(rows)} rows ({kept} offers, {dropped} flagged/discarded).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the mapper test passes**

Run: `poetry run pytest tests/test_analyze.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Full unit suite green**

Run: `poetry run pytest -v`
Expected: all tests pass (extract_fields + db + analyze).

- [ ] **Step 6: Commit**

```bash
git add analyze.py tests/test_analyze.py
git commit -m "feat: analyze.py CLI (inspect + incremental fb_posts extraction)"
```

---

## Task 8: End-to-end verification (live Supabase + LLM)

**Files:** none (operational verification).

- [ ] **Step 1: Inspect (no writes, schema check)**

Run: `poetry run python analyze.py --inspect`
Expected: per-`source` counts for `fb_posts`, a sample-row key list, any column-drift warnings (e.g. leftover `date_inferred`), and `listings_parsed: 0 rows already parsed`.

- [ ] **Step 2: Small extract**

Run: `poetry run python analyze.py --limit 5`
Expected: "Upserted 5 rows ...". In Supabase Table Editor, `listings_parsed` has ≤5 rows; spot-check `price_thb`, `bedrooms`, `area_canonical`, `post_language`, and `listed_at` are populated sensibly on a real offer.

- [ ] **Step 3: Soft-flag check**

In Supabase, run `select id, discard_reason from listings_parsed where discard_reason is not null;` — confirm any non-listing/wanted/short-term row is present with the right reason, and that `select count(*) from listings_parsed where discard_reason is null;` counts only real offers.

- [ ] **Step 4: Incrementality / idempotency**

Run `poetry run python analyze.py` (full), note the upsert count. Run it again.
Expected: second run prints "Nothing new to extract." (0 new rows, 0 LLM calls) because every processed `id` — offers and flagged — is already in `listings_parsed`.

- [ ] **Step 5: Final commit (if any docs/notes changed)**

```bash
git add -A && git commit -m "test: verify fb_posts extraction end to end" || echo "nothing to commit"
```

---

## Notes for the executor
- **TDD:** write the test, watch it fail, implement, watch it pass, commit. Don't batch.
- **Don't touch** the Chrome extension, `combine.py`, or `wa_messages` — out of scope.
- **Secrets:** never commit real values in `.env`. Only `.env.example` is committed.
- **Cost:** `extract_fields` makes one LLM call per un-parsed row; `--limit` gates spend during verification. Run the full extract (Task 8 Step 4) only after the small run looks right.
