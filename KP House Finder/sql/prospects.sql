-- prospects: accommodation businesses discovered on Google Maps (villas, bungalows,
-- resorts, guesthouses) to cold-pitch a long-term deal. Populated by src/discover.py
-- (Google Places + Claude suitability score) and enriched by src/enrich.py (SerpApi
-- Google Hotels prices), which touches ONLY rows the user queued from the map.
--
-- Run in the Supabase SQL Editor. STEP 1 creates the table; STEP 2 enables RLS + anon
-- policies (mirrors listings_parsed.sql). Keyed by Google place_id so re-running
-- discovery upserts instead of duplicating (like listings_parsed keys on fb_posts.id).

-- STEP 1 -------------------------------------------------------------------
create table if not exists prospects (
  -- identity / discovery (Google Places)
  place_id            text primary key,   -- Google Place ID (upsert key)
  name                text,
  formatted_address   text,
  lat                 double precision,
  lng                 double precision,
  phone               text,
  website             text,
  google_rating       numeric,
  user_ratings_total  integer,
  property_type       text,               -- Google primaryType (e.g. lodging, resort_hotel)
  source              text default 'google_maps',
  -- Claude suitability scoring (for a solo long-stay tenant)
  suitability_score   integer,            -- 0-10; higher = better fit
  suitability_reason  text,
  -- enrichment lifecycle: discovered -> queued (from the map) -> enriched | error
  enrich_status       text default 'discovered',
  -- SerpApi Google Hotels enrichment (nightly price is a quality/anchor signal only)
  rate_per_night_thb  integer,
  bedrooms            integer,
  bathrooms           integer,
  ota_offers          jsonb,              -- [{source, price, link}] Booking/Agoda/Airbnb
  -- timestamps
  created_at          timestamptz not null default now(),
  enriched_at         timestamptz
);
create index if not exists prospects_enrich_status_idx on prospects (enrich_status);
create index if not exists prospects_suitability_idx   on prospects (suitability_score);

-- If you created this table earlier, run ONLY the matching lines to add new columns:
-- alter table prospects add column if not exists suitability_score  integer;
-- alter table prospects add column if not exists suitability_reason text;
-- alter table prospects add column if not exists ota_offers         jsonb;

-- STEP 2 -------------------------------------------------------------------
alter table prospects enable row level security;
create policy "anon insert prospects" on prospects for insert to anon with check (true);
create policy "anon select prospects" on prospects for select to anon using (true);
create policy "anon update prospects" on prospects for update to anon using (true) with check (true);

-- Best findings for the cards view:
--   select * from prospects order by suitability_score desc nulls last, google_rating desc;
-- The enrichment queue (exactly what src/enrich.py will spend SerpApi calls on):
--   select place_id, name from prospects where enrich_status = 'queued';
