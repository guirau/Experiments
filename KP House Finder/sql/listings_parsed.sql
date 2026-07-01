-- listings_parsed: structured fields extracted from fb_posts (and later wa_messages).
-- Run in the Supabase SQL Editor. STEP 1 creates the table; STEP 2 enables RLS + anon
-- policies (mirrors chrome_extension/KP-Rentals-Exporter/supabase_schema.sql STEP 2).
-- Keep columns in sync with extract.py MODEL_FIELDS + analyze.build_parsed_row.

-- STEP 1 -------------------------------------------------------------------
create table if not exists listings_parsed (
  -- control / metadata
  id                            text primary key,   -- = fb_posts.id
  source_table                  text not null,      -- 'fb_posts'
  source                        text,               -- FB group
  listed_at                     timestamptz,        -- = fb_posts.created_at
  raw_text                      text,
  parser_version                text,
  parsed_at                     timestamptz not null default now(),
  -- classification
  discard_reason                text,               -- not_a_listing|wanted|for_sale|not_koh_phangan|not_long_term|null
  is_offer                      text,               -- offer|wanted|ambiguous
  post_language                 text,               -- en|th|mixed|other
  parse_confidence              text,               -- low|medium|high
  multi_listing                 boolean,
  -- tier 1
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
  year_round                    boolean,            -- full year incl. high season; null = unknown
  subletting_allowed            boolean,            -- Airbnb/Booking permitted; null = unknown
  -- tier 3 (cost)
  deposit_thb                   integer,
  electricity_rate_thb_per_unit numeric,
  water_included                boolean,            -- null = unknown
  internet_included             boolean,
  -- tier 4 (amenities; null = unknown)
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
  -- tier 5 (contact / misc)
  contact_raw                   text,
  contact_phone                 text,
  size_sqm                      integer
);
create index if not exists listings_parsed_source_idx     on listings_parsed (source);
create index if not exists listings_parsed_discard_idx    on listings_parsed (discard_reason);
create index if not exists listings_parsed_parser_ver_idx on listings_parsed (parser_version);

-- STEP 2 -------------------------------------------------------------------
alter table listings_parsed enable row level security;
create policy "anon insert parsed" on listings_parsed for insert to anon with check (true);
create policy "anon select parsed" on listings_parsed for select to anon using (true);
create policy "anon update parsed" on listings_parsed for update to anon using (true) with check (true);

-- Offers view used for the house search:
--   select * from listings_parsed where discard_reason is null order by listed_at desc;
