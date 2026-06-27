-- tracker: personal pipeline of listings you're actively pursuing (manual data entry).
-- Run in the Supabase SQL Editor (Dashboard -> SQL Editor -> New query). STEP 1 creates
-- the table; STEP 2 enables RLS + anon policies (mirrors listings_parsed.sql).

-- STEP 1
create table if not exists tracker (
  id            uuid primary key default gen_random_uuid(),
  listing_url   text,           -- link to the listing (FB post or any URL)
  person_name   text,           -- who the listing belongs to
  price         integer,        -- price in THB
  location_url  text,           -- Google Maps link
  visit_date    date,           -- planned visit date
  contact       text,           -- WhatsApp / Facebook contact
  notify_before date,           -- deadline to tell the person your decision
  notes         text,
  crossed_off   boolean default false,  -- soft cross-off (kept in DB, just faded in the UI)
  sort_order    integer,                -- manual row order (drag-and-drop)
  created_at    timestamptz not null default now()
);

-- If you created this table earlier, run ONLY the matching lines to add new columns:
alter table tracker add column if not exists price integer;
alter table tracker add column if not exists crossed_off boolean default false;
alter table tracker add column if not exists sort_order integer;

-- STEP 2
alter table tracker enable row level security;
create policy "anon select tracker" on tracker for select to anon using (true);
create policy "anon insert tracker" on tracker for insert to anon with check (true);
create policy "anon update tracker" on tracker for update to anon using (true) with check (true);
create policy "anon delete tracker" on tracker for delete to anon using (true);
