-- KP Rentals Exporter — Supabase schema
-- Run these in the Supabase SQL Editor (Dashboard → SQL Editor → New query).
--
-- STEP 1: run this whole block. When prompted, choose "Run without RLS"
--         (RLS is enabled explicitly in STEP 2 with the right policies).

create table if not exists fb_posts (
  id          text primary key,    -- canonical post id (fbid_<postid>) or fallback
  source      text not null,       -- which group, e.g. 'fb_2478665265528328'
  text        text not null,
  link        text,                -- a link that reaches the post (photo url ok)
  date_raw    text,                -- YYYY-MM-DD when FB exposes a readable date,
                                   -- else null (older dates are CSS-obfuscated)
  url         text,                -- the POST permalink (/groups/<g>/posts/<id>/)
  created_at  timestamptz not null default now()
);
create index if not exists fb_posts_source_idx  on fb_posts (source);
create index if not exists fb_posts_created_idx on fb_posts (created_at);

create table if not exists wa_messages (
  id            text primary key,  -- stable message id
  source        text not null,     -- which chat, e.g. 'wa_housing_long_term_koh'
  chat          text,              -- human-readable chat name
  text          text not null,
  sender_phone  text,
  sender_name   text,
  datetime      text,              -- the [HH:MM, D/M/YYYY] string, as-is
  ts            bigint,            -- epoch millis if parsed, else null
  url           text,
  created_at    timestamptz not null default now()
);
create index if not exists wa_messages_source_idx on wa_messages (source);
create index if not exists wa_messages_ts_idx     on wa_messages (ts);


-- STEP 2: run this block AFTER step 1 succeeds. It turns on RLS and adds
--         policies so the extension's anon key can insert and read.

alter table fb_posts    enable row level security;
alter table wa_messages enable row level security;

create policy "anon insert fb" on fb_posts    for insert to anon with check (true);
create policy "anon select fb" on fb_posts    for select to anon using (true);
create policy "anon insert wa" on wa_messages for insert to anon with check (true);
create policy "anon select wa" on wa_messages for select to anon using (true);

-- Note: these policies allow anyone holding your anon key to insert/read.
-- That's the normal pattern for a personal client-side tool. Don't reuse this
-- project for sensitive data without tightening the policies.
