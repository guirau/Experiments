#!/usr/bin/env python3
"""Supabase access for the KP Rentals analysis layer (supabase-py over REST).

All full-table reads go through paginate(): PostgREST caps a single response at
1000 rows by default, so a naive .select() silently truncates — which would both
re-send already-parsed rows to the LLM and miss rows past the cap.
"""

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
    """All ids already in listings_parsed (so we never re-extract them)."""
    def fetch(offset, size):
        res = (client.table(PARSED_TABLE).select("id")
               .range(offset, offset + size - 1).execute())
        return res.data or []
    return {r["id"] for r in paginate(fetch)}


def fetch_unparsed_fb_posts(client, existing_ids, limit=None) -> list:
    """fb_posts rows whose id is not yet in listings_parsed, oldest first."""
    def fetch(offset, size):
        res = (client.table(FB_TABLE).select(FB_SELECT)
               .order("created_at").range(offset, offset + size - 1).execute())
        return res.data or []
    rows = [r for r in paginate(fetch) if r["id"] not in existing_ids]
    return rows[:limit] if limit else rows


def upsert_parsed(client, rows, batch=500):
    """Idempotent upsert keyed on id (re-runs overwrite, never duplicate)."""
    for i in range(0, len(rows), batch):
        client.table(PARSED_TABLE).upsert(rows[i:i + batch], on_conflict="id").execute()


def inspect(client):
    """Print per-source counts, a sample row's keys (vs the handoff schema), and the
    number of already-parsed rows. No writes."""
    def fetch(offset, size):
        res = (client.table(FB_TABLE).select("id,source")
               .range(offset, offset + size - 1).execute())
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
