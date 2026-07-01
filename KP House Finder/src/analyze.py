#!/usr/bin/env python3
"""Extract structured listings from Supabase fb_posts into listings_parsed.

Incremental + idempotent: only fb_posts ids not already in listings_parsed are
processed, and each is sent to the LLM at most once (in batches of BATCH_SIZE).

Usage (run from the project root):
  python src/analyze.py --inspect    # row counts + samples + schema check, no writes
  python src/analyze.py              # parse all un-parsed fb_posts
  python src/analyze.py --limit 5    # parse at most 5 new rows (cost-gated trial)
"""

import os
import sys
import argparse

from dotenv import load_dotenv

import db
import extract


def build_parsed_row(post, fields):
    """fb_posts row + extracted fields -> a listings_parsed payload dict.

    parsed_at is intentionally omitted so the DB default (now()) applies.
    """
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

    # Upsert per batch so a long run is crash-safe and resumable: persisted ids are
    # skipped on restart, so progress is never lost or re-charged to the LLM.
    total, dropped = 0, 0
    for start in range(0, len(todo), extract.BATCH_SIZE):
        chunk = todo[start:start + extract.BATCH_SIZE]
        fields_list = extract.extract_batch([p.get("text") or "" for p in chunk])
        batch_rows = []
        for post, fields in zip(chunk, fields_list):
            if fields.get("discard_reason"):
                dropped += 1
            batch_rows.append(build_parsed_row(post, fields))
        db.upsert_parsed(client, batch_rows)
        total += len(batch_rows)
        print(f"  upserted {total}/{len(todo)}")

    kept = total - dropped
    print(f"\nUpserted {total} rows ({kept} offers, {dropped} flagged/discarded).")


if __name__ == "__main__":
    main()
