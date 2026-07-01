#!/usr/bin/env python3
"""
Combine KP Rentals Exporter JSON files (Facebook + WhatsApp) into one
deduplicated master CSV.

Step 1: no text parsing yet — just collect, dedupe, and append.

Columns:
  text     - the listing text
  contact  - WhatsApp: sender phone (or sender name if no phone)
             Facebook: the post link
  date     - the post/message datetime, if any
  source   - "facebook" or "whatsapp"
  hash     - internal dedupe key (normalized-text hash)

Dedup is two-layer:
  1. Within this run's input files.
  2. Against the existing master CSV, so running daily with overlapping
     JSONs never inserts a listing already present.

Usage (run from the project root):
  python src/combine.py ./listings          # combine all JSONs in ./listings -> listings.csv
  python src/combine.py mine.csv ./listings # -> mine.csv
  python src/combine.py a.json b.json       # -> listings.csv
  (Any arg ending in .csv is the output file; everything else is input.)
"""

import sys
import os
import csv
import json
import glob
import re
import hashlib

COLUMNS = ["text", "contact", "date", "source", "hash"]


def normalize(text):
    """Normalize text for duplicate comparison: lowercase, strip emoji/punct,
    collapse whitespace. Two posts that differ only in spacing/case/emoji
    will hash the same."""
    t = (text or "").lower()
    t = re.sub(r"https?://\S+", "", t)          # drop URLs (tracking varies)
    t = re.sub(r"[^a-z0-9\u0e00-\u0e7f ]+", " ", t)  # keep latin+thai+digits
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_hash(text):
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def load_existing(csv_path):
    """Return (rows, hashes) already in the master CSV."""
    rows, hashes = [], set()
    if not os.path.exists(csv_path):
        return rows, hashes
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            h = r.get("hash") or text_hash(r.get("text", ""))
            hashes.add(h)
    return rows, hashes


def rows_from_fb(data):
    """Facebook: one row per post. Contact = post link."""
    out = []
    for p in data.get("posts", []):
        text = (p.get("text") or "").strip()
        if len(text) < 15:
            continue
        out.append({
            "text": text,
            "contact": p.get("link", ""),
            "date": p.get("date", ""),
            "source": "facebook",
        })
    return out


def rows_from_wa(data):
    """WhatsApp: merge CONSECUTIVE messages from the same sender into one row
    (a listing is often split across several quick messages). Contact = sender
    phone, or sender name if no phone."""
    msgs = data.get("messages", [])
    groups = []
    for m in msgs:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        phone = (m.get("senderPhone") or "").strip()
        name = (m.get("senderName") or "").strip()
        contact = phone or name
        # group key: same contact AND not separated from previous by a
        # different sender. Empty-contact messages (forwarded/no-attr) only
        # merge with an adjacent empty-contact message.
        if groups and groups[-1]["_contact_key"] == contact:
            groups[-1]["text"] += "\n" + text
            # keep the first non-empty datetime/contact we saw for the group
            if not groups[-1]["date"] and m.get("datetime"):
                groups[-1]["date"] = m.get("datetime", "")
        else:
            groups.append({
                "text": text,
                "contact": contact,
                "date": m.get("datetime", ""),
                "source": "whatsapp",
                "_contact_key": contact,
            })
    for g in groups:
        g.pop("_contact_key", None)
    return groups


def collect_input_files(args):
    """Expand folders/globs into a flat list of .json paths."""
    files = []
    for a in args:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, "*.json")))
        elif any(ch in a for ch in "*?["):
            files += sorted(glob.glob(a))
        else:
            files.append(a)
    return files


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(
            "Usage: python src/combine.py [output.csv] <input.json | folder/> ...\n"
            "  - Any argument ending in .csv is the master CSV (optional;\n"
            "    defaults to listings.csv).\n"
            "  - Everything else is input: json files and/or folders.\n"
            "Examples:\n"
            "  python src/combine.py ./listings            # -> listings.csv\n"
            "  python src/combine.py mine.csv ./listings   # -> mine.csv\n"
            "  python src/combine.py a.json b.json         # -> listings.csv"
        )

    # split: the .csv arg (if any) is the output; the rest are inputs
    csv_args = [a for a in args if a.lower().endswith(".csv")]
    input_args = [a for a in args if not a.lower().endswith(".csv")]
    if len(csv_args) > 1:
        sys.exit("Specify at most one .csv output file.")
    csv_path = csv_args[0] if csv_args else "listings.csv"

    if not input_args:
        sys.exit("No input given. Pass JSON files or a folder of JSONs.")

    input_files = collect_input_files(input_args)
    if not input_files:
        sys.exit(f"No .json files found in: {', '.join(input_args)}")

    print(f"Output CSV: {csv_path}")
    existing_rows, seen_hashes = load_existing(csv_path)
    print(f"Existing CSV: {len(existing_rows)} rows.")

    new_rows = []
    for path in input_files:
        if not os.path.exists(path):
            print(f"  ! missing, skipping: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "posts" in data:
            rows = rows_from_fb(data)
            kind = "facebook"
        elif "messages" in data:
            rows = rows_from_wa(data)
            kind = "whatsapp"
        else:
            print(f"  ! unrecognized JSON (no posts/messages): {path}")
            continue

        added = 0
        for r in rows:
            h = text_hash(r["text"])
            if h in seen_hashes:
                continue                 # dupe: across runs OR within this batch
            seen_hashes.add(h)
            r["hash"] = h
            new_rows.append(r)
            added += 1
        print(f"  {os.path.basename(path)} [{kind}]: {len(rows)} rows, {added} new")

    if not new_rows:
        print("\nNo new listings to add.")
        return

    # write: existing rows first (unchanged), then new rows appended
    all_rows = existing_rows + new_rows
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})

    print(f"\nAdded {len(new_rows)} new listings. Master CSV now has {len(all_rows)} rows.")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
