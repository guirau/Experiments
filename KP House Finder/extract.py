#!/usr/bin/env python3
"""
Extract structured listing fields from listings.csv (produced by combine.py)
using the Claude SDK, writing parsed_listings.csv.

Incremental: only rows whose `hash` is not already in parsed_listings.csv get
sent to Claude, so running daily is cheap.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python extract.py                       # listings.csv -> parsed_listings.csv
  python extract.py in.csv out.csv        # custom names

Output columns: the original (text, contact, date, source, hash) plus the
extracted fields below.
"""

import os
import sys
import csv
import json
import re
from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5-20251001"

# Extracted fields = the OUTPUT csv columns. The prompt is built from this list,
# plus one extra classification key ("is_housing", see MODEL_KEYS below) that the
# model returns but is NOT written as a column — it's only used to drop non-house
# rows. So prompt keys = MODEL_KEYS, output columns = EXTRACT_FIELDS (differ by one).
EXTRACT_FIELDS = [
    "is_offer",        # true = someone OFFERING a place; false = looking/wanted ad
    "rental_type",     # "long-term" / "short-term" / "both" / null
    "location",        # area/village, e.g. "Sri Thanu", "Hin Kong", "Ban Tai"
    "price_thb",       # monthly price in THB (integer). If a range, the lower value.
    "price_note",      # any nuance: range, "per night", seasonal, deposit, etc.
    "bedrooms",        # integer or null
    "bathrooms",       # integer or null
    "house_size_sqm",  # integer or null
    "available_from",  # date/phrase as written, or null
    "min_term",        # minimum rental term if stated, else null
    "aircon",          # true / false / null
    "wifi",            # true / false / null
    "pool",            # true / false / null
    "kitchen",         # true / false / null
    "washing_machine", # true / false / null
    "parking",         # true / false / null
    "furnished",       # true / false / null
    "pets_allowed",    # true / false / null
    "amenities",       # short comma-joined list of extras not covered above
    "electricity",     # electricity terms if stated (e.g. "8 THB/unit"), else null
    "contact_in_text", # phone/line/email/whatsapp found INSIDE the text, else null
    "truncated",       # true if the text was cut off ("Read more"/"See more")
    "notes",           # one short line for anything notable not captured above
]

ORIGINAL_COLS = ["text", "contact", "date", "source", "hash"]

# Classification key the model returns but we do NOT write to the CSV: used only
# to filter out listings that are not about a house/room/accommodation.
CLASSIFY_FIELD = "is_housing"
MODEL_KEYS = [CLASSIFY_FIELD] + EXTRACT_FIELDS

_client = None


def get_client():
    """Lazily build the Anthropic client so .env is loaded first (see main)."""
    global _client
    if _client is None:
        _client = Anthropic()
    return _client

SYSTEM_PROMPT = f"""You extract structured data from a single rental-listing message \
posted in a Koh Phangan (Thailand) housing group on Facebook or WhatsApp.

The text is one post/message. It may be an OFFER (someone renting a place out), \
a WANTED ad (someone looking for a place), short-term or long-term, and may be \
truncated with "Read more"/"See more".

Return ONE JSON object with exactly these keys:
{json.dumps(MODEL_KEYS, indent=2)}

Rules:
- Output ONLY the JSON object. No prose, no markdown, no code fences.
- Use null for anything not stated. NEVER invent or guess values.
- is_housing: NOT every message in this group is about a place to live. Set true \
ONLY if the message is about renting/finding a house, room, villa, bungalow, \
apartment, studio, condo, or any accommodation. Set false if it is about something \
else entirely: a motorbike/scooter/car/bicycle, a job, a service, a pet, an item \
for sale (furniture, phone, surfboard), an event/party, or general chat. This is \
independent of is_offer: a person LOOKING FOR a house is still housing (true).
- is_offer: true if the poster is offering a place to rent; false if they are \
looking for/seeking a place ("looking for", "we need", "anyone know a...").
- rental_type: "long-term" (months+), "short-term" (nightly/weekly/holiday), \
"both" if it explicitly offers both, else null.
- price_thb: integer Thai Baht, monthly if available. "12,000 THB/month" -> 12000. \
"55000/-per month" -> 55000. If only a nightly/weekly price, put it in price_note \
and set price_thb to the monthly figure only if stated.
- booleans (aircon, wifi, pool, etc.): true only if clearly present, false only if \
clearly stated absent (e.g. "no pets" -> pets_allowed false), null if not mentioned.
- amenities: short comma-separated extras (e.g. "TV, terrace, garden, fridge"). \
Don't repeat things already in dedicated fields.
- contact_in_text: copy any phone/line id/whatsapp/email found in the text verbatim; \
null if none.
- truncated: true if the text ends with or contains a "Read more"/"See more" cutoff.
- notes: at most one short sentence; null if nothing extra."""


def call_claude(text):
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        print("  ! could not parse model JSON for one row, leaving fields blank",
              file=sys.stderr)
        return {}


def load_parsed_hashes(out_path):
    """Hashes already extracted, so we skip them."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("hash"):
                done.add(r["hash"])
    return done


def load_parsed_rows(out_path):
    if not os.path.exists(out_path):
        return []
    with open(out_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    load_dotenv()  # read .env into os.environ if present
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in .env or the environment first.")

    args = [a for a in sys.argv[1:]]
    in_path = args[0] if len(args) >= 1 else "listings.csv"
    out_path = args[1] if len(args) >= 2 else "parsed_listings.csv"

    if not os.path.exists(in_path):
        sys.exit(f"Input not found: {in_path} (run combine.py first).")

    with open(in_path, encoding="utf-8") as f:
        in_rows = list(csv.DictReader(f))
    print(f"{in_path}: {len(in_rows)} rows.")

    done = load_parsed_hashes(out_path)
    existing_parsed = load_parsed_rows(out_path)
    todo = [r for r in in_rows if (r.get("hash") or "") not in done]
    print(f"Already parsed: {len(done)}. To parse now: {len(todo)}.")

    if not todo:
        print("Nothing new to extract.")
        return

    out_cols = ORIGINAL_COLS + EXTRACT_FIELDS
    new_parsed = []
    dropped = 0
    for i, r in enumerate(todo, 1):
        text = (r.get("text") or "").strip()
        fields = call_claude(text) if len(text) >= 15 else {}
        # Drop listings the model is confident are NOT housing (bikes, jobs,
        # items for sale, etc.). Conservative: only drop on an explicit False,
        # so uncertainty (None/missing/unparseable) keeps the row.
        if fields.get(CLASSIFY_FIELD) is False:
            dropped += 1
            if i % 10 == 0 or i == len(todo):
                print(f"  parsed {i}/{len(todo)}")
            continue
        row = {c: r.get(c, "") for c in ORIGINAL_COLS}
        for k in EXTRACT_FIELDS:
            v = fields.get(k, "")
            # flatten lists (amenities) to a comma string for CSV
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            elif isinstance(v, bool):
                v = "true" if v else "false"
            elif v is None:
                v = ""
            row[k] = v
        new_parsed.append(row)
        if i % 10 == 0 or i == len(todo):
            print(f"  parsed {i}/{len(todo)}")

    all_rows = existing_parsed + new_parsed
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in out_cols})

    msg = f"\nExtracted {len(new_parsed)} new rows. "
    if dropped:
        msg += f"Dropped {dropped} non-housing row(s). "
    msg += f"{out_path} now has {len(all_rows)} rows."
    print(msg)


if __name__ == "__main__":
    main()
