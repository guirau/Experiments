#!/usr/bin/env python3
"""
Extract structured rental-listing fields from free text using the Claude SDK.

Primary engine (source-agnostic, used by the v0 notebook and the v1 Supabase
pipeline):
    extract_batch(texts: list[str]) -> list[dict]   # one LLM call per batch
    extract_fields(text: str)       -> dict          # thin single-row wrapper

LLM-call minimization:
  - texts shorter than MIN_TEXT_LEN are flagged not_a_listing WITHOUT a call,
  - the rest are sent as ONE batched call returning a JSON array,
  - on parse failure / length mismatch we fall back to per-row calls for that
    batch only (correctness preserved, savings kept on the happy path).

CSV path (legacy, still supported):
  export ANTHROPIC_API_KEY=sk-ant-...
  python extract.py                       # listings.csv -> parsed_listings.csv
  python extract.py in.csv out.csv        # custom names
"""

import os
import sys
import csv
import json
import re
from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5-20251001"
PARSER_VERSION = "fb-1.0"   # bump when the prompt/schema changes
BATCH_SIZE = 10             # posts per LLM call
MIN_TEXT_LEN = 15           # below this, skip the LLM and flag not_a_listing

# --- field schema -----------------------------------------------------------
# The model returns EXACTLY these keys as one JSON object per post.
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

# canonical Koh Phangan areas; area_canonical is coerced into this set.
AREA_ENUM = ["thong_sala", "ban_tai", "ban_kai", "haad_rin", "srithanu", "chaloklum",
             "mae_haad", "hin_kong", "woktum", "haad_yao", "haad_salad", "haad_son",
             "thong_nai_pan", "bottle_beach", "than_sadet", "haad_yuan_tien",
             "madeua_wan", "plai_laem", "other", "unknown"]

# enum field -> (allowed values, fallback for anything else)
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

# typed SQL columns -> coerced so PG never rejects a stray LLM value (invalid -> None).
INT_FIELDS = {"price_thb", "price_low_thb", "price_high_thb", "bedrooms", "bathrooms",
              "min_stay_months", "deposit_thb", "size_sqm"}
NUM_FIELDS = {"electricity_rate_thb_per_unit"}
BOOL_FIELDS = {"multi_listing", "year_round", "subletting_allowed", "water_included",
               "internet_included", "has_aircon", "has_wifi", "furnished", "has_kitchen",
               "has_pool", "has_parking", "pet_friendly", "sea_view", "has_workspace",
               "has_terrace", "near_road"}
DATE_FIELDS = {"available_until"}   # PG `date`; only ISO YYYY-MM-DD survives, else None
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_client = None


def get_client():
    """Lazily build the Anthropic client so .env is loaded first (see main)."""
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


# --- prompts ----------------------------------------------------------------
# Shared rule block, reused by both the single and the batch system prompts.
_SCHEMA_RULES = f"""GENERAL RULES
- Output ONLY JSON. No prose, no markdown, no code fences.
- Use null for anything not stated. NEVER invent or guess values.
- Booleans are true ONLY if clearly present, false ONLY if clearly stated absent, \
null if not mentioned. null means UNKNOWN, which is NOT the same as false.

CLASSIFICATION
- discard_reason: set to ONE of these when the post should be excluded, else null:
  - "not_a_listing": not about renting a place to live (bike/scooter/car, job, \
service, item for sale, pet, event, general chat). Do NOT use not_a_listing merely \
because a post is short or truncated ("See more" / "…"): if it clearly refers to a \
dwelling to rent (house/villa/bungalow/apartment/studio/room) keep it (discard_reason \
null) with parse_confidence "low", even when most details are missing.
  - "wanted": the poster is LOOKING FOR a place, not offering one.
  - "for_sale": the property is being SOLD, not rented (e.g. "for sale", "land for \
sale", a purchase/asking price to buy, leasehold/freehold sale). We want RENTALS only.
  - "not_koh_phangan": clearly for another location (Koh Samui, Koh Tao, mainland, etc.).
  - "not_long_term": clearly ONLY a short-term/holiday rental (nightly/weekly, "per \
night", holiday let) with no long-term option.
  A valid long-term-capable rental OFFER on Koh Phangan -> null.
- is_offer: "offer" (offering a place), "wanted" (seeking a place), or "ambiguous". \
A dwelling described as "for rent" / "available" is an "offer" even if the text is \
truncated; use "ambiguous" only when the intent is genuinely unclear.
- post_language: "en", "th", "mixed", or "other" (any other language).
- parse_confidence: "high" if clear and complete, "medium" if partial, "low" if \
vague/truncated/hard to read.
- multi_listing: true if the post bundles several distinct properties; then set \
parse_confidence to "low" and extract the FIRST property's details.

TIER 1 — CORE
- price_thb: integer THB, MONTHLY. "12,000 THB/month" -> 12000. If a range, the LOWER \
monthly value. If only nightly/weekly, leave price_thb null and set price_period.
- price_low_thb / price_high_thb: when the post quotes a seasonal/low-high range.
- price_period: "month" | "week" | "night" | "unknown".
- season: "low" | "high" | "full_year" | "unknown".
- property_type: house | villa | bungalow | apartment | studio | room | unknown.
- area_raw: the area/village as written (e.g. "Sri Thanu", "Hin Kong").
- area_canonical: map area_raw to EXACTLY ONE of: {", ".join(AREA_ENUM)}. Use "other" \
for a real KP area not in the list, "unknown" if no area is stated.

TIER 2 — LONG-TERM SUITABILITY: min_stay_months (integer), available_from (date or \
"now"), available_until (usually null); \
year_round: true if the place is explicitly available for the FULL year INCLUDING high \
season (Dec-Mar); false if it is low-season-only or the owner reclaims it / raises the \
price in high season; null if not stated. \
subletting_allowed: true if subletting / Airbnb / Booking is explicitly permitted, \
false if explicitly forbidden ("no Airbnb", "no subletting"), null if not mentioned.
TIER 3 — COST: deposit_thb, electricity_rate_thb_per_unit (number), water_included, \
internet_included (bool/null).
TIER 4 — AMENITIES (bool/null unless noted): has_aircon, has_wifi, furnished, \
has_kitchen, has_pool, has_parking, pet_friendly, sea_view, has_workspace, has_terrace, \
near_road; near_construction ("construction"|"quiet"|"unknown"); furnishings_list \
(short comma-free list of extras mentioned).
TIER 5 — CONTACT: contact_raw (phone/Line/WhatsApp as written), contact_phone \
(digits-normalized phone or null), size_sqm (integer or null)."""

_KEYS_JSON = json.dumps(MODEL_FIELDS, indent=2)

SYSTEM_PROMPT = f"""You extract structured data from ONE rental-related message posted \
in a Koh Phangan (Thailand) Facebook housing group.

Return ONE JSON object with EXACTLY these keys (no others):
{_KEYS_JSON}

{_SCHEMA_RULES}"""

SYSTEM_PROMPT_BATCH = f"""You extract structured data from rental-related messages \
posted in Koh Phangan (Thailand) Facebook housing groups.

You will receive several posts, each delimited and numbered as "POST k:". Return a \
JSON ARRAY containing EXACTLY ONE object per post, IN THE SAME ORDER as given. Do not \
merge, skip, or reorder posts. Each object MUST have EXACTLY these keys (no others):
{_KEYS_JSON}

{_SCHEMA_RULES}"""


def _strip_fences(raw):
    return re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()


def call_claude(text):
    """Single-post call -> dict. Returns {} on unparseable output (caller defaults)."""
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = _strip_fences("".join(b.text for b in resp.content if b.type == "text").strip())
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        print("  ! could not parse model JSON for one row, leaving fields blank",
              file=sys.stderr)
        return {}


def call_claude_batch(texts):
    """Batched call -> list[dict] (one per input, in order). Raises on bad JSON.

    Length validation lives in extract_batch so a wrong-length list triggers the
    per-row fallback there.
    """
    numbered = "\n\n=====\n\n".join(f"POST {i}:\n{t}" for i, t in enumerate(texts, 1))
    user = (f"There are {len(texts)} posts below. Return a JSON array of exactly "
            f"{len(texts)} objects, in the same order.\n\n{numbered}")
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=min(8192, 800 * len(texts) + 512),
        system=SYSTEM_PROMPT_BATCH,
        messages=[{"role": "user", "content": user}],
    )
    raw = _strip_fences("".join(b.text for b in resp.content if b.type == "text").strip())
    obj = json.loads(raw)            # JSONDecodeError -> caught by extract_batch
    if not isinstance(obj, list):
        raise ValueError("batch response was not a JSON array")
    return obj


def _empty_fields():
    return {k: None for k in MODEL_FIELDS}


def _to_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    m = re.search(r"-?\d[\d,]*", str(v))
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _to_num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    m = re.search(r"-?\d[\d,]*\.?\d*", str(v))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "yes"):
        return True
    if s in ("false", "no"):
        return False
    return None


def _to_date(v):
    s = str(v).strip()
    return s if _ISO_DATE_RE.match(s) else None


def _coerce(fields):
    """Normalize a raw model dict into MODEL_FIELDS: lowercase enums (fallback on
    invalid), coerce typed columns (int/numeric/bool/date; invalid -> None), join list
    fields, keep None as unknown, drop unknown keys."""
    out = _empty_fields()
    if not isinstance(fields, dict):
        return out
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
        elif k in INT_FIELDS:
            out[k] = _to_int(v)
        elif k in NUM_FIELDS:
            out[k] = _to_num(v)
        elif k in BOOL_FIELDS:
            out[k] = _to_bool(v)
        elif k in DATE_FIELDS:
            out[k] = _to_date(v)
        else:
            out[k] = v
    return out


def _short_text_fields():
    f = _empty_fields()
    f["discard_reason"] = "not_a_listing"
    f["parse_confidence"] = "low"
    return f


def extract_batch(texts):
    """Raw texts -> normalized dicts over MODEL_FIELDS, aligned to input order.

    Short texts skip the LLM. The rest go in ONE batched call; on parse failure or
    length mismatch, fall back to per-row calls for that batch only.
    """
    out = [None] * len(texts)
    call_idx, call_texts = [], []
    for i, raw in enumerate(texts):
        t = (raw or "").strip()
        if len(t) < MIN_TEXT_LEN:
            out[i] = _short_text_fields()
        else:
            call_idx.append(i)
            call_texts.append(t)

    if call_texts:
        try:
            results = call_claude_batch(call_texts)
            if not isinstance(results, list) or len(results) != len(call_texts):
                raise ValueError(
                    f"batch returned {len(results) if isinstance(results, list) else type(results).__name__} "
                    f"for {len(call_texts)} posts")
        except Exception as e:  # JSON error, length mismatch, API hiccup -> per-row
            print(f"  ! batch failed ({e}); falling back to per-row calls", file=sys.stderr)
            results = [call_claude(t) for t in call_texts]
        for idx, res in zip(call_idx, results):
            out[idx] = _coerce(res)

    return out


def extract_fields(text):
    """Single-text convenience wrapper used by the CSV path + tests."""
    return extract_batch([text])[0]


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


def _csv_value(v):
    if v is None:
        return ""
    if v is True:
        return "true"
    if v is False:
        return "false"
    return v


def main():
    load_dotenv()  # read .env into os.environ if present
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in .env or the environment first.")

    args = list(sys.argv[1:])
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

    base_cols = ["text", "contact", "date", "source", "hash"]
    out_cols = base_cols + MODEL_FIELDS
    new_parsed = []
    dropped = 0
    for start in range(0, len(todo), BATCH_SIZE):
        chunk = todo[start:start + BATCH_SIZE]
        fields_list = extract_batch([r.get("text") or "" for r in chunk])
        for r, fields in zip(chunk, fields_list):
            # CSV path keeps the "drop obvious junk" behavior (only not_a_listing).
            if fields.get("discard_reason") == "not_a_listing":
                dropped += 1
                continue
            row = {c: r.get(c, "") for c in base_cols}
            for k in MODEL_FIELDS:
                row[k] = _csv_value(fields.get(k))
            new_parsed.append(row)
        done_n = min(start + BATCH_SIZE, len(todo))
        print(f"  parsed {done_n}/{len(todo)}")

    all_rows = existing_parsed + new_parsed
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in out_cols})

    msg = f"\nExtracted {len(new_parsed)} new rows. "
    if dropped:
        msg += f"Dropped {dropped} not-a-listing row(s). "
    msg += f"{out_path} now has {len(all_rows)} rows."
    print(msg)


if __name__ == "__main__":
    main()
