#!/usr/bin/env python3
"""Discover Koh Phangan accommodation businesses on Google Maps into `prospects`.

Two idempotent passes, both resumable:
  1. Google Places Text Search (New) -> upsert into `prospects` keyed on place_id.
     Re-running never duplicates and never resets enrich_status / suitability_score
     (the discovery payload simply omits those columns, so upsert preserves them).
  2. Claude suitability scoring over rows that have no suitability_score yet
     (reuses the Anthropic client + prompt-caching idioms from extract.py).

Prices are NOT fetched here — Google Places does not expose rental prices. Pricing is a
separate, user-gated step (src/enrich.py via SerpApi) on places selected from the map.

Usage (run from the project root):
  python src/discover.py --inspect     # prospect counts by enrich_status, no writes
  python src/discover.py --limit 10    # discover + score at most ~10 new places (trial)
  python src/discover.py               # full discovery + score all unscored rows
"""

import os
import sys
import json
import argparse

import requests
from dotenv import load_dotenv

import db
import extract  # reuse: get_client(), MODEL, _cached_system(), _strip_fences()

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELD_MASK = ",".join([
    "places.id", "places.displayName", "places.formattedAddress", "places.location",
    "places.rating", "places.userRatingCount", "places.primaryType",
    "places.nationalPhoneNumber", "places.websiteUri", "nextPageToken",
])
# Text queries covering the property types we want to cold-pitch on Koh Phangan.
SEARCH_QUERIES = [
    "villas for rent in Koh Phangan",
    "bungalows in Koh Phangan",
    "houses for rent in Koh Phangan",
    "long term rental Koh Phangan",
    "resort in Koh Phangan",
    "guesthouse in Koh Phangan",
]
# Bias results to the island so unrelated mainland/Samui places drop out.
KP_CENTER = {"latitude": 9.7500, "longitude": 100.0333}
KP_RADIUS_M = 15000.0
MAX_PAGES_PER_QUERY = 3          # Places returns up to 20/page, 60 total
SCORE_BATCH_SIZE = 15
REQUEST_TIMEOUT = 30


# --- pure transforms (unit-testable, no network) ----------------------------

def build_prospect_row(place):
    """A Places (New) result dict -> a `prospects` upsert payload.

    Intentionally omits enrich_status / suitability_score so re-running discovery
    preserves any queue state and scores already on the row (upsert updates only the
    columns present here).
    """
    loc = place.get("location") or {}
    name = (place.get("displayName") or {}).get("text")
    return {
        "place_id": place.get("id"),
        "name": name,
        "formatted_address": place.get("formattedAddress"),
        "lat": loc.get("latitude"),
        "lng": loc.get("longitude"),
        "phone": place.get("nationalPhoneNumber"),
        "website": place.get("websiteUri"),
        "google_rating": place.get("rating"),
        "user_ratings_total": place.get("userRatingCount"),
        "property_type": place.get("primaryType"),
        "source": "google_maps",
    }


def coerce_score(raw):
    """Normalize one model result into {suitability_score:0-10|None, suitability_reason}."""
    score, reason = None, None
    if isinstance(raw, dict):
        v = raw.get("suitability_score")
        if isinstance(v, bool):
            v = None
        elif isinstance(v, (int, float)):
            score = max(0, min(10, int(v)))
        r = raw.get("suitability_reason")
        reason = str(r).strip()[:400] if r not in (None, "") else None
    return {"suitability_score": score, "suitability_reason": reason}


def score_input_line(p):
    """Compact one-line description of a prospect for the scoring prompt."""
    rating = p.get("google_rating")
    reviews = p.get("user_ratings_total")
    rating_str = f"{rating}★ ({reviews} reviews)" if rating is not None else "no rating"
    return (f"name: {p.get('name') or 'unknown'} | type: {p.get('property_type') or 'unknown'} "
            f"| {rating_str} | address: {p.get('formatted_address') or 'unknown'}")


SCORE_SYSTEM_PROMPT = """You score Koh Phangan (Thailand) accommodation businesses on how \
promising each is to cold-contact for a LONG-TERM (multi-month) rental for a SINGLE person.

You will receive several places, numbered "PLACE k:". Return a JSON ARRAY with EXACTLY ONE \
object per place, IN THE SAME ORDER. Each object has EXACTLY these keys:
  "suitability_score": integer 0-10
  "suitability_reason": one short sentence (max ~15 words)

Scoring guidance (0 = poor fit, 10 = excellent):
- Favor villas, bungalows, houses, small guesthouses that plausibly do monthly deals for one \
person; a solid rating with a healthy number of reviews is a plus.
- Penalize large hotels/resorts unlikely to negotiate long-term, party hostels, unrated or \
almost-no-review places, and anything that reads as short-stay-only.
- Judge only from the given fields. Do NOT invent details. Output ONLY the JSON array, no prose."""


def _call_claude_scores(prospects):
    """Batched scoring call -> list[dict] (one per input, in order). Raises on bad JSON."""
    numbered = "\n\n".join(f"PLACE {i}:\n{score_input_line(p)}"
                           for i, p in enumerate(prospects, 1))
    user = (f"There are {len(prospects)} places below. Return a JSON array of exactly "
            f"{len(prospects)} objects, in the same order.\n\n{numbered}")
    resp = extract.get_client().messages.create(
        model=extract.MODEL,
        max_tokens=min(4096, 120 * len(prospects) + 256),
        system=extract._cached_system(SCORE_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user}],
    )
    raw = extract._strip_fences(
        "".join(b.text for b in resp.content if b.type == "text").strip())
    obj = json.loads(raw)
    if not isinstance(obj, list):
        raise ValueError("score response was not a JSON array")
    return obj


def score_batch(prospects):
    """Prospect dicts -> coerced score dicts aligned to input order.

    On a parse failure / length mismatch, returns empty scores so the rows stay
    unscored and get retried on the next run (never a wrong score, never a crash).
    """
    try:
        results = _call_claude_scores(prospects)
        if len(results) != len(prospects):
            raise ValueError(f"scored {len(results)} for {len(prospects)} places")
    except Exception as e:  # noqa: BLE001 - degrade to "retry next run"
        print(f"  ! scoring batch failed ({e}); leaving these unscored", file=sys.stderr)
        return [{"suitability_score": None, "suitability_reason": None} for _ in prospects]
    return [coerce_score(r) for r in results]


# --- network I/O ------------------------------------------------------------

def _places_page(session, api_key, query, page_token=None):
    body = {"textQuery": query,
            "locationBias": {"circle": {"center": KP_CENTER, "radius": KP_RADIUS_M}}}
    if page_token:
        body["pageToken"] = page_token
    resp = session.post(
        PLACES_URL, json=body, timeout=REQUEST_TIMEOUT,
        headers={"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": PLACES_FIELD_MASK},
    )
    resp.raise_for_status()
    return resp.json()


def search_places(api_key, queries=SEARCH_QUERIES):
    """Run every query, page through results, dedupe by place_id. Returns list[place]."""
    session = requests.Session()
    seen, places = set(), []
    for query in queries:
        token, pages = None, 0
        while pages < MAX_PAGES_PER_QUERY:
            data = _places_page(session, api_key, query, token)
            for place in data.get("places", []):
                pid = place.get("id")
                if pid and pid not in seen:
                    seen.add(pid)
                    places.append(place)
            token = data.get("nextPageToken")
            pages += 1
            if not token:
                break
        print(f"  '{query}': {len(places)} unique places so far")
    return places


# --- orchestration ----------------------------------------------------------

def run_discovery(client, api_key, limit=None):
    """Search Places, upsert new/updated prospects. Returns count upserted."""
    places = search_places(api_key)
    rows = [r for r in (build_prospect_row(p) for p in places) if r["place_id"]]
    if limit:
        # Cost-gated trial: only introduce up to `limit` brand-new places this run.
        existing = db.existing_prospect_ids(client)
        new_rows = [r for r in rows if r["place_id"] not in existing][:limit]
        known_rows = [r for r in rows if r["place_id"] in existing]
        rows = known_rows + new_rows
    if rows:
        db.upsert_prospects(client, rows)
    print(f"Discovery: upserted {len(rows)} prospect rows.")
    return len(rows)


def run_scoring(client, limit=None):
    """Score prospects that have no suitability_score yet. Returns count scored."""
    todo = db.fetch_unscored_prospects(client, limit)
    print(f"Scoring: {len(todo)} unscored prospect(s).")
    scored = 0
    for start in range(0, len(todo), SCORE_BATCH_SIZE):
        chunk = todo[start:start + SCORE_BATCH_SIZE]
        for p, score in zip(chunk, score_batch(chunk)):
            db.update_prospect(client, p["place_id"], score)
            scored += 1
        print(f"  scored {min(start + SCORE_BATCH_SIZE, len(todo))}/{len(todo)}")
    return scored


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="prospect counts, no writes")
    ap.add_argument("--limit", type=int, default=None, help="max new places to discover/score")
    args = ap.parse_args()

    client = db.get_client()

    if args.inspect:
        db.inspect_prospects(client)
        return

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        sys.exit("Set GOOGLE_MAPS_API_KEY in .env or the environment first.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in .env or the environment first.")

    run_discovery(client, api_key, args.limit)
    run_scoring(client, args.limit)


if __name__ == "__main__":
    main()
