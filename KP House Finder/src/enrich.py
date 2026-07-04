#!/usr/bin/env python3
"""Enrich map-selected prospects with pricing via the SerpApi Google Hotels API.

SAFETY GUARD (the whole point of this script): it enriches ONLY prospects the user
selected on the map — i.e. rows with enrich_status = 'queued'. That guard is enforced
twice: the DB query (db.fetch_queued_prospects) filters on 'queued', and queued_only()
filters again client-side before any paid SerpApi call. A queued row that succeeds
becomes 'enriched'; one that errors / has no match becomes 'error'. No other row is ever
touched, so SerpApi is only ever billed for explicitly-selected places.

Prices are NIGHTLY (a quality tier / negotiation anchor). The long-term monthly rate is
not published anywhere — you record it manually after contacting the owner.

Usage (run from the project root):
  python src/enrich.py --inspect   # list the queued places + SerpApi call count, no calls
  python src/enrich.py             # enrich exactly the queued places, then mark them done
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

import db

SERP_URL = "https://serpapi.com/search.json"
CURRENCY = "THB"
STAY_OFFSET_DAYS = 30      # look ~a month out so real availability/prices come back
REQUEST_TIMEOUT = 30
# Rough SerpApi cost per search (varies by plan) — used only for the --inspect estimate.
COST_PER_SEARCH_USD = 0.015


# --- pure helpers (unit-testable, no network) -------------------------------

def queued_only(rows):
    """Defense-in-depth: keep ONLY enrich_status == 'queued' rows.

    db.fetch_queued_prospects already filters at the DB, but this guarantees enrich.py
    never spends a SerpApi call on a non-queued place even if that query ever changes.
    """
    return [r for r in rows if r.get("enrich_status") == "queued"]


def _norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum() or ch.isspace()).strip()


def best_match(properties, name):
    """Pick the property whose name best matches `name` (substring, else token overlap)."""
    if not properties:
        return None
    target = _norm(name)
    if not target:
        return properties[0]
    target_tokens = set(target.split())
    best, best_score = None, -1.0
    for prop in properties:
        pname = _norm(prop.get("name"))
        if not pname:
            continue
        if target in pname or pname in target:
            return prop
        overlap = len(target_tokens & set(pname.split()))
        score = overlap / max(1, len(target_tokens))
        if score > best_score:
            best, best_score = prop, score
    # Require at least some token overlap to avoid matching an unrelated property.
    return best if best_score > 0 else None


def _extracted_rate(rate_obj):
    if isinstance(rate_obj, dict):
        v = rate_obj.get("extracted_lowest") or rate_obj.get("lowest")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def map_hotel_result(prop):
    """A SerpApi Google Hotels property dict -> a prospects enrichment patch.

    Missing fields stay None so a partial match still writes what it found.
    """
    if not isinstance(prop, dict):
        return {"rate_per_night_thb": None, "bedrooms": None, "bathrooms": None, "ota_offers": None}
    offers = []
    for entry in (prop.get("prices") or []):
        price = _extracted_rate(entry.get("rate_per_night")) or entry.get("extracted_price")
        offers.append({
            "source": entry.get("source"),
            "price": price,
            "link": entry.get("link"),
        })
    essential = prop.get("essential_info") or []
    bedrooms = _first_int_after(essential, ("bedroom", "bed"))
    bathrooms = _first_int_after(essential, ("bathroom", "bath"))
    return {
        "rate_per_night_thb": _extracted_rate(prop.get("rate_per_night")),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "ota_offers": offers or None,
    }


def _first_int_after(info_items, keywords):
    """Best-effort pull of a leading integer from an info string mentioning a keyword."""
    for item in info_items:
        low = str(item).lower()
        if any(k in low for k in keywords):
            digits = "".join(ch for ch in low if ch.isdigit())
            if digits:
                return int(digits[:2])
    return None


def stay_dates(today):
    """(check_in, check_out) ISO strings ~a month out for a 1-night query."""
    check_in = today + timedelta(days=STAY_OFFSET_DAYS)
    check_out = check_in + timedelta(days=1)
    return check_in.isoformat(), check_out.isoformat()


# --- network I/O ------------------------------------------------------------

def serp_hotels(session, api_key, query, check_in, check_out):
    resp = session.get(SERP_URL, timeout=REQUEST_TIMEOUT, params={
        "engine": "google_hotels", "q": query,
        "check_in_date": check_in, "check_out_date": check_out,
        "currency": CURRENCY, "gl": "th", "hl": "en",
        "vacation_rentals": "true", "api_key": api_key,
    })
    resp.raise_for_status()
    return resp.json()


# --- orchestration ----------------------------------------------------------

def enrich_prospect(session, api_key, row, check_in, check_out):
    """Enrich one queued prospect -> (patch, status). Never raises."""
    query = " ".join(x for x in [row.get("name"), "Koh Phangan"] if x)
    try:
        data = serp_hotels(session, api_key, query, check_in, check_out)
        prop = best_match(data.get("properties", []), row.get("name"))
        if not prop:
            return {}, "error"
        patch = map_hotel_result(prop)
        return patch, "enriched"
    except Exception as e:  # noqa: BLE001 - one bad place must not abort the batch
        print(f"  ! enrich failed for {row.get('name')!r}: {e}", file=sys.stderr)
        return {}, "error"


def run_enrichment(client, api_key, queued):
    """Enrich every already-queued row and write results back. Returns (enriched, errored).

    Callers (the CLI and the FastAPI service) are responsible for building `queued` via
    queued_only(db.fetch_queued_prospects(...)) so the cost guard is applied before we get here.
    """
    check_in, check_out = stay_dates(datetime.now(timezone.utc).date())
    session = requests.Session()
    enriched = errored = 0
    for i, row in enumerate(queued, 1):
        patch, status = enrich_prospect(session, api_key, row, check_in, check_out)
        patch = {**patch, "enrich_status": status,
                 "enriched_at": datetime.now(timezone.utc).isoformat()}
        db.update_prospect(client, row["place_id"], patch)
        if status == "enriched":
            enriched += 1
        else:
            errored += 1
        print(f"  {i}/{len(queued)} {row.get('name')}: {status}")
    return enriched, errored


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="list queued places + call count/cost, make no SerpApi calls")
    args = ap.parse_args()

    client = db.get_client()
    queued = queued_only(db.fetch_queued_prospects(client))  # guarded twice

    if not queued:
        print("No prospects are queued for enrichment "
              "(select places on the map and press 'Queue for enrichment').")
        return

    if args.inspect:
        print(f"{len(queued)} prospect(s) queued for enrichment "
              f"(~${len(queued) * COST_PER_SEARCH_USD:.2f} in SerpApi calls):")
        for r in queued:
            print(f"  - {r.get('name')}  [{r.get('place_id')}]")
        return

    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        sys.exit("Set SERPAPI_KEY in .env or the environment first.")

    enriched, errored = run_enrichment(client, api_key, queued)
    print(f"\nEnriched {enriched}, errored {errored}. Non-queued prospects untouched.")


if __name__ == "__main__":
    main()
