"""Unit tests for enrich.py — the SerpApi cost guard + response mapping.

The single most important test here is the guard: enrich.py must NEVER select a prospect
the user did not queue on the map, or it would spend paid SerpApi calls on the wrong places.
Live SerpApi calls are exercised manually in the end-to-end run, not here.
"""

from datetime import date

import enrich


def test_queued_only_keeps_only_queued_rows():
    rows = [
        {"place_id": "a", "enrich_status": "queued"},
        {"place_id": "b", "enrich_status": "discovered"},
        {"place_id": "c", "enrich_status": "enriched"},
        {"place_id": "d", "enrich_status": "queued"},
        {"place_id": "e", "enrich_status": "error"},
        {"place_id": "f"},  # missing status must not slip through
    ]
    kept = enrich.queued_only(rows)
    assert [r["place_id"] for r in kept] == ["a", "d"]


def test_map_hotel_result_extracts_rate_and_offers():
    prop = {
        "name": "Sunset Villas",
        "rate_per_night": {"lowest": "฿1,500", "extracted_lowest": 1500},
        "prices": [
            {"source": "Booking.com", "rate_per_night": {"extracted_lowest": 1500}, "link": "http://b"},
            {"source": "Agoda", "rate_per_night": {"extracted_lowest": 1600}, "link": "http://a"},
        ],
    }
    patch = enrich.map_hotel_result(prop)
    assert patch["rate_per_night_thb"] == 1500
    assert patch["ota_offers"] == [
        {"source": "Booking.com", "price": 1500, "link": "http://b"},
        {"source": "Agoda", "price": 1600, "link": "http://a"},
    ]


def test_map_hotel_result_tolerates_missing_fields():
    patch = enrich.map_hotel_result({"name": "Bare Bungalow"})
    assert patch == {"rate_per_night_thb": None, "bedrooms": None,
                     "bathrooms": None, "ota_offers": None}


def test_best_match_prefers_name_substring():
    props = [{"name": "Random Hostel"}, {"name": "Sunset Villas Resort"}]
    assert enrich.best_match(props, "Sunset Villas")["name"] == "Sunset Villas Resort"


def test_best_match_returns_none_when_no_overlap():
    props = [{"name": "Completely Different Place"}]
    assert enrich.best_match(props, "Sunset Villas") is None


def test_stay_dates_check_in_before_check_out():
    check_in, check_out = enrich.stay_dates(date(2026, 7, 3))
    assert check_in < check_out
    assert check_in == "2026-08-02"  # +30 days
    assert check_out == "2026-08-03"


def test_run_enrichment_updates_each_queued_row(monkeypatch):
    # 'a' matches (enriched), 'b' has no match (error) — the extracted loop must persist both
    # results and return the right counts, proving the refactor is behavior-preserving.
    queued = [{"place_id": "a", "name": "A"}, {"place_id": "b", "name": "B"}]

    def fake_enrich_prospect(session, api_key, row, check_in, check_out):
        return ({"rate_per_night_thb": 1500}, "enriched") if row["place_id"] == "a" else ({}, "error")
    monkeypatch.setattr(enrich, "enrich_prospect", fake_enrich_prospect)

    calls = []
    monkeypatch.setattr(enrich.db, "update_prospect", lambda client, pid, patch: calls.append((pid, patch)))

    enriched, errored = enrich.run_enrichment(client=None, api_key="k", queued=queued)

    assert (enriched, errored) == (1, 1)
    assert [pid for pid, _ in calls] == ["a", "b"]           # every queued row written, in order
    assert calls[0][1]["enrich_status"] == "enriched"
    assert calls[0][1]["rate_per_night_thb"] == 1500
    assert "enriched_at" in calls[0][1]
    assert calls[1][1]["enrich_status"] == "error"
