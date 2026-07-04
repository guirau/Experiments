"""Unit tests for discover.py pure transforms (Places result -> row, score coercion)."""

import discover


def test_build_prospect_row_maps_places_fields():
    place = {
        "id": "ChIJ_test", "displayName": {"text": "Sunset Villas"},
        "formattedAddress": "Srithanu, Koh Phangan", "location": {"latitude": 9.74, "longitude": 100.0},
        "rating": 4.6, "userRatingCount": 212, "primaryType": "resort_hotel",
        "nationalPhoneNumber": "077 000 000", "websiteUri": "http://sunset",
    }
    row = discover.build_prospect_row(place)
    assert row["place_id"] == "ChIJ_test"
    assert row["name"] == "Sunset Villas"
    assert row["lat"] == 9.74 and row["lng"] == 100.0
    assert row["phone"] == "077 000 000"
    assert row["google_rating"] == 4.6 and row["user_ratings_total"] == 212
    assert row["source"] == "google_maps"
    # enrich_status / suitability_score are intentionally omitted so upsert preserves them
    assert "enrich_status" not in row
    assert "suitability_score" not in row


def test_build_prospect_row_tolerates_missing_optional_fields():
    row = discover.build_prospect_row({"id": "x", "location": {}})
    assert row["place_id"] == "x"
    assert row["name"] is None
    assert row["lat"] is None and row["lng"] is None
    assert row["google_rating"] is None


def test_coerce_score_clamps_to_0_10():
    assert discover.coerce_score({"suitability_score": 15, "suitability_reason": "great"}) == {
        "suitability_score": 10, "suitability_reason": "great"}
    assert discover.coerce_score({"suitability_score": -4})["suitability_score"] == 0


def test_coerce_score_handles_garbage():
    assert discover.coerce_score("not a dict") == {
        "suitability_score": None, "suitability_reason": None}
    assert discover.coerce_score({"suitability_score": True})["suitability_score"] is None
