"""Unit tests for analyze.build_parsed_row (the fb_posts -> listings_parsed mapper)."""

import analyze
import extract


def test_build_parsed_row_maps_metadata_and_fields():
    post = {"id": "fbid_123", "source": "fb_999", "text": "2-bed villa Sri Thanu",
            "created_at": "2026-06-01T10:00:00Z"}
    canned = {k: None for k in extract.MODEL_FIELDS}
    canned.update({"is_offer": "offer", "price_thb": 15000, "area_canonical": "srithanu"})

    row = analyze.build_parsed_row(post, canned)

    assert row["id"] == "fbid_123"
    assert row["source_table"] == "fb_posts"
    assert row["source"] == "fb_999"
    assert row["listed_at"] == "2026-06-01T10:00:00Z"
    assert row["raw_text"] == "2-bed villa Sri Thanu"
    assert row["parser_version"] == extract.PARSER_VERSION
    assert row["price_thb"] == 15000
    assert row["area_canonical"] == "srithanu"
    # every model field is carried through
    for k in extract.MODEL_FIELDS:
        assert k in row
    # parsed_at is left to the DB default, not sent in the payload
    assert "parsed_at" not in row


def test_build_parsed_row_handles_missing_optional_post_fields():
    post = {"id": "fbid_x"}  # no source/text/created_at
    canned = {k: None for k in extract.MODEL_FIELDS}
    row = analyze.build_parsed_row(post, canned)
    assert row["id"] == "fbid_x"
    assert row["source_table"] == "fb_posts"
    assert row["source"] is None
    assert row["listed_at"] is None
    assert row["raw_text"] is None
