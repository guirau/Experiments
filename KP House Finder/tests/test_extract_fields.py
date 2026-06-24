"""Unit tests for the extract.py batching engine.

These tests are the behavioral spec for the LLM-call-minimizing design:
- short texts are filtered out BEFORE any LLM call (0 cost),
- a whole batch is parsed/normalized from ONE call,
- enums are lowercased and invalid values fall back to a default,
- a malformed / wrong-length batch transparently falls back to per-row calls,
- results always realign to the original input order.

No network: `call_claude_batch` / `call_claude` are monkeypatched.
"""

import extract


# ---------------------------------------------------------------------------
# short-text short-circuit (no LLM call at all)
# ---------------------------------------------------------------------------

def test_short_text_marked_not_a_listing_without_any_call(monkeypatch):
    def boom(*a, **k):  # any call here is a bug — short text must not hit the LLM
        raise AssertionError("LLM must not be called for short text")
    monkeypatch.setattr(extract, "call_claude_batch", boom)
    monkeypatch.setattr(extract, "call_claude", boom)

    [fields] = extract.extract_batch(["hi"])

    assert fields["discard_reason"] == "not_a_listing"
    assert fields["parse_confidence"] == "low"
    # every model key is present, defaulted to None unless set above
    for key in extract.MODEL_FIELDS:
        assert key in fields


# ---------------------------------------------------------------------------
# happy path: one batched call, normalized output, order preserved
# ---------------------------------------------------------------------------

def test_batch_one_call_normalizes_and_preserves_order(monkeypatch):
    calls = {"batch": 0, "single": 0}

    def fake_batch(texts):
        calls["batch"] += 1
        # model returns one object per input, in order, with messy casing/types
        return [
            {"is_offer": "OFFER", "post_language": "EN", "price_thb": 15000,
             "property_type": "Villa", "area_canonical": "srithanu",
             "has_pool": True, "has_wifi": None,
             "furnishings_list": ["fridge", "TV", "terrace"]},
            {"is_offer": "wanted", "post_language": "th", "bedrooms": 1,
             "area_canonical": "ban_tai"},
        ]

    def fake_single(text):
        calls["single"] += 1
        return {}

    monkeypatch.setattr(extract, "call_claude_batch", fake_batch)
    monkeypatch.setattr(extract, "call_claude", fake_single)

    out = extract.extract_batch(["a long enough villa listing here",
                                 "a long enough wanted post here"])

    assert calls == {"batch": 1, "single": 0}          # exactly ONE call for two posts
    assert len(out) == 2
    # row 0 normalized
    assert out[0]["is_offer"] == "offer"               # lowercased enum
    assert out[0]["post_language"] == "en"
    assert out[0]["price_thb"] == 15000                # int preserved
    assert out[0]["property_type"] == "villa"
    assert out[0]["has_pool"] is True                  # bool preserved
    assert out[0]["has_wifi"] is None                  # unknown preserved
    assert out[0]["furnishings_list"] == "fridge, TV, terrace"  # list joined
    # row 1 aligned to its input
    assert out[1]["is_offer"] == "wanted"
    assert out[1]["bedrooms"] == 1
    assert out[1]["area_canonical"] == "ban_tai"


def test_invalid_enum_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(extract, "call_claude_batch",
                        lambda texts: [{"area_canonical": "atlantis",
                                        "property_type": "spaceship",
                                        "is_offer": "maybe"}])
    [fields] = extract.extract_batch(["a long enough listing text for parsing here"])
    assert fields["area_canonical"] == "unknown"       # AREA_ENUM fallback
    assert fields["property_type"] == "unknown"
    assert fields["is_offer"] == "ambiguous"           # is_offer fallback


# ---------------------------------------------------------------------------
# robustness: bad batch -> per-row fallback (savings lost only for that batch)
# ---------------------------------------------------------------------------

def test_wrong_length_batch_triggers_per_row_fallback(monkeypatch):
    calls = {"batch": 0, "single": 0}

    def short_batch(texts):
        calls["batch"] += 1
        return [{"is_offer": "offer"}]                 # only 1 obj for 2 inputs -> mismatch

    def single(text):
        calls["single"] += 1
        return {"is_offer": "offer", "price_thb": 9000}

    monkeypatch.setattr(extract, "call_claude_batch", short_batch)
    monkeypatch.setattr(extract, "call_claude", single)

    out = extract.extract_batch(["a long enough listing number one here",
                                 "a long enough listing number two here"])

    assert calls["batch"] == 1                          # tried the batch once
    assert calls["single"] == 2                         # then fell back per row
    assert len(out) == 2
    assert all(r["price_thb"] == 9000 for r in out)


def test_batch_exception_triggers_per_row_fallback(monkeypatch):
    calls = {"single": 0}

    def raising_batch(texts):
        raise ValueError("model returned non-JSON")

    def single(text):
        calls["single"] += 1
        return {"is_offer": "offer"}

    monkeypatch.setattr(extract, "call_claude_batch", raising_batch)
    monkeypatch.setattr(extract, "call_claude", single)

    out = extract.extract_batch(["a long enough listing text alpha here",
                                 "a long enough listing text bravo here"])

    assert calls["single"] == 2
    assert [r["is_offer"] for r in out] == ["offer", "offer"]


# ---------------------------------------------------------------------------
# mixed batch: short rows skipped (no call), long rows batched, order kept
# ---------------------------------------------------------------------------

def test_mixed_short_and_long_only_calls_for_long(monkeypatch):
    seen = {}

    def fake_batch(texts):
        seen["n"] = len(texts)                          # how many actually went to the LLM
        return [{"is_offer": "offer"} for _ in texts]

    monkeypatch.setattr(extract, "call_claude_batch", fake_batch)
    monkeypatch.setattr(extract, "call_claude",
                        lambda t: (_ for _ in ()).throw(AssertionError("no fallback")))

    out = extract.extract_batch([
        "hi",                                           # short -> skipped
        "a long enough real listing in srithanu here",  # batched
        "x",                                            # short -> skipped
        "another long enough real listing here too",    # batched
    ])

    assert seen["n"] == 2                               # only the 2 long ones called
    assert len(out) == 4
    assert out[0]["discard_reason"] == "not_a_listing"  # short row 0
    assert out[1]["is_offer"] == "offer"               # long row 1
    assert out[2]["discard_reason"] == "not_a_listing"  # short row 2
    assert out[3]["is_offer"] == "offer"               # long row 3


# ---------------------------------------------------------------------------
# type coercion: typed SQL columns must never receive a value PG would reject
# ---------------------------------------------------------------------------

def test_coerces_messy_numbers_and_dates_and_bools(monkeypatch):
    monkeypatch.setattr(extract, "call_claude_batch", lambda texts: [{
        "price_thb": "15,000 THB",          # messy int string
        "bedrooms": "studio",               # not a number -> None
        "bathrooms": 2.0,                    # float -> int
        "deposit_thb": 10000,                # already int
        "electricity_rate_thb_per_unit": "8 per unit",  # numeric from string
        "available_until": "October",        # invalid date -> None
        "available_from": "October",         # TEXT field -> kept as-is
        "year_round": "yes",                 # bool from string
        "has_pool": "false",                 # bool from string
        "size_sqm": None,                    # unknown stays None
    }])
    [f] = extract.extract_batch(["a long enough listing text to parse here"])
    assert f["price_thb"] == 15000
    assert f["bedrooms"] is None
    assert f["bathrooms"] == 2
    assert f["deposit_thb"] == 10000
    assert f["electricity_rate_thb_per_unit"] == 8.0
    assert f["available_until"] is None          # invalid date dropped
    assert f["available_from"] == "October"      # text preserved
    assert f["year_round"] is True
    assert f["has_pool"] is False
    assert f["size_sqm"] is None


def test_valid_iso_date_is_kept(monkeypatch):
    monkeypatch.setattr(extract, "call_claude_batch",
                        lambda texts: [{"available_until": "2026-10-01"}])
    [f] = extract.extract_batch(["a long enough listing text to parse here"])
    assert f["available_until"] == "2026-10-01"


# ---------------------------------------------------------------------------
# single-row wrapper used by the CSV path + convenience
# ---------------------------------------------------------------------------

def test_extract_fields_wrapper_delegates_to_batch(monkeypatch):
    monkeypatch.setattr(extract, "call_claude_batch",
                        lambda texts: [{"is_offer": "offer", "bedrooms": 3}])
    fields = extract.extract_fields("a long enough single listing text here")
    assert fields["is_offer"] == "offer"
    assert fields["bedrooms"] == 3
    assert set(fields) == set(extract.MODEL_FIELDS)
