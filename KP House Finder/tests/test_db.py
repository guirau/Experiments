"""Unit tests for db.paginate — the guard against PostgREST's 1000-row read cap.

The live Supabase functions are exercised in the Task 8 end-to-end run, not here.
"""

import db


def test_paginate_stops_on_short_page():
    pages = [list(range(1000)), list(range(1000, 1500))]  # 1000 then 500 (< page_size)
    calls = []

    def fetch_page(offset, size):
        calls.append((offset, size))
        return pages.pop(0) if pages else []

    out = db.paginate(fetch_page, page_size=1000)
    assert len(out) == 1500
    assert calls == [(0, 1000), (1000, 1000)]  # stops after the short second page


def test_paginate_single_short_page():
    out = db.paginate(lambda offset, size: [1, 2, 3] if offset == 0 else [], page_size=1000)
    assert out == [1, 2, 3]


def test_paginate_exact_multiple_makes_trailing_empty_call():
    # exactly one full page then nothing: must make a second call to learn it's done
    pages = {0: list(range(1000)), 1000: []}
    calls = []

    def fetch_page(offset, size):
        calls.append(offset)
        return pages.get(offset, [])

    out = db.paginate(fetch_page, page_size=1000)
    assert len(out) == 1000
    assert calls == [0, 1000]
