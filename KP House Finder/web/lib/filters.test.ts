import { describe, it, expect } from "vitest";
import { defaultFilters, applyFilters, sortListings, filtersToParams, paramsToFilters } from "./filters";
import type { Listing, FilterState } from "./types";

function L(over: Partial<Listing>): Listing {
  return { id: "x", source_table: null, source: null, listed_at: null, raw_text: null,
    parser_version: null, discard_reason: null, is_offer: null, post_language: null,
    parse_confidence: null, multi_listing: null, price_thb: null, price_low_thb: null,
    price_high_thb: null, price_period: null, season: null, bedrooms: null, bathrooms: null,
    property_type: null, area_raw: null, area_canonical: null, min_stay_months: null,
    available_from: null, available_until: null, year_round: null, subletting_allowed: null,
    deposit_thb: null, electricity_rate_thb_per_unit: null, water_included: null,
    internet_included: null, has_aircon: null, has_wifi: null, furnished: null,
    has_kitchen: null, has_pool: null, has_parking: null, pet_friendly: null, sea_view: null,
    has_workspace: null, has_terrace: null, near_road: null, near_construction: null,
    furnishings_list: null, contact_raw: null, contact_phone: null, size_sqm: null,
    link: null, url: null, ...over };
}
const f = (over: Partial<FilterState> = {}): FilterState => ({ ...defaultFilters(), ...over });

describe("applyFilters", () => {
  it("returns all rows with default filters", () => {
    const rows = [L({ price_thb: 5000 }), L({ price_thb: 9000 })];
    expect(applyFilters(rows, f())).toHaveLength(2);
  });

  it("price range keeps in-range and keeps null price (unknown not hidden)", () => {
    const rows = [L({ price_thb: 5000 }), L({ price_thb: 20000 }), L({ price_thb: null })];
    const out = applyFilters(rows, f({ priceMax: 10000 }));
    expect(out.map(r => r.price_thb).sort()).toEqual([5000, null].sort());
  });

  it("area multiselect constrains to selected slugs", () => {
    const rows = [L({ area_canonical: "srithanu" }), L({ area_canonical: "ban_tai" })];
    expect(applyFilters(rows, f({ areas: ["srithanu"] }))).toHaveLength(1);
  });

  it("tri-state yes requires true and hides null+false", () => {
    const rows = [L({ year_round: true }), L({ year_round: false }), L({ year_round: null })];
    expect(applyFilters(rows, f({ yearRound: "yes" })).map(r => r.year_round)).toEqual([true]);
  });

  it("tri-state no requires false", () => {
    const rows = [L({ subletting_allowed: true }), L({ subletting_allowed: false })];
    expect(applyFilters(rows, f({ subletting: "no" })).map(r => r.subletting_allowed)).toEqual([false]);
  });

  it("bedroomsMin keeps >= and keeps null", () => {
    const rows = [L({ bedrooms: 1 }), L({ bedrooms: 3 }), L({ bedrooms: null })];
    const out = applyFilters(rows, f({ bedroomsMin: 2 })).map(r => r.bedrooms);
    expect(out.sort()).toEqual([3, null].sort());
  });

  it("minStayMax keeps <= and keeps null", () => {
    const rows = [L({ min_stay_months: 2 }), L({ min_stay_months: 12 }), L({ min_stay_months: null })];
    const out = applyFilters(rows, f({ minStayMax: 6 })).map(r => r.min_stay_months);
    expect(out.sort()).toEqual([2, null].sort());
  });

  it("amenities require each field true", () => {
    const rows = [L({ has_pool: true, has_wifi: true }), L({ has_pool: true, has_wifi: null })];
    expect(applyFilters(rows, f({ amenities: ["has_pool", "has_wifi"] }))).toHaveLength(1);
  });

  it("combines filters with AND", () => {
    const rows = [
      L({ price_thb: 8000, area_canonical: "srithanu", year_round: true }),
      L({ price_thb: 8000, area_canonical: "ban_tai", year_round: true }),
    ];
    expect(applyFilters(rows, f({ priceMax: 10000, areas: ["srithanu"], yearRound: "yes" }))).toHaveLength(1);
  });
});

describe("sortListings", () => {
  it("price_asc puts nulls last", () => {
    const rows = [L({ price_thb: null }), L({ price_thb: 9000 }), L({ price_thb: 5000 })];
    expect(sortListings(rows, "price_asc").map(r => r.price_thb)).toEqual([5000, 9000, null]);
  });
  it("newest sorts listed_at desc", () => {
    const rows = [L({ listed_at: "2026-01-01" }), L({ listed_at: "2026-06-01" })];
    expect(sortListings(rows, "newest").map(r => r.listed_at)).toEqual(["2026-06-01", "2026-01-01"]);
  });
  it("confidence orders high > medium > low", () => {
    const rows = [L({ parse_confidence: "low" }), L({ parse_confidence: "high" }), L({ parse_confidence: "medium" })];
    expect(sortListings(rows, "confidence").map(r => r.parse_confidence)).toEqual(["high", "medium", "low"]);
  });
});

describe("URL codec round-trip", () => {
  it("survives filters -> params -> filters", () => {
    const state = f({ priceMin: 5000, priceMax: 15000, areas: ["srithanu", "ban_tai"],
      yearRound: "yes", subletting: "no", bedroomsMin: 2, amenities: ["has_pool"],
      seasons: ["full_year"], sort: "price_asc" });
    expect(paramsToFilters(filtersToParams(state))).toEqual(state);
  });
  it("empty params yields defaults", () => {
    expect(paramsToFilters(new URLSearchParams())).toEqual(defaultFilters());
  });
});
