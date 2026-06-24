import type { Listing, FilterState, BoolState, SortKey } from "./types";

export function defaultFilters(): FilterState {
  return { priceMin: null, priceMax: null, areas: [], propertyTypes: [], bedroomsMin: null,
    bathroomsMin: null, yearRound: [], seasons: [], minStayMax: null, subletting: [],
    depositMax: null, waterIncluded: [], internetIncluded: [], amenities: [],
    confidences: [], languages: [], sort: "newest" };
}

const geOrNull = (v: number | null, min: number | null) => min == null || v == null || v >= min;
const leOrNull = (v: number | null, max: number | null) => max == null || v == null || v <= max;
const inSet = (v: string | null, set: string[]) => set.length === 0 || (v != null && set.includes(v));
const boolState = (v: boolean | null): BoolState => (v === true ? "yes" : v === false ? "no" : "unknown");
// empty set = no constraint; otherwise the listing's state (yes/no/unknown) must be selected
const boolSet = (v: boolean | null, sel: BoolState[]) => sel.length === 0 || sel.includes(boolState(v));

export function applyFilters(rows: Listing[], s: FilterState): Listing[] {
  return rows.filter((r) =>
    leOrNull(r.price_thb, s.priceMax) && geOrNull(r.price_thb, s.priceMin) &&
    inSet(r.area_canonical, s.areas) && inSet(r.property_type, s.propertyTypes) &&
    geOrNull(r.bedrooms, s.bedroomsMin) && geOrNull(r.bathrooms, s.bathroomsMin) &&
    boolSet(r.year_round, s.yearRound) && inSet(r.season, s.seasons) &&
    leOrNull(r.min_stay_months, s.minStayMax) && boolSet(r.subletting_allowed, s.subletting) &&
    leOrNull(r.deposit_thb, s.depositMax) && boolSet(r.water_included, s.waterIncluded) &&
    boolSet(r.internet_included, s.internetIncluded) &&
    s.amenities.every((a) => (r as unknown as Record<string, unknown>)[a] === true) &&
    inSet(r.parse_confidence, s.confidences) && inSet(r.post_language, s.languages)
  );
}

const CONF_RANK: Record<string, number> = { high: 3, medium: 2, low: 1 };
const nlast = (v: number | null) => (v == null ? Infinity : v);

export function sortListings(rows: Listing[], sort: SortKey): Listing[] {
  const out = [...rows];
  if (sort === "price_asc") out.sort((a, b) => nlast(a.price_thb) - nlast(b.price_thb));
  else if (sort === "price_desc") out.sort((a, b) => (b.price_thb ?? -Infinity) - (a.price_thb ?? -Infinity));
  else if (sort === "confidence") out.sort((a, b) => (CONF_RANK[b.parse_confidence ?? ""] ?? 0) - (CONF_RANK[a.parse_confidence ?? ""] ?? 0));
  else out.sort((a, b) => (b.listed_at ?? "").localeCompare(a.listed_at ?? ""));
  return out;
}

const CSV = (a: string[]) => a.join(",");
const unCSV = (s: string | null) => (s ? s.split(",").filter(Boolean) : []);
const numOrNull = (s: string | null) => {
  const n = Number(s);
  return s != null && s !== "" && !Number.isNaN(n) ? n : null;
};
const SORT_KEYS: SortKey[] = ["newest", "price_asc", "price_desc", "confidence"];

export function filtersToParams(s: FilterState): URLSearchParams {
  const p = new URLSearchParams();
  const d = defaultFilters();
  const setNum = (k: string, v: number | null) => v != null && p.set(k, String(v));
  const setArr = (k: string, v: string[]) => v.length && p.set(k, CSV(v));
  setNum("priceMin", s.priceMin); setNum("priceMax", s.priceMax);
  setArr("areas", s.areas); setArr("types", s.propertyTypes);
  setNum("bedsMin", s.bedroomsMin); setNum("bathsMin", s.bathroomsMin);
  setArr("yearRound", s.yearRound); setArr("seasons", s.seasons);
  setNum("minStayMax", s.minStayMax); setArr("sublet", s.subletting);
  setNum("depositMax", s.depositMax); setArr("water", s.waterIncluded);
  setArr("internet", s.internetIncluded); setArr("amenities", s.amenities);
  setArr("conf", s.confidences); setArr("lang", s.languages);
  if (s.sort !== d.sort) p.set("sort", s.sort);
  return p;
}

const BOOL_STATES = ["yes", "no", "unknown"];
const unBool = (s: string | null): BoolState[] => unCSV(s).filter((x): x is BoolState => BOOL_STATES.includes(x));

export function paramsToFilters(p: URLSearchParams): FilterState {
  const d = defaultFilters();
  return { ...d,
    priceMin: numOrNull(p.get("priceMin")), priceMax: numOrNull(p.get("priceMax")),
    areas: unCSV(p.get("areas")), propertyTypes: unCSV(p.get("types")),
    bedroomsMin: numOrNull(p.get("bedsMin")), bathroomsMin: numOrNull(p.get("bathsMin")),
    yearRound: unBool(p.get("yearRound")), seasons: unCSV(p.get("seasons")),
    minStayMax: numOrNull(p.get("minStayMax")), subletting: unBool(p.get("sublet")),
    depositMax: numOrNull(p.get("depositMax")), waterIncluded: unBool(p.get("water")),
    internetIncluded: unBool(p.get("internet")), amenities: unCSV(p.get("amenities")),
    confidences: unCSV(p.get("conf")), languages: unCSV(p.get("lang")),
    sort: SORT_KEYS.includes(p.get("sort") as SortKey) ? (p.get("sort") as SortKey) : d.sort };
}
