export type BoolState = "yes" | "no" | "unknown";

export interface Listing {
  // control / meta
  id: string;
  source_table: string | null;
  source: string | null;
  listed_at: string | null;
  raw_text: string | null;
  parser_version: string | null;
  parsed_at: string | null;
  discard_reason: string | null;
  // classification
  is_offer: string | null;
  post_language: string | null;
  parse_confidence: string | null;
  multi_listing: boolean | null;
  // tier 1
  price_thb: number | null;
  price_low_thb: number | null;
  price_high_thb: number | null;
  price_period: string | null;
  season: string | null;
  bedrooms: number | null;
  bathrooms: number | null;
  property_type: string | null;
  area_raw: string | null;
  area_canonical: string | null;
  // tier 2
  min_stay_months: number | null;
  available_from: string | null;
  available_until: string | null;
  year_round: boolean | null;
  subletting_allowed: boolean | null;
  // tier 3
  deposit_thb: number | null;
  electricity_rate_thb_per_unit: number | null;
  water_included: boolean | null;
  internet_included: boolean | null;
  // tier 4
  has_aircon: boolean | null;
  has_wifi: boolean | null;
  furnished: boolean | null;
  has_kitchen: boolean | null;
  has_pool: boolean | null;
  has_parking: boolean | null;
  pet_friendly: boolean | null;
  sea_view: boolean | null;
  has_workspace: boolean | null;
  has_terrace: boolean | null;
  near_road: boolean | null;
  near_construction: string | null;
  furnishings_list: string | null;
  // tier 5
  contact_raw: string | null;
  contact_phone: string | null;
  size_sqm: number | null;
  // joined from fb_posts
  link: string | null;
  url: string | null;
}

export type SortKey = "newest" | "recent" | "price_asc" | "price_desc" | "confidence";

// A row in the personal "Tracker" table (separate Supabase table, manual entry).
export interface TrackerRow {
  id: string;
  listing_url: string | null;
  person_name: string | null;
  price: number | null;
  location_url: string | null;
  visit_date: string | null;    // YYYY-MM-DD
  contact: string | null;
  notify_before: string | null; // YYYY-MM-DD
  notes: string | null;
  crossed_off: boolean | null;
  sort_order: number | null;
}

// One OTA price offer (Booking/Agoda/Airbnb…) stored in prospects.ota_offers (jsonb).
export interface OtaOffer {
  source: string | null;
  price: number | null;
  link: string | null;
}

// A Koh Phangan accommodation business discovered on Google Maps (prospects table).
// Populated by src/discover.py; price fields filled by src/enrich.py after the user
// queues it from the map.
export interface Prospect {
  place_id: string;
  name: string | null;
  formatted_address: string | null;
  lat: number | null;
  lng: number | null;
  phone: string | null;
  website: string | null;
  google_rating: number | null;
  user_ratings_total: number | null;
  property_type: string | null;
  source: string | null;
  suitability_score: number | null;
  suitability_reason: string | null;
  enrich_status: string | null; // discovered | queued | enriched | error
  rate_per_night_thb: number | null;
  bedrooms: number | null;
  bathrooms: number | null;
  ota_offers: OtaOffer[] | null;
  created_at: string | null;
  enriched_at: string | null;
}

export interface FilterState {
  listingType: string[]; // "rent" | "sale"; empty = both
  parsedWithin: string;  // "" = any; else number of days ("1"|"3"|"7"|"30")
  priceMin: number | null;
  priceMax: number | null;
  areas: string[];
  propertyTypes: string[];
  bedroomsMin: number | null;
  bedroomsMax: number | null;
  bathroomsMin: number | null;
  yearRound: BoolState[];
  seasons: string[];
  minStayMax: number | null;
  subletting: BoolState[];
  depositMax: number | null;
  waterIncluded: BoolState[];
  internetIncluded: BoolState[];
  amenities: string[];
  confidences: string[];
  languages: string[];
  sort: SortKey;
}
