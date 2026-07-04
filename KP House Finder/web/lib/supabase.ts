import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { Listing, Prospect, TrackerRow } from "./types";
import { mergeLinks, type PostLink } from "./merge";

const PAGE = 1000;

let cached: SupabaseClient | null = null;
function client() {
  if (cached) return cached;
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) throw new Error("Missing NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY in web/.env.local");
  cached = createClient(url, key);
  return cached;
}

async function pageAll<T>(fetchPage: (from: number, to: number) => Promise<T[]>): Promise<T[]> {
  const rows: T[] = [];
  let from = 0;
  for (;;) {
    const page = await fetchPage(from, from + PAGE - 1);
    rows.push(...page);
    if (page.length < PAGE) return rows;
    from += PAGE;
  }
}

// Loads rentals (discard_reason IS NULL) and for-sale listings; other discard reasons
// (wanted, not_a_listing, not_koh_phangan, not_long_term) stay excluded.
export async function fetchListings(): Promise<Listing[]> {
  const sb = client();
  const offers = await pageAll<Listing>(async (from, to) => {
    // id tiebreaker makes the sort a TOTAL order so .range() pagination can't
    // duplicate or skip rows that share a listed_at timestamp.
    const { data, error } = await sb.from("listings_parsed").select("*")
      .or("discard_reason.is.null,discard_reason.eq.for_sale")
      .order("listed_at", { ascending: false }).order("id", { ascending: true }).range(from, to);
    if (error) throw error;
    return (data ?? []) as Listing[];
  });
  const posts = await pageAll<PostLink>(async (from, to) => {
    const { data, error } = await sb.from("fb_posts").select("id,link,url")
      .order("id", { ascending: true }).range(from, to);
    if (error) throw error;
    return (data ?? []) as PostLink[];
  });
  // defensive dedup by id in case the backend ever returns a boundary repeat
  const seen = new Set<string>();
  const unique = offers.filter((o) => (seen.has(o.id) ? false : (seen.add(o.id), true)));
  return mergeLinks(unique, posts);
}

// Persist edited fields to listings_parsed (anon UPDATE policy required).
export async function updateListing(id: string, patch: Record<string, unknown>): Promise<void> {
  const sb = client();
  const { error } = await sb.from("listings_parsed").update(patch).eq("id", id);
  if (error) throw error;
}

export const updatePrice = (id: string, price: number | null) => updateListing(id, { price_thb: price });

// ---- Tracker (separate table) ----
export async function fetchTracker(): Promise<TrackerRow[]> {
  const sb = client();
  // order by created_at server-side; sort by sort_order client-side so the load stays
  // resilient even before the sort_order column migration (stable -> created_at fallback).
  const { data, error } = await sb.from("tracker").select("*").order("created_at", { ascending: true });
  if (error) throw error;
  const out = (data ?? []) as TrackerRow[];
  out.sort((a, b) => (a.sort_order ?? Infinity) - (b.sort_order ?? Infinity));
  return out;
}

// Columns we write on upsert. Excludes DB-managed columns (created_at) so new rows fall
// back to their defaults instead of being sent created_at: null in a mixed batch.
const TRACKER_COLS = ["id", "listing_url", "person_name", "price", "location_url",
  "visit_date", "contact", "notify_before", "notes", "crossed_off", "sort_order"] as const;

function toPayload(r: TrackerRow): Record<string, unknown> {
  const src = r as unknown as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const k of TRACKER_COLS) {
    const v = src[k];
    out[k] = v === "" ? null : v ?? null; // "" (empty input) and undefined -> null
  }
  return out;
}

// Bulk sync: delete removed rows, then upsert the current rows (keyed on id).
export async function saveTracker(rows: TrackerRow[], deletedIds: string[]): Promise<void> {
  const sb = client();
  if (deletedIds.length) {
    const { error } = await sb.from("tracker").delete().in("id", deletedIds);
    if (error) throw error;
  }
  if (rows.length) {
    const { error } = await sb.from("tracker").upsert(rows.map(toPayload), { onConflict: "id" });
    if (error) throw error;
  }
}

// ---- Prospects (Google Maps discovery / SerpApi enrichment) ----
export async function fetchProspects(): Promise<Prospect[]> {
  const sb = client();
  return pageAll<Prospect>(async (from, to) => {
    // best findings first; place_id tiebreaker keeps .range() pagination a total order.
    const { data, error } = await sb.from("prospects").select("*")
      .order("suitability_score", { ascending: false, nullsFirst: false })
      .order("place_id", { ascending: true }).range(from, to);
    if (error) throw error;
    return (data ?? []) as Prospect[];
  });
}

// Mark the places the user selected on the map as queued. src/enrich.py then enriches
// EXACTLY these (it only ever reads enrich_status = 'queued'), never anything else.
export async function queueForEnrichment(placeIds: string[]): Promise<void> {
  if (!placeIds.length) return;
  const sb = client();
  const { error } = await sb.from("prospects").update({ enrich_status: "queued" }).in("place_id", placeIds);
  if (error) throw error;
}

// place_ids that already have a tracker row (so cards can show "Tracked").
export async function fetchTrackedProspectIds(): Promise<Set<string>> {
  const sb = client();
  const { data, error } = await sb.from("tracker").select("prospect_id").not("prospect_id", "is", null);
  if (error) throw error;
  return new Set((data ?? []).map((r) => (r as { prospect_id: string }).prospect_id));
}

function mapsLink(p: Prospect): string {
  const q = encodeURIComponent(p.name ?? p.formatted_address ?? "");
  return `https://www.google.com/maps/search/?api=1&query=${q}&query_place_id=${p.place_id}`;
}

// Create a tracker row from a prospect card. Price is left blank on purpose — the tracker
// price is the LONG-TERM monthly rate you negotiate, not the nightly SerpApi figure (which
// goes into notes as a reference). Links back via prospect_id.
export async function createTrackerFromProspect(p: Prospect): Promise<void> {
  const sb = client();
  const ref = [
    p.google_rating != null ? `Google ${p.google_rating}★` : null,
    p.rate_per_night_thb != null ? `nightly ~฿${p.rate_per_night_thb.toLocaleString()}` : null,
  ].filter(Boolean).join(" · ");
  const { error } = await sb.from("tracker").insert({
    id: crypto.randomUUID(),
    listing_url: p.website ?? null,
    person_name: p.name ?? null,
    price: null,
    location_url: mapsLink(p),
    contact: p.phone ?? p.website ?? null,
    notes: ref ? `From Prospecting — ${ref}` : "From Prospecting",
    crossed_off: false,
    prospect_id: p.place_id,
  });
  if (error) throw error;
}
