import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { Listing } from "./types";
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

// Persist an edited price to listings_parsed (anon UPDATE policy required). null clears it.
export async function updatePrice(id: string, price: number | null): Promise<void> {
  const sb = client();
  const { error } = await sb.from("listings_parsed").update({ price_thb: price }).eq("id", id);
  if (error) throw error;
}
