import { createClient } from "@supabase/supabase-js";
import type { Listing } from "./types";
import { mergeLinks, type PostLink } from "./merge";

const PAGE = 1000;

function client() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) throw new Error("Missing NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY in web/.env.local");
  return createClient(url, key);
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

export async function fetchOffers(): Promise<Listing[]> {
  const sb = client();
  const offers = await pageAll<Listing>(async (from, to) => {
    const { data, error } = await sb.from("listings_parsed").select("*")
      .is("discard_reason", null).order("listed_at", { ascending: false }).range(from, to);
    if (error) throw error;
    return (data ?? []) as Listing[];
  });
  const posts = await pageAll<PostLink>(async (from, to) => {
    const { data, error } = await sb.from("fb_posts").select("id,link,url").range(from, to);
    if (error) throw error;
    return (data ?? []) as PostLink[];
  });
  return mergeLinks(offers, posts);
}
