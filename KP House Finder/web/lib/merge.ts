import type { Listing } from "./types";

export interface PostLink { id: string; link: string | null; url: string | null; }

export function mergeLinks(offers: Listing[], posts: PostLink[]): Listing[] {
  const byId = new Map(posts.map((p) => [p.id, p]));
  return offers.map((o) => {
    const p = byId.get(o.id);
    return p ? { ...o, link: p.link, url: p.url } : o;
  });
}
