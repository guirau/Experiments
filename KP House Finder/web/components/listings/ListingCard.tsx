"use client";
import { useState } from "react";
import type { Listing } from "@/lib/types";
import { areaName } from "@/lib/areas";
import { Badge } from "@/components/ui/Badge";
import { ListingDetails } from "./ListingDetails";

function postHref(l: Listing): string | null {
  if (l.url) return l.url;
  if (l.link) return l.link;
  if (l.id?.startsWith("http")) return l.id;
  return null;
}
const price = (l: Listing) => (l.price_thb != null ? `฿${l.price_thb.toLocaleString()}/mo` : "Ask");

export function ListingCard({ listing }: { listing: Listing }) {
  const [open, setOpen] = useState(false);
  const href = postHref(listing);
  const specs = [listing.property_type, listing.bedrooms != null ? `${listing.bedrooms}bd` : null, listing.bathrooms != null ? `${listing.bathrooms}ba` : null].filter(Boolean).join(" · ");
  return (
    <article className="rounded-2xl border p-4 transition-shadow hover:shadow-[var(--shadow)]" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      <div className="flex items-baseline justify-between">
        <span className="text-lg font-semibold">{price(listing)}</span>
        <span className="text-sm" style={{ color: "var(--muted)" }}>{areaName(listing.area_canonical)}</span>
      </div>
      <p className="mt-0.5 text-sm" style={{ color: "var(--muted)" }}>{specs || "—"}</p>
      <div className="mt-2 flex flex-wrap gap-1">
        {listing.year_round === true && <Badge tone="good">Year-round</Badge>}
        {listing.subletting_allowed === true && <Badge>Sublet OK</Badge>}
        {listing.season && listing.season !== "unknown" && <Badge tone="muted">{listing.season}</Badge>}
        {listing.parse_confidence && <Badge tone="muted">{listing.parse_confidence}</Badge>}
      </div>
      <div className="mt-3 flex items-center gap-3 text-sm">
        <button onClick={() => setOpen((o) => !o)} className="underline" style={{ color: "var(--accent)" }}>{open ? "Hide details" : "Details"}</button>
        {href && <a href={href} target="_blank" rel="noreferrer" className="underline" style={{ color: "var(--accent)" }}>View on FB ↗</a>}
      </div>
      {open && <ListingDetails listing={listing} />}
    </article>
  );
}
