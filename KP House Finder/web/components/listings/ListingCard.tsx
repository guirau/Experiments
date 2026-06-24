"use client";
import { useState } from "react";
import type { Listing } from "@/lib/types";
import { areaName } from "@/lib/areas";
import { listingKind } from "@/lib/filters";
import { Badge } from "@/components/ui/Badge";
import { ListingDetails } from "./ListingDetails";

function postHref(l: Listing): string | null {
  if (l.url) return l.url;
  if (l.link) return l.link;
  if (l.id?.startsWith("http")) return l.id;
  return null;
}
const price = (l: Listing) => {
  if (l.price_thb == null) return "Ask";
  return listingKind(l) === "sale" ? `฿${l.price_thb.toLocaleString()}` : `฿${l.price_thb.toLocaleString()}/mo`;
};

export interface ListingCardProps {
  listing: Listing;
  saved?: boolean;
  onSave?: (id: string) => void;
  onRemove?: (id: string) => void;
  onRestore?: (id: string) => void;
}

export function ListingCard({ listing, saved = false, onSave, onRemove, onRestore }: ListingCardProps) {
  const [open, setOpen] = useState(false);
  const href = postHref(listing);
  const specs = [listing.property_type, listing.bedrooms != null ? `${listing.bedrooms}bd` : null, listing.bathrooms != null ? `${listing.bathrooms}ba` : null].filter(Boolean).join(" · ");
  return (
    <article className="rounded-2xl border p-4 transition-shadow hover:shadow-[var(--shadow)]" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-lg font-semibold">{price(listing)}</span>
        <span className="text-sm" style={{ color: "var(--muted)" }}>{areaName(listing.area_canonical)}</span>
      </div>
      <p className="mt-0.5 text-sm" style={{ color: "var(--muted)" }}>{specs || "—"}</p>
      <div className="mt-2 flex flex-wrap gap-1">
        {listingKind(listing) === "sale" && <Badge>For sale</Badge>}
        {listing.year_round === true && <Badge tone="good">Year-round</Badge>}
        {listing.subletting_allowed === true && <Badge>Sublet OK</Badge>}
        {listing.season && listing.season !== "unknown" && <Badge tone="muted">{listing.season}</Badge>}
        {listing.parse_confidence && <Badge tone="muted">{listing.parse_confidence}</Badge>}
      </div>
      <div className="mt-3 flex items-center gap-3 text-sm">
        <button onClick={() => setOpen((o) => !o)} className="underline" style={{ color: "var(--accent)" }}>{open ? "Hide details" : "Details"}</button>
        {href && <a href={href} target="_blank" rel="noreferrer" className="underline" style={{ color: "var(--accent)" }}>View on FB ↗</a>}
        <span className="ml-auto flex items-center gap-2">
          {onSave && (
            <button onClick={() => onSave(listing.id)} aria-pressed={saved} title={saved ? "Remove from saved" : "Save to favourites"}
              className="rounded-full px-2 py-0.5" style={saved ? { color: "var(--accent)" } : { color: "var(--muted)" }}>
              {saved ? "★ Saved" : "☆ Save"}
            </button>
          )}
          {onRemove && (
            <button onClick={() => onRemove(listing.id)} aria-label="Remove listing" title="Not interested (remove)"
              className="rounded-full px-2 py-0.5" style={{ color: "var(--muted)" }}>✕</button>
          )}
          {onRestore && (
            <button onClick={() => onRestore(listing.id)} title="Restore to listings"
              className="rounded-full px-2 py-0.5 underline" style={{ color: "var(--accent)" }}>↩ Restore</button>
          )}
        </span>
      </div>
      {open && <ListingDetails listing={listing} />}
    </article>
  );
}
