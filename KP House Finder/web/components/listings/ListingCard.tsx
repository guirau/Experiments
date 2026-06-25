"use client";
import { useState } from "react";
import type { Listing } from "@/lib/types";
import { areaName } from "@/lib/areas";
import { listingKind } from "@/lib/filters";
import { Badge } from "@/components/ui/Badge";

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
  contacted?: boolean;
  onSave?: (id: string) => void;
  onContacted?: (id: string) => void;
  onRemove?: (id: string) => void;
  onRestore?: (id: string) => void;
  onEditPrice?: (id: string, price: number | null) => Promise<void> | void;
  onOpen?: (listing: Listing) => void;
}

export function ListingCard({ listing, saved = false, contacted = false, onSave, onContacted, onRemove, onRestore, onEditPrice, onOpen }: ListingCardProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const href = postHref(listing);
  const specs = [listing.property_type, listing.bedrooms != null ? `${listing.bedrooms}bd` : null, listing.bathrooms != null ? `${listing.bathrooms}ba` : null].filter(Boolean).join(" · ");
  const stop = (e: React.MouseEvent) => e.stopPropagation();

  const startEdit = () => {
    if (!onEditPrice) return;
    setDraft(listing.price_thb != null ? String(listing.price_thb) : "");
    setEditing(true);
  };
  const commit = async () => {
    const trimmed = draft.trim();
    const next = trimmed === "" ? null : Math.round(Number(trimmed));
    if (trimmed !== "" && (!Number.isFinite(next) || (next as number) < 0)) { setEditing(false); return; }
    if (next === listing.price_thb) { setEditing(false); return; }
    setSaving(true);
    try { await onEditPrice!(listing.id, next); setEditing(false); }
    catch { window.alert("Couldn't save the price — please try again."); }
    finally { setSaving(false); }
  };

  return (
    <article onClick={() => { if (href) window.open(href, "_blank", "noopener,noreferrer"); }}
      title={href ? "Open the Facebook post" : undefined}
      style={{ borderColor: "var(--line)", background: "var(--surface)", cursor: href ? "pointer" : "default" }}
      className="rounded-2xl border p-4 transition-shadow hover:shadow-[var(--shadow)]">
      <div className="flex items-baseline justify-between gap-2">
        {editing ? (
          <input type="number" inputMode="numeric" autoFocus aria-label="Edit price (THB)" disabled={saving}
            value={draft} onClick={stop} onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") commit(); else if (e.key === "Escape") setEditing(false); }}
            onBlur={() => !saving && setEditing(false)}
            className="w-28 rounded-lg border px-2 py-0.5 text-lg font-semibold" style={{ borderColor: "var(--accent)" }} />
        ) : (
          <span className="text-lg font-semibold" onClick={stop} onDoubleClick={startEdit}
            title={onEditPrice ? "Double-click to edit price" : undefined}
            style={{ cursor: onEditPrice ? "text" : "inherit" }}>
            {price(listing)}{saving ? " …" : ""}
          </span>
        )}
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

      <div className="mt-3 flex flex-col gap-2 text-sm">
        <div className="flex items-center gap-4">
          {onOpen && <button onClick={(e) => { stop(e); onOpen(listing); }} className="underline" style={{ color: "var(--accent)" }}>Details</button>}
          {href && <a href={href} onClick={stop} target="_blank" rel="noreferrer" className="underline" style={{ color: "var(--accent)" }}>View on FB ↗</a>}
        </div>
        <div className="flex items-center gap-2">
          {onSave && (
            <button onClick={(e) => { stop(e); onSave(listing.id); }} aria-pressed={saved} title={saved ? "Remove from saved" : "Save to favourites"}
              className="rounded-full px-2 py-0.5" style={saved ? { color: "var(--accent)" } : { color: "var(--muted)" }}>
              {saved ? "★ Saved" : "☆ Save"}
            </button>
          )}
          {onContacted && (
            <button onClick={(e) => { stop(e); onContacted(listing.id); }} aria-pressed={contacted} title={contacted ? "Mark as not contacted" : "Mark as contacted"}
              className="rounded-full px-2 py-0.5" style={contacted ? { color: "var(--good)" } : { color: "var(--muted)" }}>
              {contacted ? "✓ Contacted" : "✆ Contacted"}
            </button>
          )}
          {onRemove && (
            <button onClick={(e) => { stop(e); onRemove(listing.id); }} aria-label="Remove listing" title="Not interested (remove)"
              className="rounded-full px-2 py-0.5" style={{ color: "var(--muted)" }}>✕ Remove</button>
          )}
          {onRestore && (
            <button onClick={(e) => { stop(e); onRestore(listing.id); }} title="Restore to listings"
              className="rounded-full px-2 py-0.5 underline" style={{ color: "var(--accent)" }}>↩ Restore</button>
          )}
        </div>
      </div>
    </article>
  );
}
