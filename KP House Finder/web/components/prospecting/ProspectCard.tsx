"use client";
import { useState } from "react";
import type { Prospect } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";

function mapsUrl(p: Prospect): string {
  const q = encodeURIComponent(p.name ?? p.formatted_address ?? "");
  return `https://www.google.com/maps/search/?api=1&query=${q}&query_place_id=${p.place_id}`;
}

const priceLabel = (p: Prospect): string => {
  if (p.rate_per_night_thb == null) return "Not priced yet";
  const rooms = [p.bedrooms != null ? `${p.bedrooms}bd` : null, p.bathrooms != null ? `${p.bathrooms}ba` : null]
    .filter(Boolean).join(" · ");
  return `~฿${p.rate_per_night_thb.toLocaleString()}/night${rooms ? ` · ${rooms}` : ""}`;
};

export interface ProspectCardProps {
  prospect: Prospect;
  tracked: boolean;
  onAddToTracker: (p: Prospect) => Promise<void>;
}

export function ProspectCard({ prospect: p, tracked, onAddToTracker }: ProspectCardProps) {
  const [saving, setSaving] = useState(false);
  const score = p.suitability_score;

  const addToTracker = async () => {
    if (tracked || saving) return;
    setSaving(true);
    try { await onAddToTracker(p); }
    catch { window.alert("Couldn't add to tracker — is the prospect_id column added? (see tracker.sql)"); }
    finally { setSaving(false); }
  };

  return (
    <article className="rounded-2xl border p-4" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-base font-semibold">{p.name ?? "Unnamed place"}</span>
        {p.google_rating != null && (
          <span className="shrink-0 text-sm" style={{ color: "var(--muted)" }}>
            {p.google_rating}★{p.user_ratings_total != null ? ` (${p.user_ratings_total})` : ""}
          </span>
        )}
      </div>
      {p.formatted_address && <p className="mt-0.5 text-sm" style={{ color: "var(--muted)" }}>{p.formatted_address}</p>}

      <div className="mt-2 flex flex-wrap gap-1">
        {score != null
          ? <Badge tone={score >= 7 ? "good" : undefined}>Fit {score}/10</Badge>
          : <Badge tone="muted">unscored</Badge>}
        {p.property_type && <Badge tone="muted">{p.property_type.replace(/_/g, " ")}</Badge>}
        {p.enrich_status === "queued" && <Badge>queued</Badge>}
        {p.enrich_status === "enriched" && <Badge tone="good">priced</Badge>}
        {p.enrich_status === "error" && <Badge tone="muted">no price found</Badge>}
      </div>

      {p.suitability_reason && (
        <p className="mt-2 text-sm" style={{ color: "var(--ink)" }}>{p.suitability_reason}</p>
      )}

      <p className="mt-2 text-sm font-medium">{priceLabel(p)}</p>
      {p.ota_offers && p.ota_offers.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs" style={{ color: "var(--muted)" }}>
          {p.ota_offers.map((o, i) => (
            <span key={i}>
              {o.link ? <a href={o.link} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>{o.source ?? "offer"}</a> : (o.source ?? "offer")}
              {o.price != null ? ` ฿${o.price.toLocaleString()}` : ""}
            </span>
          ))}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <a href={mapsUrl(p)} target="_blank" rel="noreferrer" className="card-btn">Map ↗</a>
        {p.website && <a href={p.website} target="_blank" rel="noreferrer" className="card-btn">Website ↗</a>}
        {p.phone && <a href={`tel:${p.phone.replace(/\s/g, "")}`} className="card-btn">{p.phone}</a>}
        <button onClick={addToTracker} disabled={tracked || saving}
          title={tracked ? "Already in the tracker" : "Add to the tracker"}
          className={`card-btn${tracked ? " card-btn-good" : ""}`}>
          {tracked ? "✓ Tracked" : saving ? "Adding…" : "+ Add to tracker"}
        </button>
      </div>
    </article>
  );
}
