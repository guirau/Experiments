"use client";
import { useEffect, useState } from "react";
import type { Listing } from "@/lib/types";
import { listingKind } from "@/lib/filters";
import { areaName } from "@/lib/areas";
import { AREA_ENUM, PROPERTY_TYPES, SEASONS, CONFIDENCES, LANGUAGES } from "@/lib/constants";

// Editable, filterable fields (typed inputs so the typed DB columns can't get garbage).
const NUM_FIELDS: [string, string][] = [
  ["price_thb", "Price (฿)"], ["bedrooms", "Bedrooms"], ["bathrooms", "Bathrooms"],
  ["min_stay_months", "Min stay (months)"], ["deposit_thb", "Deposit (฿)"],
];
const ENUM_FIELDS: [string, string, string[]][] = [
  ["area_canonical", "Area", AREA_ENUM], ["property_type", "Property type", PROPERTY_TYPES],
  ["season", "Season", SEASONS], ["parse_confidence", "Confidence", CONFIDENCES],
  ["post_language", "Language", LANGUAGES],
];
const TRI_FIELDS: [string, string][] = [
  ["year_round", "Year-round"], ["subletting_allowed", "Subletting allowed"],
  ["water_included", "Water included"], ["internet_included", "Internet included"],
  ["has_aircon", "Aircon"], ["has_wifi", "Wifi"], ["has_pool", "Pool"], ["has_kitchen", "Kitchen"],
  ["has_parking", "Parking"], ["furnished", "Furnished"], ["pet_friendly", "Pet friendly"],
  ["sea_view", "Sea view"], ["has_workspace", "Workspace"], ["has_terrace", "Terrace"],
];

const get = (l: Listing, k: string) => (l as unknown as Record<string, unknown>)[k];

function initDraft(l: Listing): Record<string, string> {
  const d: Record<string, string> = { __listingType: listingKind(l) };
  for (const [k] of NUM_FIELDS) { const v = get(l, k); d[k] = v != null ? String(v) : ""; }
  for (const [k] of ENUM_FIELDS) d[k] = (get(l, k) as string) ?? "";
  for (const [k] of TRI_FIELDS) { const v = get(l, k); d[k] = v === true ? "yes" : v === false ? "no" : "unknown"; }
  return d;
}

function buildPatch(d: Record<string, string>): Record<string, unknown> {
  const p: Record<string, unknown> = {};
  for (const [k] of NUM_FIELDS) {
    const t = d[k].trim();
    const n = Math.round(Number(t));
    p[k] = t === "" || !Number.isFinite(n) ? null : n;
  }
  for (const [k] of ENUM_FIELDS) p[k] = d[k] || null;
  for (const [k] of TRI_FIELDS) p[k] = d[k] === "yes" ? true : d[k] === "no" ? false : null;
  p.discard_reason = d.__listingType === "sale" ? "for_sale" : null;
  return p;
}

const sel = "rounded-lg border px-2 py-1 text-sm";
const triOpts = [["unknown", "Unknown"], ["yes", "Yes"], ["no", "No"]];

export function EditModal({ listing, onClose, onSave }:
  { listing: Listing; onClose: () => void; onSave: (id: string, patch: Record<string, unknown>) => Promise<void> }) {
  const [draft, setDraft] = useState<Record<string, string>>(() => initDraft(listing));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: string, v: string) => setDraft((d) => ({ ...d, [k]: v }));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const save = async () => {
    setSaving(true); setError(null);
    try { await onSave(listing.id, buildPatch(draft)); onClose(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setSaving(false); }
  };

  const href = listing.url || listing.link || (listing.id.startsWith("http") ? listing.id : null);

  return (
    <div className="fixed inset-0 z-30 flex items-start justify-center overflow-y-auto p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/40" />
      <div role="dialog" aria-modal="true" aria-label="Edit listing" onClick={(e) => e.stopPropagation()}
        className="relative my-8 w-full max-w-2xl rounded-2xl border p-5 shadow-[var(--shadow)]"
        style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
        <div className="mb-3 flex items-start justify-between gap-3">
          <h2 className="text-lg font-semibold">Edit listing — {areaName(listing.area_canonical)}</h2>
          <button onClick={onClose} aria-label="Close" className="text-xl leading-none" style={{ color: "var(--muted)" }}>×</button>
        </div>

        {listing.raw_text && (
          <div className="mb-3 max-h-32 overflow-y-auto rounded-lg border p-2 text-xs whitespace-pre-wrap"
            style={{ borderColor: "var(--line)", color: "var(--muted)" }}>{listing.raw_text}</div>
        )}
        <div className="mb-4 flex flex-wrap gap-3 text-xs" style={{ color: "var(--muted)" }}>
          {listing.contact_raw && <span>Contact: {listing.contact_raw}</span>}
          {href && <a href={href} target="_blank" rel="noreferrer" className="underline" style={{ color: "var(--accent)" }}>View on FB ↗</a>}
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
          <label className="flex flex-col gap-1 text-xs font-medium">Listing
            <select className={sel} style={{ borderColor: "var(--line)" }} value={draft.__listingType} onChange={(e) => set("__listingType", e.target.value)}>
              <option value="rent">For rent</option><option value="sale">For sale</option>
            </select>
          </label>
          {NUM_FIELDS.map(([k, label]) => (
            <label key={k} className="flex flex-col gap-1 text-xs font-medium">{label}
              <input type="number" inputMode="numeric" className={sel} style={{ borderColor: "var(--line)" }}
                value={draft[k]} onChange={(e) => set(k, e.target.value)} />
            </label>
          ))}
          {ENUM_FIELDS.map(([k, label, opts]) => (
            <label key={k} className="flex flex-col gap-1 text-xs font-medium">{label}
              <select className={sel} style={{ borderColor: "var(--line)" }} value={draft[k]} onChange={(e) => set(k, e.target.value)}>
                <option value="">—</option>
                {opts.map((o) => <option key={o} value={o}>{k === "area_canonical" ? areaName(o) : o}</option>)}
              </select>
            </label>
          ))}
          {TRI_FIELDS.map(([k, label]) => (
            <label key={k} className="flex flex-col gap-1 text-xs font-medium">{label}
              <select className={sel} style={{ borderColor: "var(--line)" }} value={draft[k]} onChange={(e) => set(k, e.target.value)}>
                {triOpts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
          ))}
        </div>

        {error && <p className="mt-3 text-sm" style={{ color: "var(--warn)" }}>Couldn’t save: {error}</p>}
        <div className="mt-5 flex items-center justify-end gap-3">
          <button onClick={onClose} className="rounded-full px-4 py-2 text-sm" style={{ color: "var(--muted)" }}>Cancel</button>
          <button onClick={save} disabled={saving} className="rounded-full px-4 py-2 text-sm font-medium"
            style={{ background: "var(--accent)", color: "var(--accent-ink)", opacity: saving ? 0.6 : 1 }}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
