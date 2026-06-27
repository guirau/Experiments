"use client";
import { useRef, useState } from "react";
import type { useTracker } from "@/hooks/useTracker";

const inputCls = "w-full rounded border px-2 py-1 text-xs";
const todayStr = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

function UrlCell({ value, placeholder, onChange }: { value: string; placeholder: string; onChange: (v: string) => void }) {
  return (
    <div className="flex items-center gap-1">
      <input className={inputCls} style={{ borderColor: "var(--line)" }} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
      {/^https?:\/\//.test(value) && (
        <a href={value} target="_blank" rel="noreferrer" title="Open" className="shrink-0 text-sm" style={{ color: "var(--accent)" }}>↗</a>
      )}
    </div>
  );
}

export function TrackerTable({ tracker }: { tracker: ReturnType<typeof useTracker> }) {
  const { rows, loading, error, saving, dirty, addRow, updateRow, removeRow, toggleCrossed, moveRow, save } = tracker;
  const today = todayStr();
  const dragId = useRef<string | null>(null);
  const [dragOverId, setDragOverId] = useState<string | null>(null);

  if (loading) return <p className="p-6 text-sm" style={{ color: "var(--muted)" }}>Loading tracker…</p>;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <button onClick={addRow} className="card-btn">+ Add row</button>
        <button onClick={save} disabled={saving || !dirty} className={`card-btn${dirty ? " card-btn-on" : ""}`}>{saving ? "Saving…" : "Save"}</button>
        {dirty && !error && <span className="text-xs" style={{ color: "var(--muted)" }}>unsaved changes</span>}
        {error && <span className="text-xs" style={{ color: "var(--warn)" }}>Couldn’t save: {error} — is the `tracker` table created?</span>}
      </div>

      <div className="overflow-x-auto rounded-2xl border" style={{ borderColor: "var(--line)" }}>
        <table className="w-full border-collapse text-xs" style={{ minWidth: "980px" }}>
          <thead>
            <tr style={{ background: "var(--bg)" }}>
              {["Listing", "Name", "Price", "Location", "Date", "Contact", "Notify before", "Notes", ""].map((c, i) => (
                <th key={i} className="px-2 py-2 text-left font-semibold" style={{ color: "var(--muted)" }}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr><td colSpan={9} className="px-2 py-8 text-center" style={{ color: "var(--muted)" }}>No rows yet — click “+ Add row”.</td></tr>
            )}
            {rows.map((r) => {
              const overdue = !!r.notify_before && r.notify_before <= today;
              const crossed = !!r.crossed_off;
              const dropTarget = dragOverId === r.id;
              return (
                <tr key={r.id}
                  onDragOver={(e) => { e.preventDefault(); if (dragOverId !== r.id) setDragOverId(r.id); }}
                  onDragLeave={() => setDragOverId((cur) => (cur === r.id ? null : cur))}
                  onDrop={(e) => { e.preventDefault(); if (dragId.current) moveRow(dragId.current, r.id); dragId.current = null; setDragOverId(null); }}
                  className={`align-top${crossed ? " tracker-crossed" : ""}`}
                  style={{ borderTop: `${dropTarget ? 2 : 1}px solid ${dropTarget ? "var(--accent)" : "var(--line)"}` }}>
                  <td className="p-1" style={{ minWidth: 170 }}><UrlCell value={r.listing_url ?? ""} placeholder="https://…" onChange={(v) => updateRow(r.id, { listing_url: v })} /></td>
                  <td className="p-1" style={{ minWidth: 120 }}><input className={inputCls} style={{ borderColor: "var(--line)" }} value={r.person_name ?? ""} onChange={(e) => updateRow(r.id, { person_name: e.target.value })} /></td>
                  <td className="p-1" style={{ minWidth: 100 }}><input type="number" inputMode="numeric" className={inputCls} style={{ borderColor: "var(--line)" }} placeholder="฿" value={r.price != null ? String(r.price) : ""} onChange={(e) => updateRow(r.id, { price: e.target.value === "" ? null : Math.round(Number(e.target.value)) })} /></td>
                  <td className="p-1" style={{ minWidth: 170 }}><UrlCell value={r.location_url ?? ""} placeholder="Google Maps link" onChange={(v) => updateRow(r.id, { location_url: v })} /></td>
                  <td className="p-1" style={{ minWidth: 140 }}><input type="date" className={inputCls} style={{ borderColor: "var(--line)" }} value={r.visit_date ?? ""} onChange={(e) => updateRow(r.id, { visit_date: e.target.value })} /></td>
                  <td className="p-1" style={{ minWidth: 130 }}><input className={inputCls} style={{ borderColor: "var(--line)" }} value={r.contact ?? ""} onChange={(e) => updateRow(r.id, { contact: e.target.value })} /></td>
                  <td className="p-1" style={{ minWidth: 140 }}>
                    <input type="date" className={inputCls} value={r.notify_before ?? ""} title={overdue ? "Overdue" : undefined}
                      onChange={(e) => updateRow(r.id, { notify_before: e.target.value })}
                      style={overdue ? { borderColor: "var(--warn)", background: "color-mix(in oklch, var(--warn) 20%, transparent)" } : { borderColor: "var(--line)" }} />
                  </td>
                  <td className="p-1" style={{ minWidth: 220 }}><input className={inputCls} style={{ borderColor: "var(--line)" }} value={r.notes ?? ""} onChange={(e) => updateRow(r.id, { notes: e.target.value })} /></td>
                  <td className="p-1">
                    <div className="flex items-center gap-1">
                      <button onClick={() => removeRow(r.id)} aria-label="Delete row" title="Delete row (from database)" className="card-btn">✕</button>
                      <button onClick={() => toggleCrossed(r.id)} aria-pressed={crossed} title={crossed ? "Un-cross" : "Cross off (keep in database)"} className={`card-btn${crossed ? " card-btn-on" : ""}`}>~</button>
                      <span draggable onDragStart={() => { dragId.current = r.id; }} onDragEnd={() => { dragId.current = null; setDragOverId(null); }}
                        role="button" tabIndex={0} aria-label="Drag to reorder" title="Drag to reorder" className="card-btn" style={{ cursor: "grab" }}>⠿</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
