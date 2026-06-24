"use client";
import type { Tri } from "@/lib/types";
const OPTS: Tri[] = ["any", "yes", "no"];
export function TriStateToggle({ label, value, onChange }: { label: string; value: Tri; onChange: (v: Tri) => void }) {
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">{label}</div>
      <div role="group" aria-label={label} className="inline-flex overflow-hidden rounded-lg border" style={{ borderColor: "var(--line)" }}>
        {OPTS.map((o) => (
          <button key={o} type="button" aria-pressed={o === value} onClick={() => onChange(o)} className="px-3 py-1 text-xs capitalize"
            style={o === value ? { background: "var(--accent)", color: "var(--accent-ink)" } : { background: "var(--surface)" }}>{o}</button>
        ))}
      </div>
    </div>
  );
}
