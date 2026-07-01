"use client";
import { AMENITY_FIELDS } from "@/lib/constants";
export function AmenityToggles({ selected, onChange }: { selected: string[]; onChange: (v: string[]) => void }) {
  const toggle = (f: string) => onChange(selected.includes(f) ? selected.filter((x) => x !== f) : [...selected, f]);
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">Amenities</div>
      <div className="flex flex-wrap gap-1.5">
        {AMENITY_FIELDS.map(({ field, label }) => (
          <button key={field as string} onClick={() => toggle(field as string)} className="rounded-full border px-2.5 py-1 text-xs"
            style={selected.includes(field as string) ? { background: "var(--accent)", color: "var(--accent-ink)", borderColor: "var(--accent)" } : { borderColor: "var(--line)" }}>{label}</button>
        ))}
      </div>
    </div>
  );
}
