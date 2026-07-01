"use client";
import type { BoolState } from "@/lib/types";

const OPTS: { v: BoolState; label: string }[] = [
  { v: "yes", label: "Yes" },
  { v: "no", label: "No" },
  { v: "unknown", label: "Unknown" },
];

// Yes/No are mutually exclusive; Unknown can combine with either (so Yes+Unknown or
// No+Unknown are allowed, but not Yes+No). Empty selection = no constraint.
export function BoolFilter({ label, value, onChange }:
  { label: string; value: BoolState[]; onChange: (v: BoolState[]) => void }) {
  const toggle = (s: BoolState) => {
    if (value.includes(s)) return onChange(value.filter((x) => x !== s));
    if (s === "unknown") return onChange([...value, s]);
    // selecting yes/no replaces the other yes/no but keeps unknown
    return onChange([...value.filter((x) => x !== "yes" && x !== "no"), s]);
  };
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">{label}</div>
      <div role="group" aria-label={label} className="flex flex-wrap gap-1.5">
        {OPTS.map(({ v, label: l }) => (
          <button key={v} type="button" aria-pressed={value.includes(v)} onClick={() => toggle(v)}
            className="rounded-full border px-2.5 py-1 text-xs"
            style={value.includes(v) ? { background: "var(--accent)", color: "var(--accent-ink)", borderColor: "var(--accent)" } : { borderColor: "var(--line)" }}>
            {l}
          </button>
        ))}
      </div>
    </div>
  );
}
