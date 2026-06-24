"use client";
export function MultiSelect({ label, options, selected, onChange, render }:
  { label: string; options: string[]; selected: string[]; onChange: (v: string[]) => void; render?: (o: string) => string }) {
  const toggle = (o: string) => onChange(selected.includes(o) ? selected.filter((x) => x !== o) : [...selected, o]);
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">{label}</div>
      <div role="group" aria-label={label} className="flex flex-wrap gap-1.5">
        {options.map((o) => (
          <button key={o} type="button" aria-pressed={selected.includes(o)} onClick={() => toggle(o)} className="rounded-full border px-2.5 py-1 text-xs"
            style={selected.includes(o) ? { background: "var(--accent)", color: "var(--accent-ink)", borderColor: "var(--accent)" } : { borderColor: "var(--line)" }}>
            {render ? render(o) : o}
          </button>
        ))}
      </div>
    </div>
  );
}
