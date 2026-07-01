"use client";
export function PriceRange({ min, max, onChange }: { min: number | null; max: number | null; onChange: (min: number | null, max: number | null) => void }) {
  const parse = (s: string) => (s === "" ? null : Number(s));
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">Price ฿/month</div>
      <div className="flex items-center gap-2">
        <input type="number" inputMode="numeric" aria-label="Minimum price (THB/month)" placeholder="min" value={min ?? ""} onChange={(e) => onChange(parse(e.target.value), max)} className="w-24 rounded-lg border px-2 py-1 text-sm" style={{ borderColor: "var(--line)" }} />
        <span style={{ color: "var(--muted)" }}>–</span>
        <input type="number" inputMode="numeric" aria-label="Maximum price (THB/month)" placeholder="max" value={max ?? ""} onChange={(e) => onChange(min, parse(e.target.value))} className="w-24 rounded-lg border px-2 py-1 text-sm" style={{ borderColor: "var(--line)" }} />
      </div>
    </div>
  );
}
