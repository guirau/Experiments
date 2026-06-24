"use client";
export function NumberMax({ label, value, onChange, placeholder = "any" }: { label: string; value: number | null; onChange: (v: number | null) => void; placeholder?: string }) {
  return (
    <div className="mb-3">
      <div className="mb-1 text-sm font-medium">{label}</div>
      <input type="number" inputMode="numeric" placeholder={placeholder} value={value ?? ""} onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} className="w-24 rounded-lg border px-2 py-1 text-sm" style={{ borderColor: "var(--line)" }} />
    </div>
  );
}
