export function EmptyState({ onClear }: { onClear: () => void }) {
  return (
    <div className="rounded-2xl border p-10 text-center" style={{ borderColor: "var(--line)" }}>
      <p className="text-lg font-medium">No listings match these filters</p>
      <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>Try widening your price or area.</p>
      <button onClick={onClear} className="mt-4 rounded-full px-4 py-2 text-sm font-medium" style={{ background: "var(--accent)", color: "var(--accent-ink)" }}>Clear all filters</button>
    </div>
  );
}
