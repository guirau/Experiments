export function Chip({ label, onRemove }: { label: string; onRemove?: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      {label}
      {onRemove && <button onClick={onRemove} aria-label={`Remove ${label}`} className="opacity-60 hover:opacity-100">×</button>}
    </span>
  );
}
