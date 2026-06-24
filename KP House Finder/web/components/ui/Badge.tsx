export function Badge({ children, tone = "default" }: { children: React.ReactNode; tone?: "default" | "good" | "muted" }) {
  const bg = tone === "good" ? "var(--good)" : tone === "muted" ? "var(--line)" : "var(--accent)";
  const fg = tone === "muted" ? "var(--muted)" : "var(--accent-ink)";
  return <span style={{ background: bg, color: fg }} className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium">{children}</span>;
}
